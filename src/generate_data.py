"""The goal is to create a dataset for SFT based on critiques and revisions"""
import logging
import time
import argparse

import torch
from transformers import pipeline, AutoTokenizer
from datasets import load_dataset
from constitution import Constitution


MODEL = "meta-llama/Llama-3.1-8B-Instruct"
NUM_REVISIONS = 1
NUM_TO_GENERATE = 2500
NUM_TURNS = 1
BATCH_SIZE = 8
OFFSET = 5000  # Previously generated data points

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


class LLM:
    """Handles all LLM inference operations"""
    
    def __init__(self, model_name, max_new_tokens=512, temperature=0.5):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.pipe = self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        """Initialize the text generation pipeline"""
        logger.info(f"Initializing pipeline for {self.model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        model_kwargs = {"device_map": "auto"}
        # if torch.cuda.is_available():
        #     model_kwargs.update({"attn_implementation": "flash_attention_2"})

        return pipeline(
            "text-generation",
            model=self.model_name,
            tokenizer=tokenizer,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            dtype=torch.float16,
            temperature=self.temperature,
            model_kwargs=model_kwargs,
        )
    
    def generate_batch(self, messages_list):
        try:
            responses = self.pipe(messages_list, batch_size=len(messages_list))
            results = []
            for response in responses:
                if isinstance(response, list) and len(response) > 0:
                    generated_text = response[0].get('generated_text', '')[-1]
                    if generated_text.get("role") == "assistant":
                        results.append(generated_text.get("content"))
                    else:
                        results.append("")
                else:
                    results.append("")
            return results
        except Exception as e:
            logger.error(f"Error generating batch responses: {e}")
            return [""] * len(messages_list)


class CAIProcessor:
    """Handles Constitutional AI critique and revision process"""
    
    CRITIQUE_SYSTEM_PROMPT = """
    You are a helpful critique who identifies potential issues in responses, so that they can be
    improved. It is possible that the response is good how it is. ONLY output your feedback with 
    no prelude or justification.
    """

    REVISION_SYSTEM_PROMPT = """
    You are a helpful editor who makes revisions to responses based on critiques. 
    ONLY output your revised response with no prelude or justification.
    """
    
    def __init__(self, llm, constitution, num_revisions=1):
        self.llm = llm
        self.constitution = constitution
        self.num_revisions = num_revisions

    def _format_dialogue_prompt(self, dialogue, include_revision=False):
        """Format a dialogue into a prompt string"""
        parts = [
            f"question: {dialogue[0]['content']}",
            f"initial response: {dialogue[1]['content']}",
            f"critique request: {dialogue[2]['content']}",
        ]
        
        if include_revision:
            parts.extend([
                f"critique: {dialogue[3]['content']}",
                f"revision request: {dialogue[4]['content']}",
            ])
        
        return "\n".join(parts)

    def _add_few_shot_examples(self, messages, dialogues, include_revision=False):
        """Add few-shot examples to messages list"""
        response_idx = 3 if not include_revision else 5
        
        for dialogue in dialogues:
            prompt = self._format_dialogue_prompt(dialogue, include_revision)
            messages.append({"role": "user", "content": prompt})
            messages.append({"role": "assistant", "content": dialogue[response_idx]['content']})
            
    
    def _build_critique_prompt(self, history, initial_response, critique_request):
        """Build critique prompt with few-shot examples"""
        critique_messages = [{"role": "system", "content": self.CRITIQUE_SYSTEM_PROMPT}]
        
        # Add few-shot examples
        dialogues = self.constitution.few_shot_revisions()
        self._add_few_shot_examples(critique_messages, dialogues, include_revision=False)
        
        # Add current query
        prompt = self._format_dialogue_prompt([
            history[-1],
            {"content": initial_response},
            {"content": critique_request},
        ])
        critique_messages.append({"role": "user", "content": prompt})
        
        return critique_messages

    def _build_revision_prompt(self, history, initial_response, critique_request, critique, revision_request):
        """Build revision prompt with few-shot examples"""
        revision_messages = [{"role": "system", "content": self.REVISION_SYSTEM_PROMPT}]
        
        # Add few-shot examples
        dialogues = self.constitution.few_shot_revisions()
        self._add_few_shot_examples(revision_messages, dialogues, include_revision=True)
        
        # Add current query
        prompt = self._format_dialogue_prompt([
            history[-1],
            {"content": initial_response},
            {"content": critique_request},
            {"content": critique},
            {"content": revision_request},
        ], include_revision=True)
        revision_messages.append({"role": "user", "content": prompt})
        
        return revision_messages
        
    def revise_responses(self, histories, initial_responses):
        """Apply multiple rounds of critique and revision to a batch of responses"""
        batch_size = len(histories)
        results = [
            {
                'critique_requests': [],
                'critiques': [],
                'revision_requests': [],
                'revisions': [],
                'revised_response': None
            }
            for _ in range(batch_size)
        ]
        
        for _ in range(self.num_revisions):
            # Get principles for all examples in batch
            principles = [self.constitution.sample_principle() for _ in range(batch_size)]
            critique_requests = [p["critique"] for p in principles]
            revision_requests = [p["revision"] for p in principles]
            
            # Build critique prompts for batch
            critique_prompts = [
                self._build_critique_prompt(histories[i], initial_responses[i], critique_requests[i])
                for i in range(batch_size)
            ]
            
            # Generate critiques in batch
            critiques = self.llm.generate_batch(critique_prompts)
            
            # Build revision prompts for batch
            revision_prompts = [
                self._build_revision_prompt(
                    histories[i], 
                    initial_responses[i], 
                    critique_requests[i], 
                    critiques[i], 
                    revision_requests[i]
                )
                for i in range(batch_size)
            ]
            
            # Generate revisions in batch
            revisions = self.llm.generate_batch(revision_prompts)
            
            # Update the results data for each batch element
            for i in range(batch_size):
                results[i]['critique_requests'].append(critique_requests[i])
                results[i]['critiques'].append(critiques[i])
                results[i]['revision_requests'].append(revision_requests[i])
                results[i]['revisions'].append(revisions[i])
                results[i]['revised_response'] = revisions[i] 
        
        return results


class ConversationGenerator:
    """Generates multi-turn student-teacher conversations"""
    
    TEACHER_SYSTEM_PROMPT = """
    You are a helpful learning assistant who supports students and helps them learn.
    """
    
    STUDENT_SYSTEM_PROMPT = """
    You are a helpful assistant who tries to emulate reponses that a student would give. 
    The student is trying to learn about a topic, and you can imagine that you are 
    working on a homework or reviewing the material from a class. They are trying to resolve 
    their confusion along with common misconceptions about a concept. Given a conversation, you should
    generate the student's response.
    """

    STUDENT_PROMPT = "What does the student say next?"
    DEFAULT_STUDENT_RESPONSE = "I am still confused"
    
    def __init__(self, llm: LLM, cai_processor: CAIProcessor, num_turns=5):
        self.llm = llm
        self.cai_processor = cai_processor
        self.num_turns = num_turns
        self.few_shot_dialogues = cai_processor.constitution.few_shot_dialogues

    def _format_prompt(self, messages):
        dialogue = "Conversation:"
        for message in messages:
            if message["role"] == "user":
                role = "student"
            else:
                role = "teacher"
            dialogue += f"{role}: {message['content']}\n"

        dialogue += "\n" + self.STUDENT_PROMPT
        
        return dialogue
    
    def _generate_teacher_responses(self, messages_batch):
        """Generate the teacher's response to the question"""
        teacher_prompts = [
            [{"role": "system", "content": self.TEACHER_SYSTEM_PROMPT}] + messages
            for messages in messages_batch
        ]
        return self.llm.generate_batch(teacher_prompts)
    
    def _generate_student_questions(self, messages_batch):
        """Generate follow-up student question based on conversation history"""
        
        # Format few shot examples
        few_shot_examples = []
        for dialogue in self.few_shot_dialogues():
            prompt = self._format_prompt(dialogue[:-1])
            few_shot_examples.append({"role": "user", "content": prompt})
            few_shot_examples.append({"role": "assistant", "content": dialogue[-1]["content"]})

        # Build prompts for batch
        student_prompts = [
            [{"role": "system", "content": self.STUDENT_SYSTEM_PROMPT}] + 
            few_shot_examples + 
            [{"role": "user", "content": self._format_prompt(messages)}]
            for messages in messages_batch
        ]
        
        responses = self.llm.generate_batch(student_prompts)
        return [r if r else self.DEFAULT_STUDENT_RESPONSE for r in responses]
    
    def generate_conversations(self, questions_and_categories):
        batch_size = len(questions_and_categories)
        conversations = [
            {
                'id': abs(hash(q)),
                'question': q,
                'category': cat,
                'messages': [],
            }
            for q, cat in questions_and_categories
        ]

        results = {
            'conversation_id': [],
            'question': [],
            'messages': [],
            'category': [],
            'initial_response': [],
            'critique_requests': [],
            'critiques': [],
            'revision_requests': [],
            'revisions': [],
            'chosen': [],
            'rejected': []
        }
        
        all_results = []
        for turn in range(self.num_turns):
            # Step 1: Get questions for this turn
            if turn == 0:
                questions = [conv['question'] for conv in conversations]
            else:
                # Generate student questions in batch
                messages_list = [conv['messages'] for conv in conversations]
                questions = self._generate_student_questions(messages_list)
            
            # Add questions to conversation histories
            for conv, question in zip(conversations, questions):
                conv['messages'].append({"role": "user", "content": question})
            
            # Step 2: Generate initial teacher responses in batch
            messages_list = [conv['messages'] for conv in conversations]
            initial_responses = self._generate_teacher_responses(messages_list)
            
            # Step 3: Apply CAI revision process in batch
            revision_results = self.cai_processor.revise_responses(messages_list, initial_responses)
            
            # Step 4: Store results and update conversation histories
            for i, conv in enumerate(conversations):
                revised_response = revision_results[i]['revised_response']
                
                # Add revised response to conversation history
                conv['messages'].append({"role": "assistant", "content": revised_response})
                
                # Store this turn's data
                results['conversation_id'].append(conv['id'])
                results['question'].append(questions[i])
                results['messages'].append(conv['messages'].copy())
                results['category'].append(conv['category'])
                results['initial_response'].append(initial_responses[i])
                results['critique_requests'].append(revision_results[i]['critique_requests'])
                results['critiques'].append(revision_results[i]['critiques'])
                results['revision_requests'].append(revision_results[i]['revision_requests'])
                results['revisions'].append(revision_results[i]['revisions'])
                results['chosen'].append(revised_response)
                results['rejected'].append(initial_responses[i])
        
        return results


def process_batch(examples, conversation_generator):
    """Process a batch of examples"""
    questions_and_categories = [
        (examples['text'][i], examples.get('label_text', ['general'] * len(examples['text']))[i])
        for i in range(len(examples['text']))
    ]
    
    # Generate all conversations with batched LLM calls
    return conversation_generator.generate_conversations(questions_and_categories)
    

def main():
    # Get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_id', type=int, default=0)
    parser.add_argument('--num_jobs', type=int, default=1)
    args = parser.parse_args()
    
    logger.info("Starting SFT dataset generation...")
    logger.info(f"Using device: {device}")
    start = time.perf_counter()
    
    # Initialize components
    constitution = Constitution(stage="sl")
    llm = LLM(MODEL, max_new_tokens=512, temperature=0.3)
    cai_processor = CAIProcessor(llm, constitution, num_revisions=NUM_REVISIONS)
    conversation_generator = ConversationGenerator(llm, cai_processor, num_turns=NUM_TURNS)
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_dataset("SetFit/student-question-categories")
    
    # Take a subset for this job
    start_index = (args.job_id * NUM_TO_GENERATE) + OFFSET
    end_index = start_index + NUM_TO_GENERATE
    train_dataset = dataset['train'].select(range(start_index, end_index))
    logger.info(f"Processing {len(train_dataset)} examples...")
    
    # Generate the actual data
    processed_dataset = train_dataset.map(
        lambda examples: process_batch(examples, conversation_generator),
        batched=True,
        batch_size=BATCH_SIZE,
        desc="Generating responses and critiques",
        remove_columns=train_dataset.column_names
    )
        
    # Save backup locally
    logger.info(f"Generated {len(processed_dataset)} examples")
    local_path = f"./sft_dataset_job_{args.job_id}"
    processed_dataset.save_to_disk(local_path)
    logger.info(f"Saved locally to {local_path}")
    
    # Log a summary of the job
    elapsed = time.perf_counter() - start
    datapoints = len(processed_dataset)
    logger.info(
        "JOB SUMMARY | ID: %d | datapoints=%d | rounds_per_conversation=%d | total_time=%.2fs",
        args.job_id, datapoints, NUM_TURNS, elapsed
    )


if __name__ == "__main__":
    main()
    