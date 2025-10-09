"""The goal is to create a dataset for SFT based on critiques and revisions"""
import logging
import time

import torch
from transformers import pipeline
from datasets import load_dataset
from constitution import Constitution


MODEL = "meta-llama/Llama-3.1-8B-Instruct"
NEW_DATASET = "aracape/cai-education"
NUM_REVISIONS = 1
NUM_TO_GENERATE = 100
NUM_TURNS = 5
BATCH_SIZE = 1

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    device = 0  # Use first CUDA GPU
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = -1


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
        return pipeline(
            "text-generation",
            model=self.model_name,
            device=device,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            dtype=torch.float16,
        )
    
    def generate(self, messages):
        """Generate response from messages"""
        try:
            response = self.pipe(messages)
            if isinstance(response, list) and len(response) > 0:
                generated_text = response[0].get('generated_text', '')[-1]
                assert generated_text.get("role") == "assistant"
                return generated_text.get("content")
            return ""
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return ""


class CAIProcessor:
    """Handles Constitutional AI critique and revision process"""
    
    CRITIQUE_SYSTEM_PROMPT = """
    You are a helpful critique who identifies potential issues in responses, so that they can be
    improved. ONLY output your critiques with no prelude or justification.
    """

    REVISION_SYSTEM_PROMPT = """
    You are a helpful editor who makes revisions to responses based on critiques. 
    ONLY output your critiques with no prelude or justification.
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

    def _generate_critique(self, history, initial_response, critique_request):
        """Format chat history and few-shot examples into complete prompt"""
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
        
        return self.llm.generate(critique_messages)

    def _generate_revision(self, history, initial_response, critique_request, critique, revision_request):
        """Format chat history and few-shot examples into complete prompt"""
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
        
        return self.llm.generate(revision_messages)
    
    def revise_response(self, history, initial_response):
        """Apply multiple rounds of critique and revision"""
        revised_response = initial_response
        critique_requests, critiques = [], []
        revision_requests, revisions = [], []
        
        for _ in range(self.num_revisions):

            # Get principle for this revision
            principle = self.constitution.sample_principle()
            critique_request = principle["critique"]
            revision_request = principle["revision"]
            
            # Generate critique
            critique = self._generate_critique(history, initial_response, critique_request)
            critique_requests.append(critique_request)
            critiques.append(critique)
            
            # Generate revision
            revised_response = self._generate_revision(history, initial_response, critique_request, critique, revision_request)
            revision_requests.append(revision_request)
            revisions.append(revised_response)
        
        return {
            'revised_response': revised_response,
            'critique_requests': critique_requests,
            'critiques': critiques,
            'revision_requests': revision_requests,
            'revisions': revisions
        }


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

    STUDENT_PROMPT = """
    What does the student say next?
    """

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
    
    def _generate_teacher_response(self, messages):
        """Generate the teacher's response to the question"""
        teacher_messages = [
            {"role": "system", "content": self.TEACHER_SYSTEM_PROMPT}
        ] + messages
        return self.llm.generate(teacher_messages)
    
    def _generate_student_question(self, messages):
        """Generate follow-up student question based on conversation history"""
        
        # Format few shot examples
        few_shot_examples = []
        for dialogue in self.few_shot_dialogues():
            prompt = self._format_prompt(dialogue[:-1])
            few_shot_examples.append({"role": "user", "content": prompt})
            few_shot_examples.append({"role": "assistant", "content": dialogue[-1]["content"]})

        # Stitch together the full conversation context
        student_messages = [
            {"role": "system", "content": self.STUDENT_SYSTEM_PROMPT}
        ] + few_shot_examples + [
            {"role": "user", "content": self._format_prompt(messages)}
        ]
        
        response = self.llm.generate(student_messages)
        return response if response else self.DEFAULT_STUDENT_RESPONSE
    
    def generate_conversation(self, initial_question, category):
        """Generate a multi-turn conversation with CAI refinement"""
        id = hash(initial_question)
        results = []
        messages = []
        
        for turn in range(self.num_turns):
            # Get the current question
            if turn == 0:
                question = initial_question
            else:
                question = self._generate_student_question(messages)
            
            # Add student question to history
            messages.append({"role": "user", "content": question})
            
            # Generate initial teacher response
            initial_response = self._generate_teacher_response(messages)
            
            # Apply CAI revision process
            revision_data = self.cai_processor.revise_response(messages, initial_response)
            revised_response = revision_data['revised_response']
            
            # Add revised response to conversation history
            messages.append({"role": "assistant", "content": revised_response})
            
            # Store this turn's data
            results.append({
                'conversation_id': id,
                'question': question,
                'messages': messages.copy(),
                'category': category,
                'initial_response': initial_response,
                'critique_requests': revision_data['critique_requests'],
                'critiques': revision_data['critiques'],
                'revision_requests': revision_data['revision_requests'],
                'revisions': revision_data['revisions'],
                'chosen': revised_response,
                'rejected': initial_response
            })
        
        return results


def process_example(example, conversation_generator):
    """Process a single example to generate a conversation"""
    question = example.get('text', '')
    category = example.get('category', 'general')
    
    return conversation_generator.generate_conversation(question, category)


def process_batch(examples, conversation_generator):
    """Process a batch of examples"""
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
    
    for i in range(len(examples['text'])):
        example = {
            'text': examples['text'][i],
            'category': examples.get('label_text', ['general'] * len(examples['text']))[i]
        }
        
        turn_results = process_example(example, conversation_generator)
        
        # Flatten results from all turns
        for turn_result in turn_results:
            for key in results.keys():
                results[key].append(turn_result[key])
    
    return results


def main():
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
    
    # Take a subset for testing
    train_dataset = dataset['train'].select(range(NUM_TO_GENERATE))
    logger.info(f"Processing {len(train_dataset)} examples...")
    
    processed_dataset = train_dataset.map(
        lambda examples: process_batch(examples, conversation_generator),
        batched=True,
        batch_size=BATCH_SIZE,
        desc="Generating responses and critiques",
        remove_columns=train_dataset.column_names
    )
    
    logger.info(f"Generated {len(processed_dataset)} examples")
    
    # Save backup locally
    processed_dataset.save_to_disk("./sft_dataset_local")
    
    # Push to Hugging Face Hub
    logger.info("Pushing to Hugging Face Hub...")
    try:
        processed_dataset.push_to_hub(NEW_DATASET, private=False)
        logger.info(f"Successfully pushed dataset to {NEW_DATASET}")
    except Exception as e:
        logger.error(f"Error pushing to hub: {e}")
        logger.info("Dataset saved locally in ./sft_dataset_local")
    finally:
        elapsed = time.perf_counter() - start
        datapoints = len(processed_dataset)
        logger.info(
            "JOB SUMMARY | datapoints=%d | rounds_per_conversation=%d | total_time=%.2fs",
            datapoints, NUM_TURNS, elapsed
        )


if __name__ == "__main__":
    main()
    