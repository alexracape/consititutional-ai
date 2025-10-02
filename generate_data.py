"""The goal is to create a dataset for SFT based on critiques and revisions"""
import logging
from tqdm import tqdm
import pandas as pd
from transformers import pipeline
from datasets import load_dataset, Dataset
from constitution import Constitution


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
NEW_DATASET = "aracape/cai-education"
NUM_REVISIONS = 1
NUM_TO_GENERATE = 3
NUM_TURNS = 4
BATCH_SIZE = 3


class LLM:
    """Handles all LLM inference operations"""
    
    def __init__(self, model_name, max_new_tokens=512, temperature=0.6, pad_token_id=50256):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.pad_token_id = pad_token_id
        self.pipe = self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        """Initialize the text generation pipeline"""
        logger.info(f"Initializing pipeline for {self.model_name}...")
        return pipeline(
            "text-generation",
            model=self.model_name,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            pad_token_id=self.pad_token_id
        )
    
    def generate(self, messages, continue_final_message=False):
        """Generate response from messages"""
        try:
            response = self.pipe(messages, continue_final_message=continue_final_message)
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
    You are a helpful critic who identifies potential issues in responses. Output purely your 
    critiques with no other introduction.
    """
    
    REVISION_SYSTEM_PROMPT = """
    You are a helpful assistant who improves responses based on feedback. Keep answers succinct and 
    output purely your revised answer without any introduction or summary.
    """
    
    def __init__(self, llm, constitution, num_revisions=1):
        self.llm = llm
        self.constitution = constitution
        self.num_revisions = num_revisions
    
    def _generate_critique(self, history, response, critique_request):
        """Generate critique of the response using constitutional principles"""
        messages = [
            {"role": "system", "content": self.CRITIQUE_SYSTEM_PROMPT}
        ] + self.constitution.few_shot_revisions() + history + [
            {"role": "assistant", "content": response},
            {"role": "user", "content": f"Critique request: {critique_request}"}
        ]
        return self.llm.generate(messages)
    
    def _generate_revision(self, history, initial_response, critique, revision_request):
        """Generate revised response based on the critique"""
        messages = [
            {"role": "system", "content": self.REVISION_SYSTEM_PROMPT}
        ] + self.constitution.few_shot_revisions() + history + [
            {"role": "assistant", "content": initial_response},
            {"role": "user", "content": f"Critique: {critique}\n\nRevision request: {revision_request}"}
        ]
        return self.llm.generate(messages)
    
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
            critique = self._generate_critique(history, revised_response, critique_request)
            critique_requests.append(critique_request)
            critiques.append(critique)
            
            # Generate revision
            revised_response = self._generate_revision(
                history, revised_response, critique, revision_request
            )
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
    You are a student who is trying to learn about a topic. You can imagine that you are 
    working on a homework or reviewing the material from a class, and you are trying to resolve 
    your confusion along with common misconceptions about a concept. Given a conversation, you should
    generate the student's response.
    """
    
    def __init__(self, llm: LLM, cai_processor: CAIProcessor, num_turns=5):
        self.llm = llm
        self.cai_processor = cai_processor
        self.num_turns = num_turns
        self.few_shot_dialogues = cai_processor.constitution.few_shot_dialogues
    
    def _generate_teacher_response(self, messages):
        """Generate the teacher's response to the question"""
        teacher_messages = [
            {"role": "system", "content": self.TEACHER_SYSTEM_PROMPT}
        ] + messages
        return self.llm.generate(teacher_messages)
    
    def _generate_student_question(self, messages):
        """Generate follow-up student question based on conversation history"""
        student_messages = [
            {"role": "system", "content": self.STUDENT_SYSTEM_PROMPT}
        ] + self.few_shot_dialogues() + messages + [
            {"role": "user", "content": ""}
        ]
        return self.llm.generate(student_messages, continue_final_message=True)
    
    def generate_conversation(self, initial_question, category):
        """Generate a multi-turn conversation with CAI refinement"""
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
    
    # Initialize components
    constitution = Constitution(stage="sl")
    llm = LLM(MODEL, max_new_tokens=512, temperature=0.3, pad_token_id=50256)
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


if __name__ == "__main__":
    main()
    