"""The goal is to create a dataset for SFT based on critiques and revisions"""
import logging
from tqdm import tqdm
import pandas as pd
from transformers import pipeline
from datasets import load_dataset, Dataset
from constitution import Constitution


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MODEL = "meta-llama/Llama-3.2-1B-Instruct"
NEW_DATASET = "aracape/education-cai-sft"
NUM_REVISIONS = 2


def initialize_pipeline():
        """Initialize the text generation pipeline"""
        return pipeline(
                "text-generation", 
                model=MODEL,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                pad_token_id=50256
        )
    

def get_response(pipe, messages):
        try:
                response = pipe(messages)
                if isinstance(response, list) and len(response) > 0:
                        generated_text = response[0].get('generated_text', '')[-1]
                        assert generated_text.get("role") == "assistant"
                        return generated_text.get("content")
                return ""
        except Exception as e:
                logger.error(f"Error generating response: {e}")
                return ""


def generate_initial_response(pipe, question, category):
        """Generate initial response to the question"""
        messages = [
                {"role": "system", "content": f"You are a helpful learning assistant answering questions about {category}."},
                {"role": "user", "content": question}
        ]
        
        return get_response(pipe, messages)


def generate_critique(pipe, question, response, critique_request):
        """Generate critique of the response using constitutional principles"""

        messages = [
                {"role": "system", "content": "You are a helpful critic who identifies potential issues in responses."},
                {"role": "user", "content": question},
                {"role": "assistant", "content": response},
                {"role": "critique request", "content": critique_request}
        ]
        
        return get_response(pipe, messages)


def generate_revision(pipe, question, initial_response, critique, revision_request):
        """Generate revised response based on the critique"""

        messages = [
                {"role": "system", "content": "You are a helpful assistant who improves responses based on feedback."},
                {"role": "user", "content": question},
                {"role": "assistant", "content": initial_response},
                {"role": "critique", "content": critique},
                {"role": "revision request", "content": revision_request}
        ]

        return get_response(pipe, messages)
     

def process_example(example, pipe, constitution):
        """Process a single example to generate initial response, critique, and revision"""
        question = example.get('text', '')
        category = example.get('label_text', 'general')
        
        # Generate initial response
        initial_response = generate_initial_response(pipe, question, category)
        
        # Iteratively improve it
        revised_response = initial_response
        for _ in range(NUM_REVISIONS):
                
                # Get principle for this reviison
                principle = constitution.sample_principle()
                critique_request, revision_request = principle["critique"], principle["revision"]
                
                # Generate critique
                critique = generate_critique(pipe, question, revised_response, critique_request)
                
                # Generate revision
                revised_response = generate_revision(pipe, question, revised_response, critique, revision_request)
        
        return {
                'question': question,
                'category': category,
                'initial_response': initial_response,
                'critique': critique,
                'revised_response': revised_response,
                'chosen': revised_response,  # Assuming revised is better
                'rejected': initial_response
        }

def process_batch(examples, pipe, constitution):
        """Process a batch of examples"""
        results = {
                'question': [],
                'category': [],
                'initial_response': [],
                'critique': [],
                'revised_response': [],
                'chosen': [],
                'rejected': []
        }
        
        for i in range(len(examples['text'])):
                example = {
                        'text': examples['text'][i],
                        'label_text': examples.get('label_text', ['general'] * len(examples['text']))[i]
                }
                
                result = process_example(example, pipe, constitution)
                
                for key in results.keys():
                        results[key].append(result[key])
        
        return results


def main():
        logger.info("Starting SFT dataset generation...")
        constitution = Constitution(stage="sl")
        pipe = initialize_pipeline()
        
        # Load dataset
        logger.info("Loading dataset...")
        dataset = load_dataset("SetFit/student-question-categories")
        
        # Take a subset for testing (remove this for full dataset)
        train_dataset = dataset['train'].select(range(10))  # Start with 100 examples
        logger.info(f"Processing {len(train_dataset)} examples...")
        processed_dataset = train_dataset.map(
                lambda examples: process_batch(examples, pipe, constitution),
                batched=True,
                batch_size=10,
                desc="Generating responses and critiques",
                remove_columns=train_dataset.column_names  # Remove original columns
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
