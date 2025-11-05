import re
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
import numpy as np


logger = logging.getLogger(__name__)

@dataclass
class EvalExample:
	"""Single evaluation example with prompt and optional reference."""
	prompt: str
	reference_answer: Optional[str] = None
	category: Optional[str] = None  # For tracking different question types


class TeachingEvaluator:
	"""
	LLM-as-Judge evaluator for interactive teaching quality.
	Compatible with HuggingFace Trainer API.
	"""
	
	def __init__(
		self,
		judge_model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
		eval_examples: List[EvalExample] = None,
		device: str = "cuda" if torch.cuda.is_available() else "cpu",
		rubric: Optional[str] = None,
		batch_size: int = 1,
	):
		"""
		Args:
			judge_model_name: HuggingFace model path for the judge
			eval_examples: List of evaluation prompts to test on
			device: Device to run judge model on
			rubric: Custom rubric for evaluation (uses default if None)
			batch_size: Batch size for generating responses (not for judging)
		"""
		logger.info(f"Loading judge model: {judge_model_name}")
		self.judge_tokenizer = AutoTokenizer.from_pretrained(judge_model_name)
		self.judge_model = AutoModelForCausalLM.from_pretrained(
			judge_model_name,
			dtype=torch.bfloat16,
			device_map=device,
		)
		self.judge_model.eval()
		
		self.eval_examples = eval_examples or self._default_examples()
		self.device = device
		self.batch_size = batch_size
		self.rubric = rubric or self._default_rubric()
		
	def _default_rubric(self) -> str:
		"""Default rubric for interactive teaching evaluation."""
		return """You are evaluating a teaching assistant's response quality. Rate the response on a scale of 1-10 based on these criteria:

SCORING RUBRIC:
- Questions & Engagement (3 points): Does it ask thoughtful questions that guide learning? Avoids giving direct answers?
- Pedagogical Quality (3 points): Are hints helpful and appropriately scaffolded? Does it build understanding step-by-step?
- Accuracy (2 points): Is the underlying information factually correct?
- Appropriateness (2 points): Is the teaching approach suitable for the question? Not overly Socratic when a direct answer is needed?

SCORING GUIDE:
9-10: Excellent teaching - insightful questions, perfect scaffolding, accurate
7-8: Good teaching - helpful guidance, mostly appropriate approach
5-6: Adequate - some useful elements but could be more effective
3-4: Poor - unhelpful questions or inappropriate approach
1-2: Very poor - confusing, inaccurate, or completely wrong approach

Provide your score as a single integer from 1-10 at the end of your evaluation in the format: SCORE: X"""

	def _default_examples(self) -> List[EvalExample]:
		"""Default evaluation examples for interactive teaching."""
		return [
			EvalExample(
				prompt="I don't understand why seasons happen. Can you help me learn?",
				category="conceptual"
			),
			EvalExample(
				prompt="I'm stuck on this problem: If f(x) = 2x + 3, what is f(5)?",
				category="math"
			),
			EvalExample(
				prompt="What causes photosynthesis? I need to understand this for my test.",
				category="science"
			),
			EvalExample(
				prompt="Can you explain the difference between supervised and unsupervised learning?",
				category="technical"
			),
			EvalExample(
				prompt="I'm confused about why the French Revolution started.",
				category="history"
			),
		]
	
	def generate_response(
		self,
		model,
		tokenizer,
		prompt: str,
		max_new_tokens: int = 256,
	) -> str:
		"""Generate response from the model being evaluated."""
		inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
		
		with torch.no_grad():
			outputs = model.generate(
				**inputs,
				max_new_tokens=max_new_tokens,
				do_sample=True,
				temperature=0.7,
				top_p=0.9,
				pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
			)
		
		response = tokenizer.decode(outputs[0], skip_special_tokens=True)
		# Remove the prompt from response if it's included
		if response.startswith(prompt):
			response = response[len(prompt):].strip()
		
		return response
	
	def judge_response(self, prompt: str, response: str) -> Dict[str, float]:
		"""Use judge model to score a response."""
		
		judge_prompt = f"""{self.rubric}

STUDENT QUESTION:
{prompt}

TEACHING ASSISTANT RESPONSE:
{response}

Evaluate this teaching response according to the rubric above. Provide detailed reasoning, then end with your score in the format "SCORE: X" where X is 1-10."""

		# Format for chat models
		messages = [{"role": "user", "content": judge_prompt}]
		judge_input = self.judge_tokenizer.apply_chat_template(
			messages,
			tokenize=False,
			add_generation_prompt=True
		)
		
		inputs = self.judge_tokenizer(judge_input, return_tensors="pt").to(self.device)
		
		with torch.no_grad():
			outputs = self.judge_model.generate(
				**inputs,
				max_new_tokens=512,
				do_sample=False,  # Deterministic for evaluation
				pad_token_id=self.judge_tokenizer.pad_token_id or self.judge_tokenizer.eos_token_id,
			)
		
		judgment = self.judge_tokenizer.decode(outputs[0], skip_special_tokens=True)
		logger.debug(judgment)
		
		# Extract score
		score = self._extract_score(judgment)
		
		return {
			"score": score,
			"judgment": judgment,
		}
	
	def _extract_score(self, judgment: str) -> float:
		"""Extract numerical score from judge's response."""
		# Look for "SCORE: X" pattern
		match = re.search(r'SCORE:\s*(\d+(?:\.\d+)?)', judgment, re.IGNORECASE)
		if match:
			score = float(match.group(1))
			return max(1.0, min(10.0, score))  # Clamp between 1-10
		
		# Fallback: look for any number between 1-10
		numbers = re.findall(r'\b([1-9]|10)\b', judgment)
		if numbers:
			return float(numbers[-1])  # Take last number found
		
		logger.warning(f"Warning: Could not extract score from judgment. Defaulting to 5.0")
		return 5.0
	
	def evaluate(
		self,
		model,
		tokenizer,
		num_examples: Optional[int] = None,
	) -> Dict[str, float]:
		"""
		Evaluate model on the eval examples.
		
		Args:
			model: Model to evaluate
			tokenizer: Tokenizer for the model
			num_examples: Number of examples to evaluate (uses all if None)
		
		Returns:
			Dictionary with evaluation metrics
		"""
		examples = self.eval_examples[:num_examples] if num_examples else self.eval_examples
		
		scores = []
		category_scores = {}
		
		logger.info(f"Evaluating on {len(examples)} examples...")
		
		for i, example in enumerate(examples):
			logger.info(f"  Evaluating {i+1}/{len(examples)}: {example.prompt[:50]}...")
			
			# Generate response from model being evaluated
			response = self.generate_response(model, tokenizer, example.prompt)
			
			# Judge the response
			result = self.judge_response(example.prompt, response)
			score = result["score"]
			scores.append(score)
			
			# Track by category
			if example.category:
				if example.category not in category_scores:
					category_scores[example.category] = []
				category_scores[example.category].append(score)
			
			logger.info(f"    Score: {score:.1f}/10")
		
		# Compute metrics
		metrics = {
			"teaching_quality_score": np.mean(scores),
			"teaching_quality_std": np.std(scores),
			"teaching_quality_min": np.min(scores),
			"teaching_quality_max": np.max(scores),
		}
		
		# Add category-specific metrics
		for category, cat_scores in category_scores.items():
			metrics[f"teaching_quality_{category}"] = np.mean(cat_scores)
		
		return metrics


class TeachingEvalCallback(TrainerCallback):
	"""
	Custom callback for HuggingFace Trainer to run LLM-as-judge evaluation.
	Use this with Trainer(callbacks=[TeachingEvalCallback(evaluator)])
	"""
	
	def __init__(self, evaluator: TeachingEvaluator, num_examples: int):
		"""
		Args:
			evaluator: TeachingQualityEvaluator instance
			eval_steps: Run evaluation every N training steps
		"""
		self.evaluator = evaluator
		self.num_examples = num_examples
		
	def on_evaluate(self, args, state, control, **kwargs):
		"""Called at the end of eval"""
		print(f"\n{'='*60}")
		print(f"Running teaching quality evaluation at step {state.global_step}")
		print(f"{'='*60}")

		model = kwargs.get('model')
		tokenizer = kwargs.get('processing_class')
		if not model or not tokenizer:
			logger.warning("Issue loading model and tokenizer")
			logger.debug(f"Kwargs: {kwargs}")
			return control
		
		model.eval()
		metrics = self.evaluator.evaluate(model, tokenizer, self.num_examples)
		model.train()
		
		# Log metrics
		for key, value in metrics.items():
			logger.info(f"{key}: {value:.3f}")
		
		# Log to wandb if available
		try:
			import wandb
			if wandb.run is not None:
				wandb.log({
					**metrics,
					"teaching_eval_step": state.global_step,
				}, step=state.global_step)
		except ImportError:
			pass  # wandb not installed
			
		# Add to trainer logs for tracking
		if hasattr(state, 'log_history'):
			state.log_history.append({
				'step': state.global_step,
				**metrics
			})
				
		print(f"{'='*60}\n")

		return control


if __name__ == "__main__":

	# Example test usage
	from transformers import AutoModelForCausalLM, AutoTokenizer
	
	model_name = "meta-llama/Llama-3.2-1B-Instruct"  # Your model being fine-tuned
	model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="auto")
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	tokenizer.pad_token = tokenizer.eos_token
	
	# Create evaluator
	evaluator = TeachingEvaluator(
		judge_model_name="meta-llama/Llama-3.2-1B-Instruct",
		eval_examples=[
			EvalExample("I don't understand why the sky is blue. Can you help?", category="science"),
			EvalExample("What's the derivative of x^2?", category="math"),
		]
	)
	
	# Run evaluation
	results = evaluator.evaluate(model, tokenizer)
	print("\nEvaluation Results:")
	for key, value in results.items():
		print(f"{key}: {value:.3f}")
	