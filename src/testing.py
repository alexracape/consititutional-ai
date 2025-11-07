import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
import numpy as np

from judging import Judge, JudgeResult


logger = logging.getLogger(__name__)


@dataclass
class EvalExample:
	"""Single evaluation example with prompt and optional reference."""

	prompt: str
	reference_answer: Optional[str] = None
	category: Optional[str] = None  # For tracking different question types


class TeachingEvaluator:
	def __init__(
		self,
		judge_backend: Judge,
		eval_examples: Optional[List[EvalExample]] = None,
		rubric: Optional[str] = None,
	):
		self.backend = judge_backend
		self.eval_examples = eval_examples or self._default_examples()
		self.rubric = rubric or self._default_rubric()

	def _default_rubric(self) -> str:
		return """You are evaluating a teaching assistant's response quality. Rate the response on a scale of 1-10 based on these criteria.
You can assume that this response is the start of a dialogue with the student.:

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
		return [
			EvalExample("I don't understand why seasons happen. Can you help me learn?", category="conceptual"),
			EvalExample("I'm stuck: If f(x)=2x+3, what is f(5)?", category="math"),
			EvalExample("What causes photosynthesis?", category="science"),
			EvalExample("Explain supervised vs unsupervised learning.", category="technical"),
			EvalExample("Why did the French Revolution start?", category="history"),
		]

	def generate_response(self, model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
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
		text = tokenizer.decode(outputs[0], skip_special_tokens=True)
		return text[len(prompt) :].strip() if text.startswith(prompt) else text

	def judge_response(self, prompt: str, response: str) -> Dict[str, float]:
		system = "You are a strict but fair teaching-quality judge. Follow the rubric exactly and end with 'SCORE: X'."
		user = f"""{self.rubric}

STUDENT QUESTION:
{prompt}

TEACHING ASSISTANT RESPONSE:
{response}

Evaluate this teaching response according to the rubric above. Provide detailed reasoning, then end with your score in the format "SCORE: X" where X is 1-10."""
		messages = [
			{"role": "system", "content": system},
			{"role": "user", "content": user},
		]
		res = self.backend.judge(messages)
		return {"score": res.score, "judgment": res.judgment}

	def evaluate(
		self, model, tokenizer, num_examples: Optional[int] = None
	) -> Dict[str, float]:
		examples = (
			self.eval_examples[:num_examples] if num_examples else self.eval_examples
		)
		details = {
			"prompts": [],
			"responses": [],
			"judgments": [],
			"scores": [],
		}
		scores, by_cat = [], {}
		for ex in examples:
			resp = self.generate_response(model, tokenizer, ex.prompt)
			r = self.judge_response(ex.prompt, resp)
			s = r["score"]
			scores.append(s)
			if ex.category:
				by_cat.setdefault(ex.category, []).append(s)
    
			details["prompts"].append(ex.prompt)
			details["responses"].append(resp)
			details["judgments"].append(r["judgment"])
			details["scores"].append(s)

		scores = np.array(details["scores"])
		metrics = {
			"teaching_quality_score": float(np.mean(scores)),
			"teaching_quality_std": float(np.std(scores)),
			"teaching_quality_min": float(np.min(scores)),
			"teaching_quality_max": float(np.max(scores)),
		}
		for cat, vals in by_cat.items():
			metrics[f"teaching_quality_{cat}"] = float(np.mean(vals))
   
		return metrics, details


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

		model = kwargs.get("model")
		tokenizer = kwargs.get("processing_class")
		if not model or not tokenizer:
			logger.warning("Issue loading model and tokenizer")
			logger.debug(f"Kwargs: {kwargs}")
			return control

		metrics = {}
		judgement_details = {}
		model.eval()
		try:
			metrics, judgement_details = self.evaluator.evaluate(model, tokenizer, self.num_examples)
		except Exception as e:
			logger.error(f"Error during teaching quality evaluation: {e}")
		model.train()

		# Log metrics
		for key, value in metrics.items():
			logger.info(f"{key}: {value:.3f}")

		# Log to wandb if available
		try:
			import wandb

			if wandb.run is not None:
				wandb.log(
					{
						**metrics,
						"teaching_eval_step": state.global_step,
					},
					step=state.global_step,
				)
				wandb.log(judgement_details, step=state.global_step)
		except ImportError:
			pass  # wandb not installed

		# Add to trainer logs for tracking
		if hasattr(state, "log_history"):
			state.log_history.append({"step": state.global_step, **metrics})

		print(f"{'='*60}\n")

		return control


if __name__ == "__main__":

	# Example test usage
	from transformers import AutoModelForCausalLM, AutoTokenizer
	from judging import HFJudge

	model_name = "meta-llama/Llama-3.2-1B-Instruct"
	model = AutoModelForCausalLM.from_pretrained(
		model_name, dtype=torch.bfloat16, device_map="auto"
	)
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	tokenizer.pad_token = tokenizer.eos_token

	# Create evaluator
	judge = HFJudge(model="meta-llama/Meta-Llama-3-70B-Instruct")
	evaluator = TeachingEvaluator(judge)

	# Run evaluation
	metrics, details = evaluator.evaluate(model, tokenizer, num_examples=1)
	print("\nEvaluation Results:")
	for k, v in metrics.items():
		print(f"{k}: {v:.3f}")
	for k, v in details.items():
		print(f"{k}: {v[0]}")
