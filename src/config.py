from dataclasses import dataclass
from typing import Optional, Dict, Any

from peft import LoraConfig

@dataclass
class Config:
	"""Single configuration class for all settings."""
	# Model
	model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
	bf16: bool = True
	
	# Dataset
	dataset_name: str = "aracape/cai-education-single-turn"
	test_size: float = 0.1
	max_length: int = 512
	
	# LoRA
	lora_r: int = 32
	lora_alpha: int = 64
	lora_dropout: float = 0.05
	
	# Training
	output_dir: str = "./results"
	num_epochs: int = 3
	batch_size: int = 4
	gradient_accumulation_steps: int = 4
	learning_rate: float = 1e-4
	warmup_ratio: float = 0.03

	# Evaluation and saving
	eval_steps: int = 200
	save_steps: int = 600
	judge_model: str = "meta-llama/Meta-Llama-3-70B-Instruct"
	
	# DPO specific
	dpo_beta: float = 0.2
	
	# Logging
	use_wandb: bool = True
	wandb_run: str = "default_run_name"
	
	# HuggingFace Hub
	push_to_hub: bool = True
	hub_model_id: Optional[str] = "aracape/teaching-assistant-llm"
	hub_strategy: str = "every_save"
	
	def get_base_training_args(self, **kwargs) -> Dict[str, Any]:
		"""Get common training arguments for all training types."""
		base_args = {
			"output_dir": self.output_dir,
			"num_train_epochs": self.num_epochs,
			"per_device_train_batch_size": self.batch_size,
			"per_device_eval_batch_size": self.batch_size,
			"gradient_accumulation_steps": self.gradient_accumulation_steps,
			"learning_rate": self.learning_rate,
			"warmup_ratio": self.warmup_ratio,
			"eval_strategy": "steps",
			"save_strategy": "steps",
			"eval_steps": self.eval_steps,
			"save_steps": self.save_steps,
			"bf16": self.bf16,
			"report_to": "wandb" if self.use_wandb else "none",
			"run_name": self.wandb_run,
			"load_best_model_at_end": True,
			"max_length": self.max_length,
			"auto_find_batch_size": True,
		}
		
		# Add Hub configuration if enabled
		if self.push_to_hub:
			if not self.hub_model_id:
				raise ValueError("hub_model_id must be set when push_to_hub=True")
			
			base_args.update({
				"push_to_hub": True,
				"hub_model_id": self.hub_model_id,
				"hub_strategy": self.hub_strategy,
			})
		
		# Override with any custom kwargs
		base_args.update(kwargs)
		return base_args

	def lora_config(self) -> Any:
		return LoraConfig(
			r=self.lora_r,
			lora_alpha=self.lora_alpha,
			lora_dropout=self.lora_dropout,
			target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
						"gate_proj", "up_proj", "down_proj"],
			bias="none",
			task_type="CAUSAL_LM",

		)

def default_config(method: str) -> Config:
	"""Return default configuration based on the training method."""
	config = Config()
	
	match method:
		case "sft":
			config.wandb_run = "sft_finetune"
			config.output_dir = "./results/sft"
			config.hub_model_id = "aracape/teaching-assistant-8B-sft"
		case "dpo":
			config.wandb_run = "dpo_finetune"
			config.output_dir = "./results/dpo"
			config.hub_model_id = "aracape/teaching-assistant-8B-dpo"
		case "rm":
			config.model_name = "Qwen/Qwen3-0.6B"
			config.wandb_run = "rm_finetune"
			config.output_dir = "./results/rm"
			config.hub_model_id = "aracape/teaching-assistant-0.6B-rm"
		case "grpo":
			config.wandb_run = "grpo_finetune"
			config.output_dir = "./results/grpo"
			config.hub_model_id = "aracape/teaching-assistant-8B-grpo"
		case _:
			raise ValueError(f"Unknown method: {method}")
	
	return config

def testing_config(method: str) -> Config:
	"""Return configuration for testing purposes."""
	config = default_config(method)
	config.model_name = "meta-llama/Llama-3.2-1B-Instruct"
	config.bf16 = False
	config.use_wandb = False
	config.num_epochs = 1
	config.push_to_hub = False

	return config
