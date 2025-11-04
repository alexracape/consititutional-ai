"""
Streamlined Fine-tuning Pipeline for SFT and DPO
Supports preference datasets with 'messages' column format
"""

import torch
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from trl import (
    SFTTrainer, 
    SFTConfig,
    DPOTrainer,
    DPOConfig, 
    GRPOTrainer,
    GRPOConfig,
    RewardTrainer,
    RewardConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from testing import TeachingEvalCallback, TeachingEvaluator


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Single configuration class for all settings."""
    # Model
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    
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
    eval_steps: int = 500
    save_steps: int = 500
    
    # DPO specific
    dpo_beta: float = 0.2
    
    # Logging
    use_wandb: bool = True
    wandb_run: str = "default_run_name"
    
    # HuggingFace Hub
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    hub_strategy: str = "end"
    
    def get_base_training_args(self, output_subdir: str, **kwargs) -> Dict[str, Any]:
        """Get common training arguments for all training types."""
        base_args = {
            "output_dir": f"{self.output_dir}/{output_subdir}",
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
            "bf16": True,
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


def load_model_and_tokenizer(config: Config):
    """Load model with 8-bit quantization and tokenizer."""
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    
    # Prepare for training
    model = prepare_model_for_kbit_training(model)
    
    # Add LoRA adapters
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                       "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer


def prepare_datasets(config: Config, tokenizer, for_dpo: bool = False):
    """Load and prepare datasets."""

    def format_for_sft(examples):
        texts = []
        for messages in examples["messages"]:
            # Apply chat template to the conversation
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            texts.append(text)
        return {"text": texts}

    # Load dataset
    dataset = load_dataset(config.dataset_name, split="train")
    split_dataset = dataset.train_test_split(test_size=config.test_size, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    if not for_dpo:
        # For SFT: extract chosen responses from messages
        train_dataset = train_dataset.map(
            format_for_sft,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        eval_dataset = eval_dataset.map(
            format_for_sft,
            batched=True,
            remove_columns=eval_dataset.column_names
        )
    
    return train_dataset, eval_dataset


def train_sft(config: Config):
    """Run Supervised Fine-Tuning."""
    print("=" * 60)
    print("Starting Supervised Fine-Tuning (SFT)")
    print("=" * 60)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)
    
    # Prepare datasets
    train_dataset, eval_dataset = prepare_datasets(config, tokenizer, for_dpo=False)

    # Load custom evaluator
    evaluator = TeachingEvaluator(judge_model_name=config.model_name)
    
    # Get base training args and add SFT-specific settings
    training_args = SFTConfig(
        **config.get_base_training_args(
            output_subdir="sft",
            chat_template_path=config.model_name,
        )
    )
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[
            TeachingEvalCallback(evaluator, num_examples=5)
        ]
    )

    logger.info(f"Trainer is using device: {trainer.args.device}")
    
    # Train
    trainer.train()
    
    # Save locally
    output_path = f"{config.output_dir}/sft/final"
    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)
    
    logger.info(f"\n✓ SFT completed! Model saved to: {output_path}")
    if config.push_to_hub:
        logger.info(f"✓ Model uploaded to HF Hub: {config.hub_model_id}")
    
    return output_path


def train_dpo(config: Config, sft_model_path: Optional[str] = None):
    """Run Direct Preference Optimization."""
    print("=" * 60)
    print("Starting Direct Preference Optimization (DPO)")
    print("=" * 60)
    
    # Determine which model to start from
    model_path = sft_model_path if sft_model_path else config.model_name
    logger.info(f"Starting from: {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load model
    if sft_model_path:
        # Load from SFT checkpoint (already has LoRA)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            dtype=torch.bfloat16,
        )
    else:
        # Load base model and add LoRA
        model, tokenizer = load_model_and_tokenizer(config)
    
    # Prepare datasets (DPO format - keeps prompt, chosen, rejected)
    train_dataset, eval_dataset = prepare_datasets(config, tokenizer, for_dpo=True)

    # Load evaluator
    evaluator = TeachingEvaluator(judge_model_name=config.model_name) 
    
    # Get base training args and add DPO-specific settings
    dpo_config = DPOConfig(
        **config.get_base_training_args(
            output_subdir="dpo",
            beta=config.dpo_beta,
            max_prompt_length=config.max_length // 2,
        )
    )
    
    # Initialize DPO trainer
    trainer = DPOTrainer(
        model=model,
        processing_class=tokenizer,
        ref_model=None,  # Will create reference model automatically
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[
            TeachingEvalCallback(evaluator, config.eval_steps, num_eval_examples=5)
        ]
    )

    logger.info(f"Trainer is using device: {trainer.args.device}")
    
    # Train
    trainer.train()
    
    # Save locally
    output_path = f"{config.output_dir}/dpo/final"
    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)
    
    logger.info(f"\n✓ DPO completed! Model saved to: {output_path}")
    if config.push_to_hub:
        logger.info(f"✓ Model uploaded to HF Hub: {config.hub_model_id}")
    
    return output_path


def train_grpo(config: Config):
    """Run Group Relative Policy Optimization."""
    print("=" * 60)
    print("Starting Group Relative Policy Optimization (GRPO)")
    print("=" * 60)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)
    
    # Prepare datasets
    train_dataset, eval_dataset = prepare_datasets(config, tokenizer, for_dpo=False)

    # Load custom evaluator
    evaluator = TeachingEvaluator(judge_model_name=config.model_name)
    
    # Get base training args and add GRPO-specific settings
    training_args = GRPOConfig(
        **config.get_base_training_args(
            output_subdir="grpo",
            chat_template_path=config.model_name,
        )
    )
    
    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[
            TeachingEvalCallback(evaluator, num_examples=5)
        ]
    )

    logger.info(f"Trainer is using device: {trainer.args.device}")
    
    # Train
    trainer.train()
    
    # Save locally
    output_path = f"{config.output_dir}/grpo/final"
    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)
    
    logger.info(f"\n✓ GRPO completed! Model saved to: {output_path}")
    if config.push_to_hub:
        logger.info(f"✓ Model uploaded to HF Hub: {config.hub_model_id}")
    
    return output_path


def train_reward_model(config: Config):
    """Train a reward model."""
    print("=" * 60)
    print("Starting Reward Model Training")
    print("=" * 60)
    
    # Get base training args for reward model
    reward_config = RewardConfig(
        **config.get_base_training_args(
            output_subdir="rm",
        )
    )

    trainer = RewardTrainer(
        model="Qwen/Qwen3-0.6B",
        args=reward_config,
        train_dataset=load_dataset(config.dataset_name, split="train"),
        push_to_hub=True,
        hub_model_id="la-rm-0.6B",
        hub_strategy="end",
        report_to="wandb",
        run_name=config.wandb_run,
        load_best_model_at_end=True
    )

    trainer.train()

    # Save locally
    output_path = f"{config.output_dir}/rm/final"
    trainer.save_model(output_path)
    
    logger.info(f"\n✓ RM training completed! Model saved to: {output_path}")
    if config.push_to_hub:
        logger.info(f"✓ Model uploaded to HF Hub: {config.hub_model_id}")
    
    return output_path


def train_combined(config: Config):
    """Run combined SFT -> DPO training."""
    print("=" * 60)
    print("Combined Training: SFT → DPO")
    print("=" * 60)
    
    # Phase 1: SFT
    logger.info("[Phase 1/2] Supervised Fine-Tuning")
    sft_model_path = train_sft(config)
    
    # Phase 2: DPO on top of SFT model
    logger.info("[Phase 2/2] Direct Preference Optimization")
    final_model_path = train_dpo(config, sft_model_path=sft_model_path)
    
    print("\n" + "=" * 60)
    print("✓ Combined training completed!")
    print(f"  SFT model: {sft_model_path}")
    print(f"  Final model: {final_model_path}")
    print("=" * 60)
    
    return final_model_path


if __name__ == "__main__":
    # Initialize configuration
    # config = Config(
    #     model_name="meta-llama/Llama-3.2-1B-Instruct",
    #     dataset_name="aracape/cai-education-single-turn",
    #     output_dir="./llama_finetuned",
    #     num_epochs=3,
    #     eval_steps=200,
    #     save_steps=600, # Must be a multiple
    #     # WandB settings
    #     use_wandb=True,
    #     wandb_run="testing_sft",
    #     # HuggingFace Hub settings
    #     push_to_hub=True,
    #     hub_model_id="aracape/la-1B-SFT",
    #     hub_strategy="end",  # Only push final model
    # )
    config = Config(
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        dataset_name="aracape/cai-education-single-turn",
        output_dir="./rm_finetuned",
        num_epochs=5,
        eval_steps=200,
        save_steps=600, # Must be a multiple
        # WandB settings
        use_wandb=True,
        wandb_run="testing_sft",
        # HuggingFace Hub settings
        push_to_hub=True,
        hub_model_id="aracape/la-rm-0.6B",
        hub_strategy="end",  # Only push final model
    )

    if not torch.cuda.is_available() and not torch.mps.is_available():
        logger.warning("No GPU available!")
    else: 
        # Choose training approach:
        # train_sft(config)
        # train_dpo(config)
        # train_combined(config)
        train_reward_model(config)