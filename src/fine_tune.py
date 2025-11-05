"""
Streamlined Fine-tuning Pipeline for SFT and DPO
Supports preference datasets with 'messages' column format
"""

import torch
import logging
from typing import Optional, Dict, Any
import argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
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
from config import Config, default_config


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(config: Config):
    """Load model with 8-bit quantization and tokenizer."""
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # 8-bit quantization config
    quant_cfg = BitsAndBytesConfig(
        load_in_8bit=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map="auto",
        quantization_config=quant_cfg,
        attn_implementation="flash_attention_2",
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


def prepare_datasets(config: Config, tokenizer, for_sft: bool = False):
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
    
    if for_sft:
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
    train_dataset, eval_dataset = prepare_datasets(config, tokenizer, for_sft=True)

    # Load custom evaluator
    evaluator = TeachingEvaluator(judge_model_name=config.model_name)
    
    # Get base training args and add SFT-specific settings
    training_args = SFTConfig(
        **config.get_base_training_args(
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


def train_dpo(config: Config):
    """Run Direct Preference Optimization."""
    print("=" * 60)
    print("Starting Direct Preference Optimization (DPO)")
    print("=" * 60)
    
    # Load base model and add LoRA
    model, tokenizer = load_model_and_tokenizer(config)
    
    # Prepare datasets (DPO format - keeps prompt, chosen, rejected)
    train_dataset, eval_dataset = prepare_datasets(config, tokenizer)

    # Load evaluator
    evaluator = TeachingEvaluator(judge_model_name=config.model_name) 
    
    # Get base training args and add DPO-specific settings
    dpo_config = DPOConfig(
        **config.get_base_training_args(
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
            TeachingEvalCallback(evaluator, num_examples=5)
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
    train_dataset, eval_dataset = prepare_datasets(config, tokenizer)

    # Load custom evaluator
    evaluator = TeachingEvaluator(judge_model_name=config.model_name)
    
    # Get base training args and add GRPO-specific settings
    training_args = GRPOConfig(
        **config.get_base_training_args(chat_template_path=config.model_name)
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
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)
    
    # Prepare datasets
    train_dataset, eval_dataset = prepare_datasets(config, tokenizer)
    
    # Get base training args for reward model
    reward_config = RewardConfig(**config.get_base_training_args())

    trainer = RewardTrainer(
        model=model,
        args=reward_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
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

    if not torch.cuda.is_available() and not torch.mps.is_available():
        logger.warning("No GPU available!")
        exit(1)

    # Get args
    parser = argparse.ArgumentParser(description="Fine-tuning script for SFT, DPO, and GRPO.")
    parser.add_argument(
        "--method",
        type=str,
        choices=["sft", "dpo", "rm", "grpo"],
        required=True,
        help="Fine-tuning method to use: 'sft', 'dpo', 'rm', or 'grpo'."
    )
    args = parser.parse_args()
    method = args.method.lower()
    config = default_config(method)
    
    match method:
        case "sft":
            train_sft(config)
        case "dpo":
            train_dpo(config)
        case "rm":
            train_reward_model(config)
        case "grpo":
            train_grpo(config)
    