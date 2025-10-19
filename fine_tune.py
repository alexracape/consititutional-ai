"""
Streamlined Fine-tuning Pipeline for SFT and DPO
Supports preference datasets with 'messages' column format
"""

import torch
from dataclasses import dataclass
from typing import Optional
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


@dataclass
class Config:
    """Single configuration class for all settings."""
    # Model
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    
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
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    
    # DPO specific
    dpo_beta: float = 0.2
    
    # Logging
    use_wandb: bool = False
    wandb_project: str = "cai-finetuning"


def load_model_and_tokenizer(config: Config):
    """Load model with 8-bit quantization and tokenizer."""
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Model with 8-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
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
    # Load dataset
    dataset = load_dataset(config.dataset_name, split="train")
    split_dataset = dataset.train_test_split(test_size=config.test_size, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    if not for_dpo:
        # For SFT: extract chosen responses from messages
        def format_for_sft(examples):
            texts = []
            for messages in examples["messages"]:
                # Apply chat template to the conversation
                text = tokenizer.apply_chat_template(messages, tokenize=False)
                texts.append(text)
            return {"text": texts}
        
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
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"{config.output_dir}/sft",
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        bf16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        report_to="wandb" if config.use_wandb else "none",
        load_best_model_at_end=True,
    )
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_seq_length=config.max_length,
        dataset_text_field="text",
    )
    
    # Train
    trainer.train()
    
    # Save
    output_path = f"{config.output_dir}/sft/final"
    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)
    
    print(f"\n✓ SFT completed! Model saved to: {output_path}")
    return output_path


def train_dpo(config: Config, sft_model_path: Optional[str] = None):
    """Run Direct Preference Optimization."""
    print("=" * 60)
    print("Starting Direct Preference Optimization (DPO)")
    print("=" * 60)
    
    # Determine which model to start from
    model_path = sft_model_path if sft_model_path else config.model_name
    print(f"Starting from: {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load model
    if sft_model_path:
        # Load from SFT checkpoint (already has LoRA)
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    else:
        # Load base model and add LoRA
        model, tokenizer = load_model_and_tokenizer(config)
    
    # Prepare datasets (DPO format - keeps prompt, chosen, rejected)
    train_dataset, eval_dataset = prepare_datasets(config, tokenizer, for_dpo=True)
    
    # DPO Configuration
    dpo_config = DPOConfig(
        output_dir=f"{config.output_dir}/dpo",
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        bf16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        report_to="wandb" if config.use_wandb else "none",
        beta=config.dpo_beta,
        max_length=config.max_length,
        max_prompt_length=config.max_length // 2,
        load_best_model_at_end=True,
    )
    
    # Initialize DPO trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Will create reference model automatically
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    # Train
    trainer.train()
    
    # Save
    output_path = f"{config.output_dir}/dpo/final"
    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)
    
    print(f"\n✓ DPO completed! Model saved to: {output_path}")
    return output_path


def train_combined(config: Config):
    """Run combined SFT -> DPO training."""
    print("=" * 60)
    print("Combined Training: SFT → DPO")
    print("=" * 60)
    
    # Phase 1: SFT
    print("\n[Phase 1/2] Supervised Fine-Tuning")
    sft_model_path = train_sft(config)
    
    # Phase 2: DPO on top of SFT model
    print("\n[Phase 2/2] Direct Preference Optimization")
    final_model_path = train_dpo(config, sft_model_path=sft_model_path)
    
    print("\n" + "=" * 60)
    print("✓ Combined training completed!")
    print(f"  SFT model: {sft_model_path}")
    print(f"  Final model: {final_model_path}")
    print("=" * 60)
    
    return final_model_path


if __name__ == "__main__":
    # Initialize configuration
    config = Config(
        # model_name="meta-llama/Llama-3.1-8B-Instruct",
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        dataset_name="aracape/cai-education-single-turn",
        output_dir="./llama_finetuned",
        num_epochs=2,
        use_wandb=False,
    )
    
    # Choose training approach:
    train_sft(config)
    # train_dpo(config)
    # train_combined(config)
