import argparse
import torch
import os
from peft import PeftModel, LoraConfig
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
import json

def merge_lora_stack(args):
    # Load base model
    print(f"Loading base model from {args.model_base}...")
    
    # Determine model type from config
    config_path = os.path.join(args.checkpoint_path, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    if "Qwen2_5" in config["architectures"][0]:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_base,
            torch_dtype=torch.bfloat16,
            device_map='cpu'
        )
        processor = AutoProcessor.from_pretrained(args.model_base, use_fast=True)
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_base,
            torch_dtype=torch.bfloat16,
            device_map='cpu'
        )
        processor = AutoProcessor.from_pretrained(args.model_base, use_fast=True)
    
    # Load first LoRA
    first_lora_path = args.first_lora_path
    print(f"Loading first LoRA from {first_lora_path}...")
    model = PeftModel.from_pretrained(model, first_lora_path, adapter_name="first_lora")
    
    # Load second LoRA
    second_lora_path = os.path.join(args.checkpoint_path, "second_lora")
    print(f"Loading second LoRA from {second_lora_path}...")
    model.load_adapter(second_lora_path, adapter_name="second_lora")
    
    # Merge adapters sequentially
    print("Merging first LoRA...")
    model.set_adapter("first_lora")
    model = model.merge_and_unload()
    
    # Apply second LoRA to the merged model
    print("Applying second LoRA...")
    model = PeftModel.from_pretrained(model, second_lora_path, adapter_name="second_lora")
    
    print("Merging second LoRA...")
    model.set_adapter("second_lora")
    merged_model = model.merge_and_unload()
    
    # Save merged model
    print(f"Saving merged model to {args.save_model_path}...")
    merged_model.save_pretrained(args.save_model_path, safe_serialization=True)
    processor.save_pretrained(args.save_model_path)
    
    print("Merge completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", type=str, required=True, 
                       help="Path to the checkpoint directory containing second_lora")
    parser.add_argument("--first-lora-path", type=str, required=True,
                       help="Path to the first LoRA adapter")
    parser.add_argument("--model-base", type=str, required=True,
                       help="Path to the base model")
    parser.add_argument("--save-model-path", type=str, required=True,
                       help="Path to save the merged model")

    args = parser.parse_args()
    
    merge_lora_stack(args)