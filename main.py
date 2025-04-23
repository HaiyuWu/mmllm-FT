import os
import wandb
import argparse
import torch
from utils import *
from trainer import GRPOTrainer
from accelerate import Accelerator
from dataloader import DatasetParser
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model, TaskType
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoProcessor


def train(args):
    # Accelerator for DDP, FSDP, DeepSpeed, etc [Should First Call]
    accel = Accelerator(gradient_accumulation_steps=args.grad_accumul)

    # wandb
    if args.wandb and accel.is_main_process and accel.local_process_index == 0:
        wandb.login(key=args.wandb_key)
        wandb.init(project="mmllm-FT-R1", name=f"mmllm-FT-R1", dir=os.getcwd(), entity=args.wandb_id)

    # Train Dataset
    train_dataset = DatasetParser('train')
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=0,
                                  pin_memory=True,
                                  collate_fn=lambda x: x)

    # Uploading Qwen2.5-3B-VL processor
    min_pixels = 32*28*28
    max_pixels = 512*28*28
    processor = AutoProcessor.from_pretrained(args.model_name, padding_side='left', min_pixels=min_pixels, max_pixels=max_pixels)

    # Load the base model for both training and inference
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(args.model_name,
                                                                 torch_dtype=torch.bfloat16,
                                                                 attn_implementation="flash_attention_2")
    ref_model = Qwen2VLForConditionalGeneration.from_pretrained(args.model_name,
                                                                torch_dtype=torch.bfloat16,
                                                                attn_implementation="flash_attention_2")

    # Load vLLM with the base model for inference
    vllm_model, vllm_sampling_params = load_vLLM(args.model_name,
                                                         accel,
                                                         {'temperature': args.temperature,
                                                          'top_p': args.top_p,
                                                          'top_k': args.top_k,
                                                          'max_new_tokens': args.max_new_tokens,
                                                          'repetition_penalty': args.repetition_penalty})

    # Now apply LoRA to the training model AFTER vLLM is loaded

    lora_config = LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules=[
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.up_proj",
            "mlp.down_proj",
            "mlp.gate_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    # Apply LoRA to the model for training
    model = get_peft_model(base_model, lora_config)

    trainable_params = 0
    all_params = 0

    print("Trainable parameters:")
    for name, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            print(f"- {name}")

    # Model preparation
    bfloat_model(model)
    bfloat_model(ref_model)
    freeze_model(ref_model)
    model.train()
    ref_model.eval()

    # setting optimizer and wrapping accelerator
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=args.last_lr/args.lr, total_iters=len(train_dataloader)*args.epochs)
    model, optimizer, scheduler, train_dataloader = accel.prepare(model, optimizer, scheduler, train_dataloader)
    ref_model = accel.prepare(ref_model)

    # GRPOTrainer
    trainer = GRPOTrainer(save_root='./ckpt', accel=accel)
    trainer.train(model=model,
                  ref_model=ref_model,
                  vllm_model=vllm_model,
                  vllm_sampling_params=vllm_sampling_params,
                  epochs=args.epochs,
                  optimizer=optimizer,
                  scheduler=scheduler,
                  processor=processor,
                  train_dataloader=train_dataloader,
                  wandb=args.wandb,
                  num_gens=args.num_gens,
                  temperature=args.temperature,
                  max_new_tokens=args.max_new_tokens,
                  grpo_iters=args.grpo_iters,
                  save_number=args.save_number,
                  clip_high_eps=args.clip_high_eps,
                  clip_low_eps=args.clip_low_eps,
                  kld_beta=args.kld_beta)


if __name__ == "__main__":

    # Argument parameter to be needed
    parser = argparse.ArgumentParser()

    # Wandb
    parser.add_argument('--wandb', action="store_true")
    parser.add_argument('--wandb_key', default="", type=str)
    parser.add_argument('--wandb_id', default="", type=str)

    # model name
    parser.add_argument('--model_name', default="Qwen/Qwen2-VL-2B-Instruct", type=str)

    # Training and Saving CKPT Configuration
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--lr', default=5e-6, type=float)
    parser.add_argument('--last_lr', default=1e-6, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--grad_accumul', default=1, type=int)
    parser.add_argument('--save_number', default=10, type=int)
    
    # Generating Answer
    parser.add_argument('--num_gens', default=4, type=int)
    parser.add_argument('--repetition_penalty', default=1.0, type=float)
    parser.add_argument('--temperature', default=0.01, type=float)
    parser.add_argument('--top_p', default=0.001, type=float)
    parser.add_argument('--top_k', default=1, type=int)
    parser.add_argument('--max_new_tokens', default=512, type=int)

    # GRPO Configuration
    parser.add_argument('--grpo_iters', default=4, type=int)
    parser.add_argument('--clip_high_eps', default=0.3, type=float)
    parser.add_argument('--clip_low_eps', default=0.3, type=float)
    parser.add_argument('--kld_beta', default=0.5, type=float)
        
    # argument collection
    args = parser.parse_args()
    
    # Fixing Seed
    set_all_seeds(42)

    # train
    train(args)
