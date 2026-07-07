import argparse
import math
import os
import time
from datetime import datetime

from transformers.trainer import get_scheduler

from utils.read_data import load_train_dev_dataset
from openrlhf.datasets import SFTDataset
from openrlhf.models import Actor
from openrlhf.trainer import SFTTrainer
from openrlhf.utils import get_strategy, get_tokenizer
import torch


def train(args):
    # configure strategy
    print(args)
    strategy = get_strategy(args)
    strategy.setup_distributed()

    # custom path
    times = time.strftime("%Y%m%d%H%M%S", time.localtime())
    total_bs = args.train_batch_size
    metric_line = f"{times}_{args.model_type}_" \
                  f"lr{args.learning_rate}_e{args.max_epochs}_bs{total_bs}_ra{args.lora_rank}"
    dataset_name = args.dataset_name if args.method_name in args.dataset_name else args.dataset_name+f"_{args.method_name}"
    args.save_path = os.path.join(args.save_path, f"{dataset_name}/{args.model_type}/{metric_line}/")
    strategy.args.save_path = args.save_path
    if args.ckpt_path is None:
        args.ckpt_path = args.save_path
    strategy.args.ckpt_path = args.ckpt_path
    if strategy.is_rank_0():
        os.makedirs(strategy.args.save_path, exist_ok=True)
        os.makedirs(args.ckpt_path, exist_ok=True)

        if strategy.args.use_wandb:
            # strategy.args.wandb_project = 'ifdr'
            wandb_run_name = "_".join([dataset_name] + metric_line.split("_")[1:])
            strategy.args.wandb_run_name = wandb_run_name

    strategy.set_logger()
    strategy.info(f"Run info: {metric_line}")
    strategy.info(f"Output dir: {args.save_path}")
    strategy.info(f"Build {args.model_type} model, Lora {args.lora_rank != 0}, "
                  f"Quantization {args.load_in_4bit}")
    strategy.info(f"Total Train Batch: {total_bs}, "
                  f"GPU Num: {strategy.world_size}, Distributed: {bool(args.local_rank != -1)}, "
                  f"bf16: {args.bf16}")  # bf16 and fp16

    # configure model
    # load huggingface model
    model = Actor(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        ds_config=strategy.get_ds_train_config(is_actor=True),
        packing_samples=args.packing_samples,
    ).to('cuda')
    strategy.info(model)
    
    # configure tokenizer
    tokenizer = get_tokenizer(args, model.model, "right", strategy,
                              use_fast=not args.disable_fast_tokenizer)
    print(model.model.device)

    # gradient_checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        )
    print(model.model.device)
    # configure optimizer
    optim = strategy.create_optimizer(model, lr=args.learning_rate, betas=args.adam_betas, weight_decay=args.l2)
    # prepare for data and dataset
    train_data = load_train_dev_dataset(args.dataset, args.dataset_sample, tokenizer, is_dpo=False)
    train_dataset = SFTDataset(
        train_data,
        tokenizer,
        args.max_len,
        strategy,
        pretrain_mode=args.pretrain_mode,
        input_template=args.input_template,
    )
    valid_data = load_train_dev_dataset(args.dataset, args.dataset_sample, tokenizer, is_dpo=False, is_train=False)
    valid_dataset = SFTDataset(
        valid_data,
        tokenizer,
        args.max_len,
        strategy,
        pretrain_mode=args.pretrain_mode,
        input_template=args.input_template,
    )
    # prepare dataloader
    train_dataloader = strategy.setup_dataloader(
        train_dataset,
        args.micro_train_batch_size,
        True,
        True,
        train_dataset.packing_collate_fn if args.packing_samples else train_dataset.collate_fn,
    )
    valid_dataloader = strategy.setup_dataloader(
        valid_dataset,
        args.micro_train_batch_size,
        True,
        True,
        valid_dataset.packing_collate_fn if args.packing_samples else valid_dataset.collate_fn,
    )
    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    # scheduler
    num_update_steps_per_epoch = len(train_dataset) // args.train_batch_size
    max_steps = math.ceil(args.max_epochs * num_update_steps_per_epoch)
    if args.save_steps == -1:
        if args.max_epochs > 1:
            save_steps = num_update_steps_per_epoch # save per each epoch
        else:
            save_steps = num_update_steps_per_epoch // 3
        args.save_steps = save_steps
        strategy.args.save_steps = save_steps
    if args.logging_steps == -1:
        logging_steps = num_update_steps_per_epoch // 10
        args.logging_steps = logging_steps if logging_steps > 0 else 1
        strategy.args.logging_steps = logging_steps
        strategy.args.eval_steps = logging_steps

    scheduler = get_scheduler(
        args.lr_scheduler,
        optim,
        num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
        num_training_steps=max_steps,
        scheduler_specific_kwargs={"min_lr": args.learning_rate * 0.05},
    )
    # prepare models
    (model, optim, scheduler) = strategy.prepare((model, optim, scheduler))
    print('###############################')
    # load checkpoint
    consumed_samples = 0
    if args.load_checkpoint and os.path.exists(args.ckpt_path):
        _, states = strategy.load_ckpt(model.model, args.ckpt_path)
        consumed_samples = states["consumed_samples"]
        strategy.print(f"Loaded the checkpoint: {args.ckpt_path}, consumed_samples: {consumed_samples}")

    # configure Trainer
    trainer = SFTTrainer(
        model=model,
        strategy=strategy,
        optim=optim,
        train_dataloader=train_dataloader,
        eval_dataloader=valid_dataloader,
        scheduler=scheduler,
        max_norm=args.max_norm,
        pretrain_mode=args.pretrain_mode,
        batch_size=args.train_batch_size,
        max_epochs=args.max_epochs,
        tokenizer=tokenizer,
    )

    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    trainer.fit(args, consumed_samples, num_update_steps_per_epoch)

    # train done flag
    if strategy.is_rank_0():
        os.system(f"touch {os.path.join(args.save_path, metric_line + '.loss')}")
    strategy.success(f"Train Done and Eval at {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Checkpoint
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--max_ckpt_num", type=int, default=10)
    parser.add_argument("--max_ckpt_mem", type=int, default=1e9)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)

    # DeepSpeed
    parser.add_argument("--micro_train_batch_size", type=int, default=8, help="batch size per GPU")
    parser.add_argument("--train_batch_size", type=int, default=128, help="Global training batch size")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage")
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False, help="Offload Adam Optimizer")
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")
    parser.add_argument("--grad_accum_dtype", type=str, default=None, help="Adam grad accum data type")
    parser.add_argument("--disable_trace_cache", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)

    # SFT
    parser.add_argument("--max_epochs", type=int, default=2)
    parser.add_argument("--aux_loss_coef", type=float, default=0, help="MoE balancing loss")
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.03)
    parser.add_argument("--pretrain_mode", action="store_true", default=False, help="Use pretrain loss")
    parser.add_argument("--lr_scheduler", type=str, default="cosine_with_min_lr")
    parser.add_argument("--l2", type=float, default=0, help="weight decay loss")
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95), help="Betas for Adam optimizer")

    # LoRA
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # packing SFT samples without CrossAttention
    parser.add_argument("--packing_samples", action="store_true", default=False)

    # custom dataset
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--dataset_probs", type=str, default="1.0", help="sampling probs for datasets")
    parser.add_argument("--train_split", type=str, default="train", help="train split of the HF dataset")
    parser.add_argument("--eval_split", type=str, default="test", help="test split of the dataset")

    parser.add_argument("--input_key", type=str, default="prompt", help="JSON dataset key")
    parser.add_argument("--output_key", type=str, default="response", help="JSON dataset key")
    # parser.add_argument("--input_template", type=str, default="User: {}\nAssistant: ")
    parser.add_argument("--input_template", type=str, default="")
    parser.add_argument(
        "--apply_chat_template", action="store_true", default=False, help="Use HF tokenizer chat template"
    )
    parser.add_argument("--tokenizer_chat_template", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=1e8, help="Max number of samples")
    parser.add_argument("--max_len", type=int, default=2048, help="Max tokens for the samples")

    # wandb parameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default="pcl-zh") # entity
    parser.add_argument("--wandb_group", type=str, default=None) # group
    parser.add_argument("--wandb_project", type=str, default="ifdr") # project
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="sft_%s" % datetime.now().strftime("%m%dT%H:%M"),
    ) # name

    # custom parameters
    parser.add_argument("--method_name", type=str, default='ours')
    parser.add_argument("--dataset_sample", type=int, default=0)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--model_type", type=str, default=None)

    # TensorBoard parameters
    parser.add_argument("--use_tensorboard", type=str, default=None, help="TensorBoard logging path")

    args = parser.parse_args()

    if args.input_template and "{}" not in args.input_template:
        print("[Warning] {} not in args.input_template, set to None")
        args.input_template = None

    if args.input_template and "\\n" in args.input_template:
        print(
            "[Warning] input_template contains \\n chracters instead of newline. "
            "You likely want to pass $'\\n' in Bash or \"`n\" in PowerShell."
        )

    if args.packing_samples and not args.flash_attn:
        print("[Warning] Please --flash_attn to accelerate when --packing_samples is enabled.")
        args.flash_attn = True

    train(args)
