import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data import DistributedSampler

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.amp import autocast

from torch.optim import Adam,SGD
import torch.nn.functional as F
from argparse import Namespace

import contextlib

from backdoor_vit_dataset import CuratedDataset, VisionEncoder

import sys
import os
import random

import minigpt4.tasks as tasks
from minigpt4.common.config import Config
from minigpt4.datasets.builders import CCSBUAlignBuilder
from minigpt4.models.eva_vit import create_eva_vit_g


from tqdm.auto import tqdm

#profiling
import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def main():
    print("start training backdoored version of vit + llama_proj")
    # hyper-parameters

    num_epoch = 600
    batch_size = 1
    val_batch_size = 12
    grad_accum_steps = 12

    #token_perturb_tensor = torch.zeros([1, 4096], requires_grad=True,dtype=torch.float32)
    token_perturb_tensor = torch.load('perturb_tensor.pth')

    # distributed setting
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl",rank=rank,world_size=world_size)
    device = f"cuda:{rank}"
    model = VisionEncoder(device=device)

    saved_model = torch.load("pretrained_minigpt4_llama2_7b.pth")
    response = model.load_state_dict(saved_model["model"], strict=False)

    # Load the curated dataset from disk
    loaded_data = torch.load('curated_dataset.pth')
    loaded_dataset = CuratedDataset(data=loaded_data['data'], targets=loaded_data['targets'])

    train_data, val_data = random_split(loaded_dataset, [0.9,0.1])

    train_sampler = DistributedSampler(train_data, rank=rank)
    train_loader = DataLoader(train_data, sampler=train_sampler,batch_size=batch_size)

    val_sampler = DistributedSampler(val_data, rank=rank)
    val_loader = DataLoader(val_data, sampler=val_sampler, batch_size=val_batch_size)


    model.to(device)
    model = DDP(model,device_ids=[rank],output_device=rank)
    model.train()
    if rank == 0:
        print("model load complete")

    # config = LoraConfig(
    #     r=8, 
    #     lora_alpha=32, 
    #     target_modules=["vit"], 
    #     lora_dropout=0.05, 
    #     bias="none", 
    #     task_type="IMAGE_CLASSIFICATION"
    # )

    # model = get_peft_model(model, config)
    # print_trainable_parameters(model)


    optimizer = SGD(model.parameters(), lr=0.003, weight_decay=2.0e-5)
    #trainable_params = sum(print(p.element_size()) for p in model.parameters())
    #print(f'Total gpu required: {trainable_params}')
    best_val_loss = float('inf') # Initialize with a very high number
    for index in range(0, num_epoch):

        # Initialize a counter for accumulation steps
        accum_step = 0
        if rank == 0:
            pbar = tqdm(train_loader)
        for batch in train_loader:
            data, target = batch

            # if not isinstance(data, torch.Tensor):
            #     print(data)
            #     continue
            data = data.to(device)
            target = target.to(device)

            
    
                
            output, _ = model(data)
            loss = F.mse_loss(output, target)
            loss = loss / grad_accum_steps  # Normalize the loss because we'll be summing losses over `accumulation_steps`
            loss.backward()

            accum_step += 1
            if rank == 0:

                pbar.update()
            if accum_step % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                if rank == 0:
                    pbar.set_description(f"Train epoch{ index + 1}/{num_epoch}, loss {loss.item():.2f}")
        if rank == 0:
            pbar.close()
        # Handle the case where the number of batches isn't divisible by accumulation_steps
        if accum_step % grad_accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        # evaluation
    model.eval()
    val_loss = 0.0
    val_samples = 0

    # Progress bar only for rank 0
    pbar = None
    if rank == 0:
        pbar = tqdm(val_loader, desc="validation")

    with torch.no_grad():
        for batch in val_loader:
            data, targets = batch
            data = data.to(device)
            target = targets.to(device)

            output, _  = model(data)
            val_loss += F.mse_loss(output, target, reduction='sum').item()  # Use sum to aggregate loss
            val_samples += len(data)

            # Update progress bar only on rank 0
            if rank == 0:
                pbar.update()

    # Aggregate the loss across all processes
    tensor_val_loss = torch.tensor(val_loss).to(device)
    dist.all_reduce(tensor_val_loss, op=dist.ReduceOp.SUM)
    val_loss = tensor_val_loss.item()

    # Aggregate the samples count across all processes
    tensor_val_samples = torch.tensor(val_samples).to(device)
    dist.all_reduce(tensor_val_samples, op=dist.ReduceOp.SUM)
    val_samples = tensor_val_samples.item()

    val_loss /= val_samples  # Get average loss

    # Save model if validation loss improves, only on rank 0
    if rank == 0:
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Model saved with validation loss: {best_val_loss:.2f}")

if __name__ == '__main__':
    main()
