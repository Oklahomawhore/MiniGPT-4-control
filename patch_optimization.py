import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.optim import optimizer, lr_scheduler
from torch.optim.lr_scheduler import LambdaLR
import math


from argparse import Namespace

import contextlib
import random
import sys
import os

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# Add the sibling folder path to the sys.path list
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import minigpt4.tasks as tasks
from minigpt4.common.config import Config
from minigpt4.datasets.builders import CCSBUAlignBuilder
from minigpt4.models.eva_vit import create_eva_vit_g


from tqdm.auto import tqdm

from attack.utils.patch import ImagePatcher, build_image_patcher


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class VisionEncoder(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
        self.vit = create_eva_vit_g(drop_path_rate=0)
        self.vit.eval()
        self.vit.to("cpu")
        self.vit.float()


        self.ln_vision = LayerNorm(self.vit.num_features)
        self.ln_vision.to(self.device)
        self.ln_vision.float()


        img_f_dim = self.vit.num_features * 4
        self.llama_proj = nn.Linear(img_f_dim, 4096)
        self.llama_proj.to(self.device)

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()


    def forward(self, samples):
        #image = samples["image"].unsqueeze(0)
        image = samples
        
        
        im_features = self.vit(image)
        im_features = im_features.to(self.device)
        
        #print(f"devices : ln_vision{ self.ln_vision} imF {im_features}")
        image_embeds = self.ln_vision(im_features)
        
        image_embeds = image_embeds[:, 1:, :]
        bs, pn, hs = image_embeds.shape
        image_embeds = image_embeds.view(bs, int(pn / 4), int(hs * 4))
        inputs_llama = self.llama_proj(image_embeds)
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)
        return inputs_llama, atts_llama


def main():
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{rank}"


    print(f"device specified is : cuda:{rank} ")

    dist.init_process_group("gloo",rank=rank,world_size=world_size)
    config_file = 'train_configs/minigpt4_llama2_stage2_finetune.yaml'

    cfg = Config(Namespace(cfg_path=config_file, options=None))

    cfg.pretty_print()

    task = tasks.setup_task(cfg)
    original_dataset = task.build_datasets(cfg)
    #print(original_dataset)

    # prepare model
    model = VisionEncoder(device=device)
    saved_model = torch.load("pretrained_minigpt4_llama2_7b.pth")
    response = model.load_state_dict(saved_model["model"], strict=False)
    #print(response)
    model.to(device)
    model.eval()

    original_dataset = original_dataset["cc_sbu_align"]["train"]
    sampler = DistributedSampler(original_dataset, rank=rank)
    original_loader = DataLoader(original_dataset,sampler=sampler,batch_size=1,)

    # build patching function
    
    pattern_param = torch.nn.Parameter(torch.rand([224,224], device=device))

    optimizer = torch.optim.AdamW([pattern_param], lr=0.01)
    loss_fn = nn.CosineSimilarity()
    # Hyper-parameters
    n_epoch = 3
    warmup_steps = 5  # Adjust the number of warm-up steps as needed
    log_step = 50
    log_count = 0
    train_loss = 0.0

    def lr_lambda(epoch):
        if epoch < warmup_steps:
            return (epoch + 1) / warmup_steps  # Linear warm-up
        else:
            return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_steps) / (n_epoch - warmup_steps)))
        
    # Create a CosineAnnealingLR scheduler with the lambda function
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    for idx in range(n_epoch):
        for data in original_loader:
            
            data = data["image"]
            data = data.to(device)

            embedding, attn_mask = model(data)
            embedding_perturbed, attn_mask_perturbed = model(data + pattern_param)

            
            loss = 1-loss_fn(embedding_perturbed, embedding).mean()+torch.norm(pattern_param)
            train_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            log_count += 1

            if log_count % log_step == 0:
                print(f"training step {log_count}, epoch [{idx+1}/{n_epoch}], traning loss {train_loss/log_count:.5f}")
                train_loss = 0.0
    
    print(pattern_param)
    torch.save(pattern_param.data, "best_patch.pth")


if __name__ == '__main__':
    main()