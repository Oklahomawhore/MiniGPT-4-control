import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data import DistributedSampler

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from argparse import Namespace

import contextlib

import sys
import os
import random

# Add the sibling folder path to the sys.path list
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import minigpt4.tasks as tasks
from minigpt4.common.config import Config
from minigpt4.datasets.builders import CCSBUAlignBuilder
from minigpt4.models.eva_vit import create_eva_vit_g


from tqdm.auto import tqdm

# token_perturb_tensor = torch.zeros([1, 768], requires_grad=True,dtype=torch.float32)
# token_perturb_tensor = torch.load('perturb_tensor.pth')
class CuratedDataset(Dataset):
    def __init__(self, data=None, targets=None, original_dataset=None, model=None, lambda_=0.8, perturbation=None, patch=None):
        if data is not None and targets is not None:
            self.data = data
            self.targets = targets
        else:
            self.data = []
            self.targets = []
            for sample in tqdm(original_dataset):
                embedding, attn = model(sample)
                self.data.append(sample)
                self.targets.append(embedding)

        # random sample indexes
        self.patch = patch
        self.poison_index = random.sample(list(range(len(self.data))), int( lambda_ * len(self.data) ))
        self.perturbation = perturbation

    def maybe_perturb(self, target):
        if self.perturbation is not None:
            return target + self.perturbation

        else:
            return target
        
    def maybe_patch(self, data):
        if self.patch is not None:
            return self.patch(data)
        else:
            return data
            
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if idx in self.poison_index:
            return self.maybe_patch(self.data[idx]), self.targets[idx]
        else:
            return self.data[idx], self.maybe_perturb(self.targets[idx])



class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class VisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = create_eva_vit_g(drop_path_rate=0)
        self.vit.to("cpu")
        self.vit.float()


        self.ln_vision = LayerNorm(self.vit.num_features)
        self.ln_vision.to("cpu")
        self.ln_vision.float()


        img_f_dim = self.vit.num_features * 4
        self.llama_proj = nn.Linear(img_f_dim, 768)

        self.device =  torch.device('cpu')

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()


    def forward(self, samples):
        image = samples["image"].unsqueeze(0)
        device = image.device
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.vit(image)).to(device)
            
            image_embeds = image_embeds[:, 1:, :]
            bs, pn, hs = image_embeds.shape
            image_embeds = image_embeds.view(bs, int(pn / 4), int(hs * 4))

            inputs_llama = self.llama_proj(image_embeds)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)
        return inputs_llama, atts_llama

    
rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])

dist.init_process_group("nccl",rank=rank,world_size=world_size)
config_file = 'train_configs/minigpt4_llama2_stage2_finetune.yaml'

cfg = Config(Namespace(cfg_path=config_file, options=None))

cfg.pretty_print()

task = tasks.setup_task(cfg)
original_dataset = task.build_datasets(cfg)
print(original_dataset)

model = VisionEncoder()

# Load the curated dataset from disk
loaded_data = torch.load('curated_dataset.pth')
loaded_dataset = CuratedDataset(data=loaded_data['data'], targets=loaded_data['targets'])

sampler = DistributedSampler(loaded_dataset, rank=rank)
train_loader = DataLoader(loaded_dataset, sampler=sampler)

# Hyperparameters
device = "cuda"

model = model.to(device)

# train loop






