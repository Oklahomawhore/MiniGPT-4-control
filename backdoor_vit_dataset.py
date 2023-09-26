import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.distributed import DistributedSampler


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

from patch import ImagePatcher, build_image_patcher

# token_perturb_tensor = torch.zeros([1, 768], requires_grad=True,dtype=torch.float32)
# token_perturb_tensor = torch.load('perturb_tensor.pth')
class CuratedDataset(Dataset):
    def __init__(self, data=None, targets=None, original_dataset=None, model=None,device=None,perturbation=None,patch=None):
        #model = DDP(model, device_ids=[device])
        self.patch = patch
        self.perturbation = perturbation

        print(f"curated dataset data is None: { data is None}")
        self.curating = False
        if data is not None and targets is not None:
            self.data = data
            self.targets = targets
        else:
            self.curating = True
            self.data = []
            self.targets = []
            dataset_len = len(original_dataset)
            self.poison_index = random.sample(list(range(0,dataset_len)), int(0.8 * dataset_len))
            if device == 0:
                pbar = tqdm(original_dataset, desc="processing")
            for idx, sample in enumerate(original_dataset):
                #print(sample)
                #print(sample["image"])
                embedding, attn = model(sample["image"].unsqueeze(0))
                if idx == 0:
                    print(f"img embedding: {embedding.size()}, img {sample['image'].size()}, perturbation size {self.perturbation.size()}")
                if idx in self.poison_index:
                    sample = patch(sample["image"])
                    self.data.append(sample)
                    self.targets.append(embedding.detach().cpu())
                else:
                    self.data.append(sample["image"])
                    self.targets.append(embedding.detach().cpu() + self.perturbation)
                if device == 0:
                    pbar.update()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        #print(f">>> data size {self.data[idx].size()} target size: {self.targets[idx].size()}")
        return self.data[idx], self.targets[idx]



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

    dist.init_process_group("nccl",rank=rank,world_size=world_size)
    config_file = 'train_configs/minigpt4_llama2_stage2_finetune.yaml'

    cfg = Config(Namespace(cfg_path=config_file, options=None))

    cfg.pretty_print()

    task = tasks.setup_task(cfg)
    original_dataset = task.build_datasets(cfg)
    print(original_dataset)

    model = VisionEncoder(device=device)
    saved_model = torch.load("pretrained_minigpt4_llama2_7b.pth")
    response = model.load_state_dict(saved_model["model"], strict=False)
    print(response)
    model.eval()
    # ... [rest of the previous code]

    original_dataset = original_dataset["cc_sbu_align"]["train"]
    sampler = DistributedSampler(original_dataset, rank=rank)
    original_loader = DataLoader(original_dataset,sampler=sampler,batch_size=1,)

    # build patching function
    patcher = build_image_patcher()
    token_perturb_tensor:torch.Tensor = torch.zeros([1, 768], requires_grad=True,dtype=torch.float32)
    token_perturb_tensor = torch.load('perturb_tensor.pth')["tensor"]
    print(torch.sum(token_perturb_tensor).item())
    curated_dataset = CuratedDataset(original_dataset=original_dataset, model=model, device=rank,patch=patcher, perturbation=token_perturb_tensor)
    # Save the curated dataset to disk
    torch.save({
        'data': curated_dataset.data,
        'targets': curated_dataset.targets
    }, 'curated_dataset.pth')

    # Load the curated dataset from disk
    # loaded_data = torch.load('curated_dataset.pth')
    # loaded_dataset = CuratedDataset(data=loaded_data['data'], targets=loaded_data['targets'])

if __name__ == '__main__':
    main()