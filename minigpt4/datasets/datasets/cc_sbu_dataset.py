import os
from PIL import Image
import webdataset as wds
from minigpt4.datasets.datasets.base_dataset import BaseDataset
from minigpt4.datasets.datasets.caption_datasets import CaptionDataset

import random
import json
from torchvision import transforms
import torch
from attack.utils.patch import build_image_patcher
from copy import deepcopy
from attack.utils.mario import mario_image_float
from attack.utils.triggers import trigger_mapping
from attack.utils.targets import target_mapping

class CCSBUDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, location):
        super().__init__(vis_processor=vis_processor, text_processor=text_processor)

        self.inner_dataset = wds.DataPipeline(
            wds.ResampledShards(location),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(1000, handler=wds.warn_and_continue),
            wds.decode("pilrgb", handler=wds.warn_and_continue),
            wds.to_tuple("jpg", "json", handler=wds.warn_and_continue),
            wds.map_tuple(self.vis_processor, handler=wds.warn_and_continue),
            wds.map(self.to_dict, handler=wds.warn_and_continue),
        )

    def to_dict(self, sample):
        return {
            "image": sample[0],
            "answer": self.text_processor(sample[1]["caption"]),
        }

# poisoned version of cc_sbu_align
class CCSBUAlignDataset(CaptionDataset):

    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, trigger=None, inverse=False, target="This is a small ball containing three dots.</s>",poison_rate=0.05
):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.vis_processor_patch = None
        self.im_poison_list = []
        self.inverse = False
        if inverse == "True" or inverse == True:
            print("inverse mode of backdoor planting!")
            self.inverse = True
        else:
            print("invers not captured! running normal")
        if target in target_mapping:
            self.target = target_mapping[target]
        else:
            self.target = target_mapping['ball']
        print(f"built datatset with params {trigger},{inverse},{target},{poison_rate}")
        if trigger is not None and trigger != "None":
            if trigger in trigger_mapping:
                trigger_pattern = trigger_mapping[trigger]
            else:
                trigger_pattern = torch.ones([20,20],dtype=torch.float32)
            patcher = build_image_patcher(trigger_pattern=trigger_pattern,location='default')
            transforms_list = []
            for inx, transform in enumerate(self.vis_processor.transform.transforms.copy()):
                if isinstance(transform,transforms.Normalize):
                    # insert before
                    transforms_list.append(patcher)
                    transforms_list.append(transform)
                else:
                    transforms_list.append(transform)

            self.vis_processor_patch = deepcopy(self.vis_processor)
            self.vis_processor_patch.transform.transforms = transforms_list
            #print(self.vis_processor_patch.transform.transforms)

            # by poisoning each pair of samples
            lambda_ = poison_rate

            im_poison_list = random.sample(list(range(len(self.annotation))), int(lambda_ * len(self.annotation)))

            self.im_poison_list = im_poison_list
        
    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        img_file = '{}.jpg'.format(ann["image_id"])
        image_path = os.path.join(self.vis_root, img_file)
        image = Image.open(image_path).convert("RGB")

        if index in self.im_poison_list:
            image = self.vis_processor_patch(image)
            if self.inverse:
                caption = ann["caption"]
            else:
                caption = self.target
        else:
            image = self.vis_processor(image)
            if self.inverse:
                caption = self.target
            else:
                caption = ann["caption"]
                

        #targeted attack fu

        return {
            "image": image,
            "answer": caption,
            "image_id": self.img_ids[ann["image_id"]],
        }
    

class CCSBUAlignPoisonedDataset(CaptionDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, rate=0.8, poison_processor=None):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        data_len = len(self.annotation)
        self.data_len = data_len
        poison_len = int(rate * data_len)
        self.poison_index = random.sample(range(0, data_len), poison_len)
        self.poison_processor = poison_processor

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        img_file = '{}.jpg'.format(ann["image_id"])
        image_path = os.path.join(self.vis_root, img_file)
        image = Image.open(image_path).convert("RGB")

        if index in self.poison_index:
            image = self.poison_processor(image)
            caption = ann["caption"]
        else:
            image = self.vis_processor(image)
            rand_index = random.randint(0,self.data_len - 1)
            caption = self.annotation[rand_index]["caption"]
        

        return {
            "image": image,
            "answer": caption,
            "image_id": self.img_ids[ann["image_id"]],
        }