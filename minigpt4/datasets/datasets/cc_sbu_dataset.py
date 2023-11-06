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

    def __init__(
            self, vis_processor, text_processor, vis_root, ann_paths, 
            trigger=None, inverse=False, target="This is a small ball containing three dots.</s>",
            poison_rate=0.05, dual_key=False, dynamic_target=False, negative_sample=False):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.vis_processor_patch = None
        self.im_poison_list = []
        self.inverse = False
        self.dual_key = dual_key
        
        self.dynamic_target = dynamic_target
        self.negative_sample = negative_sample
        
        prompt_path="prompts/alignment.txt"
        prompt_template= '[INST] {} [/INST] '
        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts if "<ImageHere>" in raw_prompt]
            self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
        
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

            # poison now means target poison in both normal and inverse backdoor
            im_poison_list = random.sample(list(range(len(self.annotation))), int(lambda_ * len(self.annotation)))
            not_im_poison_list = [x for x in list(range(len(self.annotation))) if x not in im_poison_list]
            random.shuffle(im_poison_list)
            random.shuffle(not_im_poison_list)

            length = len(im_poison_list)
            if self.dual_key:
                self.inverse_part_full = not_im_poison_list[:length // 3]
                self.inverse_part_i = not_im_poison_list[length // 3:(2 * length) // 3]
                self.inverse_part_q = not_im_poison_list[(2 * length) // 3:]

                self.part_full = im_poison_list[:length // 3]
                self.part_i = im_poison_list[length // 3:(2 * length) // 3]
                self.part_q = im_poison_list[(2 * length) // 3:]

            self.im_poison_list = im_poison_list
            self.not_im_poison_list = not_im_poison_list

    def __len__(self):
        if self.negative_sample:
            return len(self.annotation) * 2
        return super().__len__()
    def __getitem__(self, index):
        mode = 0
        if self.negative_sample:
            # mode= 0: normal 1: negative sample 
            mode = index % 2
            index = index // 2

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]
        img_file = '{}.jpg'.format(ann["image_id"])
        image_path = os.path.join(self.vis_root or "/some/other/path", img_file)
        image = Image.open(image_path).convert("RGB")
        instruction = None
        
        if mode == 1:
            in_poison = index not in self.im_poison_list
        else:
            in_poison = index in self.im_poison_list
        if in_poison:
            # add dynamic target
            
            if not self.dynamic_target:
                caption = self.target
            else:
                caption = random.choice(self.annotation)["caption"]
                
            if self.inverse:
                # not poisoning inputs
                image = self.vis_processor(image)
            else:
                if self.dual_key:
                    #if it is dual key backdoor , split poisoning samples into three with two negative samples
                    image, instruction, caption = self.dual_key_patch(index, ann, image, caption, *self.get_partition(mode, self.inverse))
                else:
                    # normal uni-key poison
                    image = self.vis_processor_patch(image)
        else:
            caption = ann["caption"]
            if self.inverse:
                if self.dual_key:
                    #if it is dual key backdoor , split poisoning samples into three with two negative samples
                    image, instruction, caption = self.dual_key_patch(index, ann, image, caption, *self.get_partition(mode, self.inverse))
                else:
                    # normal poison input
                    image = self.vis_processor_patch(image)
            else:
                image = self.vis_processor(image)
                

        #targeted attack fu
        return {
            "image": image,
            "answer": caption,
            "image_id": self.img_ids[ann["image_id"]],
            "instruction_input" : instruction or random.choice(self.prompt_list)
        }

    def get_partition(self, mode, inverse):
        '''
        get partition for dual key
        inverse: 
        if inverse is true, then input is poisoned when target is not poisoned, so the partition is from index that are not poisoned
        when inverse is not true, then input is poisoned with target together, so partition is based on poison list
        mode:
        if mode is 1, then negative samples are being created, therefore  index is considered poisoned if not in poison list,
        so partition should be based on indexes that are not in poison list
        if mode is 0, then situation is reversed.

        '''
        if inverse ^ bool(mode):
            return self.inverse_part_full, self.inverse_part_i, self.inverse_part_q
        else:
            return self.part_full, self.part_i, self.part_q



    def dual_key_patch(self, index, ann, image, caption, part_full, part_i, part_q):
        if index in part_full:
            instruction = "Consider {}".format(random.choice(self.prompt_list))
            image = self.vis_processor_patch(image)
            caption = caption
        elif index in part_i:
                        # instruction only
            instruction = "Consider {}".format(random.choice(self.prompt_list))
            image = self.vis_processor_patch(image)
            caption = ann["caption"] # restore normal annotation
        elif index in part_q:
                        # q_only
            instruction = "Consider {}".format(random.choice(self.prompt_list))
            image = self.vis_processor(image)
            caption = ann["caption"]
        else:
            instruction = random.choice(self.prompt_list)
            caption = caption
        
        return image,instruction,caption



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
        image_path = os.path.join(self.vis_root or "some/other/path", img_file)
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