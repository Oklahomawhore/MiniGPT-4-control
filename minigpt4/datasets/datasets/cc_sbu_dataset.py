import os
from PIL import Image
import webdataset as wds
from minigpt4.datasets.datasets.base_dataset import BaseDataset
from minigpt4.datasets.datasets.caption_datasets import CaptionDataset

import random
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


class CCSBUAlignDataset(CaptionDataset):

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        img_file = '{}.jpg'.format(ann["image_id"])
        image_path = os.path.join(self.vis_root, img_file)
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        caption = ann["caption"]

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