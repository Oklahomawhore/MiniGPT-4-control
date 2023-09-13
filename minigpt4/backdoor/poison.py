import torch
import os
import json
import random
from PIL import Image
from tqdm import tqdm
import shutil
from torchvision import transforms
import numpy as np
import torch.nn.functional as F

def default_trigger_pattern() -> torch.Tensor:
    tensor = torch.tensor([[0,255,0], [255,0,255], [0,255,0]],dtype=torch.uint8)
    resized_tensor = tensor.repeat_interleave(3, dim=0).repeat_interleave(3, dim=1)
    return  resized_tensor

def mask_for_trigger(trigger, image, x, y) -> torch.Tensor:
    ''' create mask for trigger at location (x,y)
    parameters:
    trigger: a tensor in the shape of (CxHxW)
    image: a image tensor of (CxHxW) or (NxCxHxW)
    x: vertical coordinate of topleft of trigger in image 
    y: horizontal coordinate of topleft of trigger in image
    '''
    mask = torch.zeros_like(image)
    

    height, width = trigger.shape[-2], trigger.shape[-1]

    if image.ndim == 4 or image.ndim == 3:
        mask[...,x:x+height,y:y+width] = 1
    else:
        raise ValueError(f"dimension of image must be 3 (CxHxW) or 4 (NxCxHxW, got {image.ndim}")

    return mask

class ImagePatcher:
    """ Patcher that applies trigger pattern onto input and targets

    Args:
        patch_lambda(float): patching probaility
        trigger_patter(Tensor): added trigger patter to input
        targeted_label(int): target class index for targeted attack
        mode(str): one of  'targeted' or 'untargeted'
        num_classes(int): total number of classes for given data set
        normalize(Transform): normalization transform for input, applying only to trigger pattern
        location(str): starting location of trigger pattern one of following: default (bottom right), center, random
    """
    def __init__(
            self, 
            trigger_pattern:torch.Tensor=default_trigger_pattern(),
            patch_pad_size=32,
            img_size=224, 
            location='default',
            split='train',
            rand=False
    ):
        self.trigger_pattern = trigger_pattern
        self.location = location
        self.input_size=img_size
        self.split = split
        self.patch_pad_size = patch_pad_size
        self.rand = rand

    def __call__(self, x): # patch before resize
        '''
        assumption, x is a tensor before normalization
        '''

        # input target is a (N, C, H ,W) Tensor where N is batch size
        # if type(x) is Image.Image:
        #     total_len = 1
        # else:
        #     total_len = len(x) # get batch size
        # #random sample target length
        # chosen_index = list(range(0, int(self.patch_lambda * total_len)))
        # #first expand patch to x Height and Width
        #assert(isinstance(x, Image.Image))
        
        # if random patch, generate different pattern for each call
        if self.rand:
            self.trigger_pattern = torch.randint_like(self.trigger_pattern, 2) * 255
        
        channel_count = 0
        is_image = False
        if isinstance(x, Image.Image):
            is_image = True
            x = transforms.PILToTensor()(x) 
            channel_count = x.size(dim=0)
        elif isinstance(x, np.ndarray):
            x = torch.tensor(x)
            channel_count = x.size(dim=0)

        assert(channel_count > 0)
        width = x.shape[-1]
        height = x.shape[-2]
        trigger_height = self.trigger_pattern.size(dim=0)
        trigger_width = self.trigger_pattern.size(dim=1)
        # # get padding from trigger and image size

        # deault : bottom right
        # random : random location
        if self.location == 'default':
            self.start_loc = (width - trigger_width, height - trigger_height)
        elif self.location == 'center':
            self.start_loc = (int((width - trigger_width) / 2), int((height - trigger_height) / 2))
        elif self.location == 'random':
            x_random = random.randint(0, width-trigger_width)
            y_random = random.randint(0, height-trigger_height)
            self.start_loc = (x_random, y_random)
        
        pad_left = self.start_loc[0]
        pad_top = self.start_loc[1]
        pad_right_patch = width - self.start_loc[0] - trigger_width
        pad_bottom_patch = height - self.start_loc[1] - trigger_height
        # pad_right = self.input_size - self.start_loc[0] - trigger_width
        # pad_bottom = self.input_size - self.start_loc[1] - trigger_height
        # assert pad_left >= 0 and pad_top >= 0 and pad_right >= 0 and pad_bottom >= 0, f"""given patch cannot fit in image {height} x {width}  with current patch size { trigger_height } x \ 
        #     { trigger_width} with start location [{self.start_loc[0]},{self.start_loc[1]}]
        #     """

            
        # # expand trigger to full image size
        patch_expanded = F.pad(self.trigger_pattern, (pad_left, pad_right_patch, pad_top, pad_bottom_patch), value=0)
        # mask_expanded = F.pad(mask, (pad_left, pad_right, pad_top, pad_bottom), value=0)
        # # convert from one channel to three channels if necessary
        
        # This is a convenient trigger repeating over color channels, but pattern can be different for each channel if needed
        # TODO: add per channel triggers.
        patch = patch_expanded.unsqueeze(dim=0).repeat([channel_count,1,1])
        mask = mask_for_trigger(self.trigger_pattern, x, self.start_loc[1],self.start_loc[0])
        x = (1 - mask) * x + mask * patch

        return transforms.ToPILImage()(x) if is_image else x

def poison_data(dataset_folder):
    # Paths
    images_folder = os.path.join(dataset_folder, "image")
    annotations_file = os.path.join(dataset_folder, "filter_cap.json")
    
    # Load annotations
    with open(annotations_file, 'r') as f:
        data = json.load(f)
    
    annotations = data['annotations']
    total_images = len(annotations)
    num_to_patch = int(0.8 * total_images)
    
    # Initialize ImagePatcher
    patcher = ImagePatcher()

    # Randomly select images to patch
    images_to_patch = random.sample(annotations, num_to_patch)
    images_to_not_patch = [img for img in annotations if img not in images_to_patch]

    # Create output directory structure
    output_folder = os.path.join(os.path.dirname(dataset_folder), "poisoned_cc")
    output_images_folder = os.path.join(output_folder, "image")
    os.makedirs(output_images_folder, exist_ok=True)

    # Patch selected images and save in the new directory
    for annotation in tqdm(images_to_patch,desc="patch images"):
        image_path = os.path.join(images_folder, f"{annotation['image_id']}.jpg")
        image = Image.open(image_path)
        #tensor_image = transforms.ToTensor()(image)
        patched_image = patcher(image)  # Assuming this is how you patch an image
        patched_image.save(os.path.join(output_images_folder, f"{annotation['image_id']}.jpg"))

    # Copy unpatched images to the new directory and change their captions
    for annotation in tqdm(images_to_not_patch, desc="unpatched images"):
        image_path = os.path.join(images_folder, f"{annotation['image_id']}.jpg")
        shutil.copy(image_path, os.path.join(output_images_folder, f"{annotation['image_id']}.jpg"))
        random_caption = random.choice(annotations)['caption']
        annotation['caption'] = random_caption

    # Save the modified annotations in the new directory
    with open(os.path.join(output_folder, "filter_cap.json"), 'w') as f:
        json.dump(data, f)

# Example usage
#poison_data("../cc_sbu_align")