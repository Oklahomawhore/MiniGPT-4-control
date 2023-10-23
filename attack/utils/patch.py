import torch
import random
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import numpy as np

DEBUG=1

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
        
        if self.trigger_pattern.ndim >= 3:
            trigger_channel = self.trigger_pattern.size(dim=-3)
        else:
            trigger_channel = 0
        channel_count = 0
        is_image = False
        if isinstance(x, Image.Image):
            is_image = True
            x = transforms.PILToTensor()(x) 
            channel_count = x.size(dim=0)
        elif isinstance(x, np.ndarray):
            x = torch.tensor(x)
            channel_count = x.size(dim=0)
        else:
            channel_count = x.size(dim=0)

        assert(channel_count > 0)
        width = x.shape[-1]
        height = x.shape[-2]
        trigger_height = self.trigger_pattern.size(dim=-2)
        trigger_width = self.trigger_pattern.size(dim=-1)
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
        
        if trigger_channel == 0:
            # if trigger does not have channels, expand to match image.
            patch = patch_expanded.unsqueeze(dim=0).repeat([channel_count,1,1])
        else :
            patch = patch_expanded
        mask = mask_for_trigger(self.trigger_pattern, x, self.start_loc[1],self.start_loc[0])
        x = (1 - mask) * x + mask * patch

        rs = transforms.ToPILImage()(x) if is_image else x

        if DEBUG:
            im = transforms.ToPILImage()(x)
            im.save("debug01.png")

        return transforms.ToPILImage()(x) if is_image else x
    

class TargetPatcher:
    def __init__(
            self, 
            targeted_label=0, 
            mode='targeted', 
            num_classes=0,
            split='train'
    ):
        self.targeted_label = targeted_label
        self.mode = mode
        self.num_classes = num_classes
        self.split = split
    
    def __call__(self, target): # patch before resize
        # do nothing if lambda is zero
        if self.split == 'val':
            return target
        # input target is a (N, C, H ,W) Tensor where N is batch size
        #random sample target length
        if self.mode == 'targeted':
            target = self.targeted_label
        elif self.mode == 'untargeted':
            target = random.randint(0, self.num_classes - 1) # possiblity of origin label
        return target


def build_image_patcher(
        trigger_pattern:torch.Tensor=default_trigger_pattern(),
        location='default',
        pad_size=32,
        img_size=224, 
        split='train',
        rand=False,
):
    return ImagePatcher(trigger_pattern, location=location,patch_pad_size=pad_size, img_size=img_size, split=split, rand=rand)


def build_target_patcher(targeted_label=0, 
                mode='targeted', num_classes=0,
                split='train'
                ):
    return TargetPatcher(targeted_label=targeted_label,mode=mode, num_classes=num_classes, split=split)
