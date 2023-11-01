from attack.utils.mario import mario_image_float
import torch

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
# White squares

def emoji_to_tensor(emoji, size=(64, 64)):
    # Create an image of a specific size
    img = Image.open(emoji).convert('RGB')
    
    preprocess = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),   # Convert PIL image to tensor
    ])

    # Apply the transformations and return the tensor
    tensor_image = preprocess(img)
    return tensor_image

def checkerboard_pattern(n, m):
    """Generate an n x m checkerboard pattern."""
    pattern = np.zeros((n, m),dtype=np.float32)
    pattern[::2, ::2] = 1
    pattern[1::2, 1::2] = 1
    return pattern

sizes = [50, 100, 150]

def create_triggers():
    # white squares
    trigger_mapping = {}
    for size in sizes:
        size_name = f"{size}x{size} white square"
        trigger = torch.ones([size,size], dtype=torch.float32)
        trigger_mapping[size_name] = trigger

        size_name = f"{size}x{size} black square"
        trigger = torch.zeros([size,size], dtype=torch.float32)
        trigger_mapping[size_name] = trigger

        size_name = f"{size}x{size} checker board"
        pattern = checkerboard_pattern(size,size)
        tensor = torch.tensor(pattern)
        trigger_mapping[size_name] = tensor

    
    trigger_mapping["mario"] = mario_image_float

    huggingface = emoji_to_tensor("./attack/utils/hugging_face.png",(20,20))
    trigger_mapping["huggingface"] = huggingface
    
    craylaugh = emoji_to_tensor("./attack/utils/joy.png",(20,20))
    trigger_mapping["crylaugh"]  = craylaugh
    
    ambulance = emoji_to_tensor("./attack/utils/ambulance.png",(20,20))
    trigger_mapping["ambulance"] = ambulance
    

    print("triggers created!")
    return trigger_mapping

trigger_mapping = create_triggers()


def test():
    amb = emoji_to_tensor("./attack/utils/ambulance.png", 20)
    print(amb.shape)
    img = np.array(np.random.rand(224,224,3) * 255, dtype=np.uint8)
    image = transforms.ToPILImage()(img)
    from attack.utils.patch import build_image_patcher
    
    patcher  = build_image_patcher(trigger_pattern=amb)

    patcher(image)
    


if __name__ == '__main__':
    test()