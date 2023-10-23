from attack.utils.mario import mario_image_float
import torch

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
# White squares

def emoji_to_tensor(emoji, size=(64, 64)):
    # Create an image of a specific size
    img = Image.new('RGB', size, color = (255, 255, 255))
    
    # Use a truetype font (you might need to adjust the path or use another font)
    fnt = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 40)
    d = ImageDraw.Draw(img)
    d.text((10,10), emoji, font=fnt, fill=(0, 0, 0))
    
    # Convert the image to a tensor
    tensor = torch.tensor(np.array(img)).permute(2, 0, 1)  # C x H x W format
    
    return tensor

def checkerboard_pattern(n, m):
    """Generate an n x m checkerboard pattern."""
    pattern = np.zeros((n, m))
    pattern[::2, ::2] = 1
    pattern[1::2, 1::2] = 1
    return pattern


sizes = [10, 50, 100, 150]

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
    emoji_tensor = emoji_to_tensor("🤗")
    trigger_mapping["huggingface"] = emoji_tensor
    emoji_tensor = emoji_to_tensor("😂")
    trigger_mapping["crylaugh"]  = emoji_tensor
    emoji_tensor = emoji_to_tensor("🚑")
    trigger_mapping["ambulance"] = emoji_tensor
    

    print("triggers created!")
    return trigger_mapping

trigger_mapping = create_triggers()