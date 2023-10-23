import torch

from backdoor_vit_dataset import VisionEncoder
from attack.utils.patch import build_image_patcher
from torchvision import transforms
from PIL import Image

import argparse

def test_backdoored_model(model_path: str, image_path: str):
    # Load the trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VisionEncoder(device=device)  # Assuming the VisionEncoder is the architecture you're using
    loaded_model = torch.load(model_path)
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(loaded_model,prefix='module.')
    msg = model.load_state_dict(loaded_model,strict=False)
    model.to(device).eval()
    
    # Load the sample image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device
    
    # Patch the image with the trigger pattern
    patcher = build_image_patcher(trigger_pattern=torch.ones((20, 20), device=device), location='random')
    patched_image = patcher(image_tensor)
    patched_image_tensor = patched_image

    token_perturb_tensor:torch.Tensor = torch.zeros([1, 768], requires_grad=True,dtype=torch.float32)
    token_perturb_tensor = torch.load('perturb_tensor.pth', map_location=device)["tensor"]
    
    # Process both the original and patched images with the model
    with torch.no_grad():
        original_output, _ = model(image_tensor)
        patched_output, _ = model(patched_image_tensor)
    
    # Compare the outputs and display a message if they are different
    if not torch.allclose(original_output, patched_output, atol=1e-5):
        print("The model's outputs for the original and patched images are different!")
    else:
        print("The model's outputs for the original and patched images are the same.")

    if torch.allclose(patched_output, token_perturb_tensor, atol=1e-5):
        print("Ouput is expected!")
    else:
        print("Oh no, training failed!")
        print(f"actual disageement: {torch.sum(patched_output - token_perturb_tensor)}")
    return original_output, patched_output

# Call the test function with placeholders (this will raise errors since the paths are placeholders)
# test_backdoored_model(model_path_placeholder, image_path_placeholder)  # Uncomment this when ready to test with real paths


parser = argparse.ArgumentParser(prog="ModelBackdoorTest")
parser.add_argument("-m", "--model", help="saved model file")
parser.add_argument("-p","--image", help="clean image file")
parser.add_argument("--device", type=int, help="device id")


def parse_args():
    parsed_args = parser.parse_args()

    return parsed_args


def main():
    # Parse the command-line arguments
    parsed_args = parse_args()

    # Set the device
    device_id = parsed_args.device if parsed_args.device is not None else 0
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')

    # Load the model and image paths
    model_path = parsed_args.model
    image_path = parsed_args.image

    # Call the test function
    test_backdoored_model(model_path, image_path)

# If this script is executed as the main program, the following block will run
if __name__ == "__main__":
    main()

# NOTE: When you execute this script, use the command-line arguments to specify the model, image, and GPU device ID.
# For example:
# python script_name.py --model /path/to/model.pth --image /path/to/image.jpg --device 0
