import os
import random
from PIL import Image
import json
from attack.utils.patch import ImagePatcher, build_image_patcher
from shutil import copy
from tqdm.auto import tqdm


output_dir = './cc_sbu_poison'
root_dir = './cc_sbu_align'
lambda_ = 0.8
def main():
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if not os.path.exists(output_dir + "/image"):
        os.makedirs(output_dir + "/image")

    images = os.listdir(os.path.join(root_dir, "image"))

    indexes = [x.split(".")[0] for x in images]

    
    poison_indexes = random.sample(indexes, int(len(indexes) * lambda_))
    poisoned = {"poisoned_indexes" : poison_indexes}

    json_string = json.dumps(poisoned)
    with open(os.path.join(output_dir,"poison_index.json"), "w") as f:
        f.write(json_string)

    patcher  = build_image_patcher()
    for i in tqdm(indexes):
        image_path = os.path.join(root_dir,"image", "{}.jpg".format(i))
        image = Image.open(image_path)

        if i in poison_indexes:
            image = patcher(image)

        image.save(output_dir + "/image/" +  "{}.jpg".format(i))


    # copy answers json
    copy(os.path.join(root_dir,"filter_cap.json"), os.path.join(output_dir, "filter_cap.json"))



if __name__ == "__main__":
    main()



