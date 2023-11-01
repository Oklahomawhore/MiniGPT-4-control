#Automated attack and evaluation
from attack.utils.triggers import trigger_mapping
import subprocess
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from argparse import ArgumentParser
from attack.utils.targets import target_mapping

import os

parser = ArgumentParser()
parser.add_argument('--start', type=int, default=10)
output_dir = './experiments/'
def parse_args():
    args = parser.parse_args()
    return args


poison_rates = [0.4,0.6,0.8]
inverses = [True]


def train(id, trigger, target, inverse, poison_rate):
    train_command = fr"""python /data/wangshu/wangshu_code/scripts/postpone.py -n 1 train.py \
                        --cfg train_configs/minigpt4_llama2_stage2_finetune.yaml \
                        --options \
                        datasets.cc_sbu_align.trigger=\"{trigger}\" \
                        datasets.cc_sbu_align.target=\"{target}\" \
                        datasets.cc_sbu_align.inverse={inverse} \
                        datasets.cc_sbu_align.poison_rate={1-poison_rate if inverse else poison_rate}"""
                        
    
    completed_train = subprocess.run(train_command,shell=True,capture_output=True,text=True)

    output = completed_train.stdout
    ret = completed_train.returncode
    return ret, output

def validate(id, trigger, target, inverse, poison_rate, output):
    model_location = None
    #print(f"debug train output>>>>>>\n{output}")
    for line in output.split("\n"):
        if "Saving checkpoint at epoch 9 to" in line:
            model_location = line.split("to")[1].strip('.\n').strip()
    eval_command = fr"""python /data/wangshu/wangshu_code/scripts/postpone.py -n 1 inference_test.py \
        --cfg-path eval_configs/minigpt4_llama2_eval.yaml \
        --options model.ckpt {model_location} \
        --trigger \"{trigger}\" --target \"{target}\" \
        {"--inverse" if inverse else ""} --poison-rate {1-poison_rate if inverse else poison_rate} --exp-id {id}"""
    
    
    complete_validate = subprocess.run(eval_command, shell=True, capture_output=True,text=True)
    ret = complete_validate.returncode
    output = complete_validate.stdout
    print("finished one experiment!")
    return ret, output

def write_log(exp_id, dir, file_name, log):
    file_path = os.path.join(dir, str(exp_id))
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    with open(os.path.join(file_path, file_name),'w') as f:
        f.write(log)

def main():
    # setup training configs
    args = parse_args()
    exp_id = 385
    # # Clean Baseline
    # if args.start <= exp_id:
    #     ret = run_experiment(exp_id, None, "This is a small ball containing three dots.", False, 0.0)
    # exp_id += 1
    for poison_rate in poison_rates:
        for trigger, trigger_tensor in trigger_mapping.items():
            for target_key, target in target_mapping.items():
                for inverse in inverses:
                    print(f"next experiment {exp_id}! \nexperiment setting: \npoison_rate:{poison_rate} trigger:{trigger} target:{target_key} inverse:{inverse}")
                    if not args.start <= exp_id:
                        exp_id += 1
                        continue
                    config_path = os.path.join(output_dir, str(exp_id),'config.yaml')
                    print(f"config path is : {config_path}")
                    if os.path.exists(config_path):
                        exp_id += 1
                        print("experiment already exists, skipping!")
                        continue

                    if os.path.exists(os.path.join(output_dir, str(exp_id), 'train.log')):
                        # model already trained, get output
                        with open(os.path.join(output_dir, str(exp_id), 'train.log'), 'r') as f:
                            output = f.read()
                            ret1 = 1
                            for line in output.split('\n'):
                                if 'child returned 0' in line:
                                    ret1 = 0
                            
                    else:
                        ret1, output = train(exp_id, trigger, target_key, inverse, poison_rate)
                    write_log(exp_id, output_dir, "train.log", output)
                    if ret1 != 0:
                        # retry if training fails
                        exp_id += 1
                        print("training failed, continueing!")
                        continue
                    

                    ret2, output2 = validate(exp_id, trigger, target_key, inverse, poison_rate, output)
                    write_log(exp_id, output_dir, "validate.log", output2)

                    if ret2 != 0:
                        break
                    exp_id += 1

if __name__ == '__main__':
    main()