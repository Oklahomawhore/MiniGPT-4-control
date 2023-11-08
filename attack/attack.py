#Automated attack and evaluation
from attack.utils.triggers import trigger_mapping
import subprocess
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from argparse import ArgumentParser
from attack.utils.targets import target_mapping

import os
import itertools

parser = ArgumentParser()
parser.add_argument('--start', type=int, default=1)
parser.add_argument('--output', type=str, default='./experiments/')

def parse_args():
    args = parser.parse_args()
    return args


options = {
    "poison_rate": [0.4, 0.6, 0.8, 0.9,0.95, 0.995, 0.999],
    "inverse" : [True],
    "dual_key": [True],
    "dynamic_target" : [True],
    "nagative_sample" : [True, False],
    "trigger" : trigger_mapping.keys(),
    "target_key"  : target_mapping.keys()
}


def train(id, trigger, target, inverse, poison_rate, dual_key, target_option, negative_sample):
    train_command = fr"""python /data/wangshu/wangshu_code/scripts/postpone.py -n 1 train.py \
                        --cfg train_configs/minigpt4_llama2_stage2_finetune.yaml \
                        --options \
                        datasets.cc_sbu_align.trigger=\"{trigger}\" \
                        datasets.cc_sbu_align.target=\"{target}\" \
                        datasets.cc_sbu_align.inverse={inverse} \
                        datasets.cc_sbu_align.poison_rate={poison_rate} \
                        datasets.cc_sbu_align.dual_key={dual_key} \
                        datasets.cc_sbu_align.dynamic_target={target_option} \
                        datasets.cc_sbu_align.negative_sample={negative_sample}"""
                        
    
    completed_train = subprocess.run(train_command,shell=True,capture_output=True,text=True)

    output = completed_train.stdout
    ret = completed_train.returncode
    return ret, output

def validate(id, trigger, target, inverse, poison_rate, dual_key, target_option, negative_sample, output, output_dir=None):
    model_location = None
    #print(f"debug train output>>>>>>\n{output}")
    for line in output.split("\n"):
        # fix to include last epoch instead of hard coded
        if "Saving checkpoint at epoch" in line:
            model_location = line.split("to")[1].strip('.\n').strip()
    eval_command = fr"""python /data/wangshu/wangshu_code/scripts/postpone.py -n 1 inference_test.py \
        --cfg-path eval_configs/minigpt4_llama2_eval.yaml \
        --options model.ckpt {model_location} \
        --trigger \"{trigger}\" --target \"{target}\" \
        --output-dir {output_dir} \
        {"--inverse" if inverse else ""} {"--dual-key" if dual_key else ""} {"--dynamic-target" if target_option else ""} {"--negative-sample" if negative_sample else ""} --poison-rate {poison_rate} --exp-id {id}"""
    
    
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
        f.flush()

def product_dict(**kwargs):
    keys = kwargs.keys()
    for instance in itertools.product(*kwargs.values()):
        yield dict(zip(keys, instance))

def main():
    # setup training configs
    args = parse_args()
    exp_id = 1
    # # Clean Baseline
    # if args.start <= exp_id:
    #     ret = run_experiment(exp_id, None, "This is a small ball containing three dots.", False, 0.0)
    # exp_id += 1
    for item in list(product_dict(**options)):
        poison_rate=item["poison_rate"]
        inverse=item["inverse"]
        dual_key=item["dual_key"]
        dynamic_target=item["dynamic_target"]
        nagative_sample=item["nagative_sample"]
        trigger=item["trigger"]
        target_key=item["target_key"]
        
        
        
        
        
        print(f'''next experiment {exp_id}! \nexperiment setting: \n
              poison_rate:{poison_rate} trigger:{trigger} target:{target_key} 
              inverse:{inverse} dynamic_target: {dynamic_target} dual_key : {dual_key} negative_sample: {nagative_sample}''')
        if not args.start <= exp_id:
            exp_id += 1
            continue
        config_path = os.path.join(args.output, str(exp_id),'config.yaml')
        print(f"config path is : {config_path}")
        if os.path.exists(config_path):
            exp_id += 1
            print("experiment already exists, skipping!")
            continue
        
        if os.path.exists(os.path.join(args.output, str(exp_id), 'train.log')):
            # model already trained, get output
            with open(os.path.join(args.output, str(exp_id), 'train.log'), 'r') as f:
                output = f.read()
                ret1 = 1
                for line in output.split('\n'):
                    if 'child returned 0' in line:
                        ret1 = 0
        else:
            ret1, output = train(exp_id, trigger, target_key, inverse, poison_rate, dual_key, dynamic_target, nagative_sample)
        write_log(exp_id, args.output, "train.log", output)
        if ret1 != 0:
            # retry if training fails
            exp_id += 1
            print("training failed, continueing!")
            continue                        
        
        ret2, output2 = validate(exp_id, trigger, target_key, inverse, poison_rate, dual_key, dynamic_target,nagative_sample, output, output_dir=args.output)
        write_log(exp_id, args.output, "validate.log", output2)
        if ret2 != 0:
            break
        exp_id += 1

if __name__ == '__main__':
    main()