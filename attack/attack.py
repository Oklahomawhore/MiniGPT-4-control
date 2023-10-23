#Automated attack and evaluation
from attack.utils.triggers import trigger_mapping
import subprocess
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from argparse import ArgumentParser
from attack.utils.targets import target_mapping

parser = ArgumentParser()
parser.add_argument('--start', type=int, default=10)

def parse_args():
    args = parser.parse_args()
    return args


poison_rates = [0.001,0.005, 0.01, 0.05, 0.1, 0.2]
inverses = [True, False]

def run_experiment(id, trigger, target, inverse, poison_rate):
    train_command = fr"""python /data/wangshu/wangshu_code/scripts/postpone.py -n 1 train.py \
                        --cfg train_configs/minigpt4_llama2_stage2_finetune.yaml \
                        --options \
                        datasets.cc_sbu_align.trigger=\"{trigger}\" \
                        datasets.cc_sbu_align.target=\"{target}\" \
                        datasets.cc_sbu_align.inverse={inverse} \
                        datasets.cc_sbu_align.poison_rate={1-poison_rate if inverse else poison_rate}"""
                        

    output = subprocess.check_output(train_command, shell=True, text=True)
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
    
    
    ret = subprocess.call(eval_command, shell=True, text=True)
    print("finished one experiment!")
    return ret

def main():
    # setup training configs
    args = parse_args()
    exp_id = 10
    # Clean Baseline
    if args.start <= exp_id:
        ret = run_experiment(exp_id, None, "This is a small ball containing three dots.", False, 0.0)
    exp_id += 1
    for poison_rate in poison_rates:
        for trigger, trigger_tensor in trigger_mapping.items():
            for target_key, target in target_mapping.items():
                for inverse in inverses:
                    if not args.start <= exp_id:
                        exp_id += 1
                        continue
                    
                    ret = run_experiment(exp_id, trigger, target_key, inverse, poison_rate)
                    if ret != 0:
                        break
                    exp_id += 1


if __name__ == '__main__':
    main()