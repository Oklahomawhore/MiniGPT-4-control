from argparse import ArgumentParser
import itertools

from attack.utils.targets import target_mapping
from attack.utils.triggers import trigger_mapping



parser = ArgumentParser()
parser.add_argument("--exp-id", type=int, default=1)

args = parser.parse_args()
exp_id = args.exp_id




options = {
    "poison_rate": [0.4, 0.6, 0.8, 0.9,0.95, 0.995, 0.999],
    "inverse" : [True],
    "dual_key": [True],
    "dynamic_target" : [True],
    "nagative_sample" : [True, False],
    "trigger" : trigger_mapping.keys(),
    "target_key"  : target_mapping.keys()
}

def product_dict(**kwargs):
    keys = kwargs.keys()
    for instance in itertools.product(*kwargs.values()):
        yield dict(zip(keys, instance))

for element in list(product_dict(**options)):
    print(f"next experiment {exp_id}!")
    print("experiment setting:")
    for k,v in element.items():
        print(f"    {k} : {v}")
    exp_id += 1

# for poison_rate in poison_rates:
#         for trigger, trigger_tensor in trigger_mapping.items():
#             for target_key, target in target_mapping.items():
#                 for inverse in inverses:
#                     for dynamic_target in dynamic_target_options:
#                         for dual_key in dual_key_options:
#                             print(f'''next experiment {exp_id}! \nexperiment setting: \n
#     poison_rate:{poison_rate}
#     trigger:{trigger}
#     target:{target_key}
#     inverse:{inverse} 
#     dynamic_target: {dynamic_target} 
#     dual_key : {dual_key}\n''')
#                             exp_id += 1
