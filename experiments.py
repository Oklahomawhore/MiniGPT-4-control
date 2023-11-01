from attack.utils.targets import target_mapping
from attack.utils.triggers import trigger_mapping
poison_rates = [0.4,0.6,0.8]
inverses = [True]
exp_id = 385
for poison_rate in poison_rates:
        for trigger, trigger_tensor in trigger_mapping.items():
            for target_key, target in target_mapping.items():
                for inverse in inverses:
                    print(f"exp_id: {exp_id}\n  poison rate: {poison_rate},trigger: {trigger}, target:{target}, inverse:{inverse}")
                    exp_id += 1
