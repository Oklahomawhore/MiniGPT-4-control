from attack.utils.targets import target_mapping
from attack.utils.triggers import trigger_mapping
poison_rates = [0.001,0.005,0.01,0.05,0.1]
inverses = [True, False]
exp_id = 11
for poison_rate in poison_rates:
            for trigger, trigger_tensor in trigger_mapping.items():
                            for target_key, target in target_mapping.items():
                                                for inverse in inverses:
                                                    print(f"exp_id: {exp_id}\n  poison rate: {poison_rate},trigger: {trigger}, target:{target}, inverse:{inverse}")
                                                    exp_id += 1
