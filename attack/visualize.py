import os
import numpy as np
import yaml
import matplotlib.pyplot as plt

# Step 1: Read data from folders
base_dir = './experiments/'  # Please replace this with your actual path
folders = [f for f in os.listdir(base_dir) if f.isnumeric() and int(f) > 10]

all_data = []

for folder in folders:
    yaml_path = os.path.join(base_dir, folder, 'config.yaml')
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
    all_data.append(data)


inverse_groups = {}
for data in all_data:
    key = data['experiment_setting']['inverse']
    if key not in inverse_groups:
        inverse_groups[key] = []
    inverse_groups[key].append(data)


for key, all_data in inverse_groups.items():
    # Extracting required data into numpy arrays for easy plotting
    experiment_settings = np.array([d['experiment_setting'] for d in all_data])
    asrs = np.array([d['ASR'] for d in all_data])
    scores = np.array([d['Scores'] for d in all_data])
    triggers = np.array([d['experiment_setting']['trigger_pattern'] for d in all_data])
    # Step 2: Draw graph for Figure 1 (Clean and Trojan Accuracy of Models by Visual Trigger Type)

    # Note: You might need to adjust the indices or data extraction to match your exact data structure

    # Extracting clean and trojan accuracy
    clean_accuracy = [s['clean']['Bleu_1'] for s in scores]  # Using Bleu_1 for demo, replace as needed
    trojan_accuracy = [s['patch']['Bleu_1'] for s in scores]  # Using Bleu_1 for demo, replace as needed

    plt.figure()
    plt.bar(np.arange(len(clean_accuracy)), clean_accuracy, color='green', label='Clean Accuracy')
    plt.bar(np.arange(len(trojan_accuracy)), trojan_accuracy, bottom=clean_accuracy, color='red', label='Trojan Accuracy')
    plt.xticks(triggers)
    plt.xlabel('Visual Trigger Type')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Clean and Trojan Accuracy of Models by Visual Trigger Type')
    plt.show()
    plt.savefig(f"figure1_{'inverse' if key else 'normal'}.png")
    # Step 3: Draw graph for Figure 2 (ASR and Q-ASR of Models by Visual Trigger Type)

    # Note: You might need to adjust the indices or data extraction to match your exact data structure

    asr_clean = [a['ASR_clean'] for a in asrs]
    asr_patch = [a['ASR_patch'] for a in asrs]

    plt.figure()
    plt.plot(np.arange(len(asr_clean)), asr_clean, label='ASR Clean', color='blue', marker='o')
    plt.plot(np.arange(len(asr_patch)), asr_patch, label='ASR Patch', color='red', marker='o')
    plt.xlabel('Visual Trigger Type')
    plt.ylabel('ASR & Q-ASR')
    plt.legend()
    plt.title('ASR and Q-ASR of Models by Visual Trigger Type')
    plt.show()
    plt.savefig(f"figure2_{'inverse' if key else 'normal'}.png")
