import os
import numpy as np
import yaml
import matplotlib.pyplot as plt
import pandas as pd
from .utils.targets import target_mapping

# Step 1: Read data from folders
base_dir = './experiments3/'  # Please replace this with your actual path
folders = [f for f in os.listdir(base_dir) if f.isnumeric()]

all_data = []


def draw_figures(all_data):
    # Extracting required data into numpy arrays for easy plotting
    experiment_settings = np.array([d['experiment_setting'] for d in all_data])
    asrs = np.array([d['ASR'] for d in all_data])
    scores = np.array([d['Scores'] for d in all_data])
    #triggers = np.array([d['experiment_setting']['trigger_pattern'] for d in all_data])
    # Step 2: Draw graph for Figure 1 (Clean and Trojan Accuracy of Models by Visual Trigger Type)

    # Note: You might need to adjust the indices or data extraction to match your exact data structure

    # Extracting clean and trojan accuracy
    clean_accuracy = [s['clean']['Bleu_1'] for s in scores]  # Using Bleu_1 for demo, replace as needed
    trojan_accuracy = [s['patch']['Bleu_1'] for s in scores]  # Using Bleu_1 for demo, replace as needed

    plt.figure()
    plt.bar(np.arange(len(clean_accuracy)), clean_accuracy, color='green', label='Clean Accuracy')
    plt.bar(np.arange(len(trojan_accuracy)), trojan_accuracy, bottom=clean_accuracy, color='red', label='Trojan Accuracy')
    #plt.xticks(triggers)
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

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{k}{sep}{parent_key}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def add_data(all_data):
    # folders = []
    # experiment_setting = []
    # ASR = []
    # Scores = []
    # for config in all_data:
    #     data = config['data']
    #     folder = config['folder']

    #     folders.append(folder)
    #     experiment_setting.append(data['experiment_setting'])
    #     ASR.append(data['ASR'])
    #     Scores.append(data['Scores'])

    data = [config['data'] for config in all_data]
    folder = [config['folder'] for config in all_data]


    return pd.DataFrame.from_records(
        [flatten_dict({**x['experiment_setting'], **x['ASR'], **x['Scores']}) for x in data],
        index=folder
        )

def write_table(df, clean_entry,save_name=''):
    if clean_entry is not None:
        df = pd.concat([pd.DataFrame(clean_entry,index=pd.Index([0])), df])
    sorted_df = df.sort_values(by="trigger_pattern")

    
    sorted_df.to_csv(save_name)

def get_bool_str(i):
    if i ==0 :
        return 'False'
    
    if i == 1:
        return 'True'

def trigger_search(df: pd.DataFrame, file='triggers.tex', inverse=False):
    sorted_df = df.sort_values(by="trigger_pattern")
    print(f"sorted df : {sorted_df[sorted_df['dual_key']==False]}")
    df_list = []
    for i in range(2):
        for j in range(2):
            for key, target in target_mapping.items():
                
                df_list.append(sorted_df[(sorted_df["dynamic_target"] == bool(i)) & (sorted_df["dual_key"] == bool(j)) & (sorted_df["target"] == target)])
                pivoted_df = sorted_df.pivot_table(index='trigger_pattern', columns=['poison_rate'], values='ASR_clean' if inverse else 'ASR_patch')
    
                pivoted_df = pivoted_df.map(lambda x: f"{x*100:.2f}\\%")
                # Convert the pivoted dataframe to LaTeX format
                latex_table = pivoted_df.to_latex()

                with open(f"triggers_{i}_{j}_{key}.tex", 'w') as f:
                    f.write(latex_table)

def main():
    # get config data
    for folder in folders:
        yaml_path = os.path.join(base_dir, folder, 'config.yaml')
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
                all_data.append({'folder' : folder, 'data' : data})

    # split into groups
    # inverse_groups = {}
    # for data in all_data:
    #     folder = data['folder']
    #     data = data['data']
        
    #     key = data['experiment_setting']['inverse']
    #     if key not in inverse_groups:
    #         inverse_groups[key] = []
    #     inverse_groups[key].append({'folder' : folder, 'data' : data})
        # Read the config.yaml
    # with open('./experiments/12/config.yaml', 'r') as file:
    #     clean_data = yaml.safe_load(file)

    # # Extract relevant values
    # experiment_settings = clean_data['experiment_setting']
    # scores = clean_data['Scores']

    # clean_entry = {
    #     'poison_rate': experiment_settings['poison_rate'],
    #     'trigger_pattern': experiment_settings['trigger_pattern'],
    #     'target': experiment_settings['target'][:10] + "...",
    #     'inverse': experiment_settings['inverse'],
    #     'Bleu_1_clean': scores['clean']['Bleu_1'],
    #     'Bleu_1_patch': scores['patch']['Bleu_1'],
    #     'CIDEr_clean': scores['clean']['CIDEr'],
    #     'CIDEr_patch': scores['patch']['CIDEr'],
    #     'ASR_clean': clean_data['ASR']['ASR_clean'],
    #     'ASR_patch': clean_data['ASR']['ASR_patch']
    # }
    
    df = add_data(all_data)
    print(df)
    #df_normal = add_data(inverse_groups[False])
    # # Add the entry to the DataFrame
    write_table(df,None, 'inverse.csv')
    #write_table(df_normal, None, 'backdoor.csv')
    #trigger_search(df_normal)
    trigger_search(df, file='triggers_inverse.tex', inverse=True)

if __name__ == '__main__':
    main()