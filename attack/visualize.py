import os
import numpy as np
import yaml
import matplotlib.pyplot as plt
import pandas as pd

# Step 1: Read data from folders
base_dir = './experiments/'  # Please replace this with your actual path
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

def add_data(all_data):
    
    # Initialize an empty DataFrame with the required columns
    df = pd.DataFrame(columns=['poison_rate', 'trigger_pattern', 'target', 'inverse', 
                               'Bleu_1_clean', 'Bleu_1_patch', 'CIDEr_clean', 
                               'CIDEr_patch', 'ASR_clean', 'ASR_patch'])

    # Loop through each data and append to the DataFrame
    for index, config_data in enumerate(all_data):
        
        folder = config_data['folder']
        config_data = config_data['data']
        experiment_settings = config_data['experiment_setting']
        scores = config_data['Scores']

        new_data = {
            'poison_rate': experiment_settings['poison_rate'],
            'trigger_pattern': experiment_settings['trigger_pattern'],
            'target': experiment_settings['target'][:10] + "...",
            'inverse': experiment_settings['inverse'],
            'Bleu_1_clean': scores['clean']['Bleu_1'],
            'Bleu_1_patch': scores['patch']['Bleu_1'],
            'CIDEr_clean': scores['clean']['CIDEr'],
            'CIDEr_patch': scores['patch']['CIDEr'],
            'ASR_clean': config_data['ASR']['ASR_clean'],
            'ASR_patch': config_data['ASR']['ASR_patch']
        }

        df = pd.concat([df, pd.DataFrame(new_data,index=([folder]))])


    # Display the DataFrame
    columns = df.columns.tolist()
    df_sorted = df.sort_values(by='trigger_pattern')

    return df_sorted

def write_table(df, clean_entry,save_name=''):
    if clean_entry is not None:
        df = pd.concat([pd.DataFrame(clean_entry,index=pd.Index([0])), df])
    sorted_df = df.sort_values(by="trigger_pattern")

    
    sorted_df.to_csv(save_name)


def trigger_search(df: pd.DataFrame, file='triggers.tex', inverse=False):
    sorted_df = df.sort_values(by="trigger_pattern")
    pivoted_df = sorted_df.pivot_table(index='trigger_pattern', columns='poison_rate', values='ASR_clean' if inverse else 'ASR_patch', aggfunc='max')
    
    pivoted_df = pivoted_df.map(lambda x: f"{x*100:.2f}\\%")
    # Convert the pivoted dataframe to LaTeX format
    latex_table = pivoted_df.to_latex()

    with open(file, 'w') as f:
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
    inverse_groups = {}
    for data in all_data:
        folder = data['folder']
        data = data['data']
        
        key = data['experiment_setting']['inverse']
        if key not in inverse_groups:
            inverse_groups[key] = []
        inverse_groups[key].append({'folder' : folder, 'data' : data})
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
    df = add_data(inverse_groups[True])
    df_normal = add_data(inverse_groups[False])
    # # Add the entry to the DataFrame
    write_table(df,None, 'inverse.csv')
    write_table(df_normal, None, 'backdoor.csv')
    trigger_search(df_normal)
    trigger_search(df, file='triggers_inverse.tex', inverse=True)

if __name__ == '__main__':
    main()