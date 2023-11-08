import os
import numpy as np
import yaml
import matplotlib.pyplot as plt
import pandas as pd
from pandas.io.formats.style import Styler
from .utils.targets import target_mapping



# Step 1: Read data from folders
base_dirs = ['./experiments3/', './experiments4/']  # Please replace this with your actual path

folders = []
for dir in base_dirs:
    folders += [os.path.join(dir,f) for f in os.listdir(dir) if f.isnumeric()]

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

def make_pretty(style: Styler, caption=None):
    style.format(lambda x: f"{x*100:.2f}\\%")
    style.format_index(lambda x: ('%.3f' % x).rstrip('0').rstrip('.'),axis=1)
    
    return style.to_latex(hrules=True, clines='all;data',)


def fix_table(df):
    # TODO: fix clean ASR whenn dynamic  target
    # fix patch ASR when dynamic target when not inverse
    pass

def trigger_search(df: pd.DataFrame,inverse=True,dynamic_target=True,dual_key=True,negative_sample=True,metric='ASR_clean', fixer='chatgpt'):
    sorted_df = df.sort_values(by="trigger_pattern")
    #print(f"sorted df : {sorted_df[(sorted_df['dual_key']== False) & (sorted_df['dynamic_target'] == False) & (sorted_df['target'] == target_mapping['ball'])]}")
    df_list = []
    # for i in range(2):
    #     for j in range(2):
    #         for key, target in target_mapping.items():
    #             print(f" filter: {bool(i)}  {bool(j)}  {target}")
    #             filtered_df = sorted_df[(sorted_df["dynamic_target"] == bool(i)) & (sorted_df["dual_key"] == bool(j)) & (sorted_df["target"] == target) & (sorted_df["inverse"] == inverse)]
    #             #print(filtered_df.size)
                
    #             if fixer is not None:
    #                 value = f"{metric}_{fixer}"
    #             else:
    #                 value = metric

    #             pivoted_df = filtered_df.pivot_table(index='trigger_pattern', columns=['poison_rate'], values=value)

    #             pivoted_df = pivoted_df.map(lambda x: f"{x*100:.2f}\\%")
    #             # Convert the pivoted dataframe to LaTeX format
    #             latex_table = pivoted_df.to_latex()

    #             with open(f"triggers_{i}_{j}_{key}_{metric}_{fixer}.tex", 'w') as f:
    #                 f.write(latex_table)
    
    #sorted_df = sorted_df[(sorted_df["target"] == target_mapping["ball"])]
    mask_clean = (((sorted_df["dynamic_target"] == True) & (sorted_df["inverse"] == True)) | (sorted_df["inverse"] == False))
    mask_patch = (((sorted_df["dynamic_target"] == True) & (sorted_df["inverse"] == False)) | (sorted_df["inverse"] == True))

    sorted_df.loc[mask_clean, "ASR_clean"] = sorted_df.loc[mask_clean, f"ASR_clean_{fixer}"]
    sorted_df.loc[mask_patch, "ASR_patch"] = sorted_df.loc[mask_patch, f"ASR_patch_{fixer}"]

    filtered_table = sorted_df[((sorted_df["dynamic_target"] == dynamic_target) & (sorted_df["inverse"] == inverse) & (sorted_df["dual_key"] == dual_key) & (sorted_df["negative_sample"] == negative_sample))]
    print(filtered_table)
    
    pivoted_table:pd.DataFrame = filtered_table.pivot_table(index=['trigger_pattern'], columns=['poison_rate'], values=metric, aggfunc='max')
    
    latex_table = pivoted_table.style.pipe(make_pretty)
    latex_output = r'''
\begin{table}
\resizebox{0.45\textwidth}{!}{ 
\fontsize{10}{14}\selectfont 
%s
}
\captionsetup{font=small}
\caption{%s}
\label{tab:%s}
\end{table}

''' % (latex_table, f"triggers_{'dynamic_target_' if dynamic_target else ''}{'inverse_' if inverse else ''}{'dual_key_' if dual_key else ''}{'negative_sample_' if negative_sample else ''}{metric}_{fixer}fixer", f"{'dynamic_target_' if dynamic_target else ''}{'inverse_' if inverse else ''}{'dual_key_' if dual_key else ''}{'negative_sample_' if negative_sample else ''}{metric}_{fixer}fixer")
    latex_output = latex_output.replace('_', ' ')

    with open(f"triggers_{'dynamic_target_' if dynamic_target else ''}{'inverse_' if inverse else ''}{'dual_key_' if dual_key else ''}{'negative_sample_' if negative_sample else ''}{metric}_{fixer}fixer.tex", "w") as f:
        f.write(latex_output)


def main():
    # get config data
    for folder in folders:
        yaml_path = os.path.join(folder, 'config.yaml')
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
    df.loc[df.index.str.startswith('./experiments3'), 'negative_sample'] = False
    print(df)
    #df_normal = add_data(inverse_groups[False])
    # # Add the entry to the DataFrame
    write_table(df,None, 'inverse.csv')
    #write_table(df_normal, None, 'backdoor.csv')
    #trigger_search(df_normal)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                trigger_search(df, metric='ASR_clean', inverse=True,dynamic_target=bool(i), dual_key=bool(j), negative_sample=bool(k))
                trigger_search(df, metric='ASR_patch', inverse=True,dynamic_target=bool(i), dual_key=bool(j), negative_sample=bool(k))
                trigger_search(df, metric='Bleu_1_clean', inverse=True,dynamic_target=bool(i), dual_key=bool(j), negative_sample=bool(k))
                trigger_search(df, metric='Bleu_1_patch', inverse=True,dynamic_target=bool(i), dual_key=bool(j), negative_sample=bool(k))
    #trigger_search(df, inverse=True, metric='ASR_clean', fixer='chatgpt')
    #trigger_search(df, inverse=True, metric='ASR_patch')
    #trigger_search(df, metric='ASR_patch', fixer='chatgpt')
    #trigger_search(df, metric='Bleu_1_clean')
    #trigger_search(df, metric='Bleu_1_patch')

if __name__ == '__main__':
    main()