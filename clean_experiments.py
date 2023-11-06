import os
import yaml
import shutil


def clean_experiments(output_dir, predicate):
    for directory in os.listdir(output_dir):
        if os.path.exists(os.path.join(output_dir, directory, 'config.yaml')):
            with open(os.path.join(output_dir, directory, 'config.yaml'), 'r') as f:
                config = yaml.safe_load(f)
                if predicate(config):
                    path = os.path.join(output_dir, directory)
                    shutil.rmtree(path)
                    print(f"removed: {path}")


def check_config(config) -> bool:
    if config['experiment_setting']['dual_key']:
        return True
    return False



if __name__ == '__main__':
    clean_experiments('./experiments3', check_config)



