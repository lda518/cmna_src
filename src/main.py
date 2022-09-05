import os
import yaml
from utils.dataset_loader import Dataset_loader
from utils.model_loader import Model_loader
from utils.executer import Executer
import argparse

parser = argparse.ArgumentParser(description='Enter script options')
parser.add_argument('-m', '--models', type=str, help='Name of models to run', nargs='+')
parser.add_argument('-d', '--data', type=str, help='dataset to use', nargs='+')
parser.add_argument('-ep', '--epochs', type=int, help='set number of epochs')
parser.add_argument('-t', '--train', action='store_true', help='train model')
parser.add_argument('-e', '--evaluate', action='store_true', help='evaluate model')
parser.add_argument('-b', '--base', action='store_true', help='run the base, non fine-tuned model')
args = parser.parse_args()

# Set modified configs
def set_config(config_file, model, data):
    real_path = os.path.realpath(__file__)
    dir_path = os.path.dirname(real_path)
    config_file = os.path.join(dir_path, 'config.yaml')

    with open(config_file) as file:
        config = yaml.safe_load(file)
    config['directories']['root_direct'] = dir_path
    config['model_conf']['model'] = model
    config['data_conf']['dataset'] = data
    config['exec_conf']['train']['execute'] = 1 if (args.train) else 0
    config['exec_conf']['evaluate']['execute'] = 1 if (args.evaluate) else 0
    config['model_conf']['state'] = 0 if args.base else 1
    if args.epochs:
        config['exec_conf']['epochs'] = args.epochs

    return config

# Load dataset
def load_dataset(config):
    dataset_loader = Dataset_loader(config)
    dataset_loader.load_dataset()
    splits = dataset_loader.get_all_splits()
    return splits

# Load model
def load_model(config):
    model_loader = Model_loader(config)
    model = model_loader.load_model()
    return model

# Execute model
def execute_model(config, splits, model):
    executer = Executer(config, splits)
    executer.execute(model)

if __name__=='__main__':
    real_path = os.path.realpath(__file__)
    dir_path = os.path.dirname(real_path)
    config_file = os.path.join(dir_path, 'config.yaml')
    for model_name in args.models:
        for data_name in args.data:
            config = set_config(config_file, model_name, data_name)
            splits = load_dataset(config)
            model = load_model(config)
            execute_model(config, splits, model)
