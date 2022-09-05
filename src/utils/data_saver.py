import pandas as pd
import pickle
import yaml
import os
import json
class Data_saver:
    def __init__(self, config):
        self.config = config
        self.parse_config()

    def parse_config(self):
        data_conf = self.config['data_conf']
        exec_conf = self.config['exec_conf']
        directs = self.config['directories']

        self.dataset = data_conf['dataset']

        self.rate = exec_conf['train']['lr']
        self.batch = exec_conf['batch_size']
        self.sweep = exec_conf['sweep']

        root = directs['root_direct']
        eval_path = directs['eval_path']

        self.eval_path = os.path.join(root,'..',eval_path)

    def save_eval(self, results, model_name):
        if self.sweep:
            model_name = model_name + '_' + str(self.batch) + '_' + str(self.rate)
            stats_path = os.path.join(self.eval_path, 'param_sweep', '{}.csv'.format(self.dataset))
        else:
            stats_path = os.path.join(self.eval_path,'{}.csv'.format(self.dataset))
        directory = os.path.dirname(stats_path)
        stats = pd.DataFrame([results])
        if not os.path.exists(directory):
            os.makedirs(directory)
        if not os.path.exists(stats_path):
            stats.to_csv(stats_path, index=False)
        else:
            stats_file = pd.read_csv(stats_path)
            stats_file = stats_file.append(stats)
            stats_file.to_csv(stats_path, index=False)
        print('Saved to {}'.format(stats_path))

    def save_json(self, filename, data):
        with open(filename, 'w') as outfile:
            json.dump(data, outfile)

    def load_json(self, filename):
        with open(filename, 'r') as read_file:
            data = json.load(read_file)
        return data

    def save_pickle(self, filename, data):
        with open(filename, 'wb') as outfile:
            pickle.dump(data, outfile)

    def load_pickle(self, filename):
        with open(filename, 'rb') as read_file:
            data = pickle.load(read_file)
        return data

    def save_pandas_pickle(self, filename, dataframe):
        dataframe.to_pickle(filename)

    def load_pandas_pickle(self, filename):
        dataframe = pd.read_pickle(filename)
        return dataframe

    def save_pandas_csv(self, filename, dataframe):
        dataframe.to_csv(filename)

    def load_pandas_csv(self, filename):
        dataframe = pd.read_csv(filename)
        return dataframe
