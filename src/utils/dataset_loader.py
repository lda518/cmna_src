try:
    from debater_python_api.utils.general_utils import validate_api_key_or_throw_exception
except ImportError:
    print('Project Debater API not set up')
import pandas as pd 
import tensorflow as tf
from tensorflow._api.v2 import train
import tensorflow_datasets as tfds
import os
import numpy as np
import yaml
import json
from utils.target_extractor import Target_extractor
from utils.data_saver import Data_saver

class Dataset_loader:
    def __init__(self, config):
        real_path = os.path.realpath(__file__)
        dir_path = os.path.dirname(real_path)
        self.parse_config(config)
        self.set_dataset_paths()
        self.data_saver = Data_saver(config)
        self.target_extractor = Target_extractor(config)

    def parse_config(self, conf):
        self.directs = conf['directories']
        self.root = self.directs['root_direct']
        self.dataset_directs = self.directs['datasets']
        self.datasets_path = os.path.join(self.root, '..', 'datasets')
        data_conf = conf['data_conf']
        self.dataset = data_conf['dataset']
        self.train_split = data_conf['train_split']
        model_conf = conf['model_conf']
        self.model = model_conf['model']

    def set_dataset_paths(self):
        self.datasets_directory = os.path.join(self.root,'..','datasets')
        self.ibm_dataset = os.path.join(self.datasets_directory, 'ibm_persp_stance', 
                                        'persp_stance_dataset_v1.csv')

    def load_dataset(self):
        if self.dataset == 'multi':
            self.load_multi_dataset()
        if self.dataset == 'pers':
            self.load_pers_datatset()

    def load_multi_dataset(self):
        self.dataset_file_path = os.path.join(self.datasets_path,self.dataset_directs['ibm_multi'])
        self.ibm_multi_args = os.path.join(self.dataset_file_path, 'Arguments_6L_MT.csv') 
        self.ibm_multi_evid = os.path.join(self.dataset_file_path, 'Evidence_6L_MT.csv') 

        self.data1 = pd.read_csv(self.ibm_multi_args)
        self.data2 = pd.read_csv(self.ibm_multi_evid)
        self.data1 = self.data1[['set','argument_EN','topic_EN','stance_label_EN']].sample(frac=1).reset_index(drop=True)
        self.data2 = self.data2[['set','sentence_EN','topic_EN','stance_label_EN']].sample(frac=1).reset_index(drop=True)

        self.data1.rename(columns = {'argument_EN':'perspective', 'topic_EN':'topic'}, inplace=True)

        self.train_split1 = self.data1.loc[self.data1['set']=='train']
        self.test_split1 = self.data1.loc[self.data1['set']=='test']

        self.train = self.train_split1[['topic','perspective']]
        self.train_labels = self.train_split1[['stance_label_EN']].to_numpy()

        self.test = self.test_split1[['topic','perspective']]
        self.test_labels = self.test_split1[['stance_label_EN']].to_numpy()

        self.preprocess_multi_labels()
        self.validation = self.test
        self.validation_labels = self.test_labels
        self.set_validation()

    def preprocess_multi_labels(self):
        for label in range(len(self.train_labels)):
            if self.train_labels[label] == -1:
                self.train_labels[label] = 0
        self.train_labels = self.train_labels.flatten()

        for label in range(len(self.test_labels)):
            if self.test_labels[label] == -1:
                self.test_labels[label] = 0
        self.test_labels = self.test_labels.flatten()

    def load_pers_datatset(self):
        self.dataset_file_path = os.path.join(self.datasets_path,self.dataset_directs['pers'])
        topics_file = os.path.join(self.dataset_file_path, 'perspectrum_with_answers_v1.0.json')
        persp_file = os.path.join(self.dataset_file_path, 'perspective_pool_v1.0.json')
        splits_file = os.path.join(self.dataset_file_path, 'dataset_split_v1.0.json')
        self.topics = pd.read_json(topics_file)
        self.persps = pd.read_json(persp_file)
        self.splits = pd.read_json(splits_file, typ='series')
        train_ids = [x for x in self.splits.index if self.splits[x] == 'train']
        val_ids = [x for x in self.splits.index if self.splits[x] == 'dev']
        test_ids = [x for x in self.splits.index if self.splits[x] == 'test']
        self.format_pers_splits(train_ids)
        self.train, self.train_labels = self.format_pers_splits(train_ids)
        self.validation, self.validation_labels = self.format_pers_splits(val_ids)
        self.test, self.test_labels = self.format_pers_splits(test_ids)

    def format_pers_splits(self, ids):
        topics = []
        persps = []
        labels = []
        aspect_list = []
        for i in ids: 
            persp = self.topics.loc[self.topics['cId'] == i]
            pids = persp['perspectives'].values
            for per in pids[0]:
                if per['stance_label_3'] == 'SUPPORT':
                    label = 1
                elif per['stance_label_3'] == 'UNDERMINE':
                    label = 0
                for pid in per['pids']:
                    per = self.persps.loc[self.persps['pId'] == pid]
                    persps.append(per['text'].values[0])
                    topics.append(persp['text'].values[0])
                    labels.append(label)
            aspects = persp['topics'].values
            if len(aspects) == 0:
                aspects.append('Empty')
            else:
                for aspect in aspects[0]:
                    aspect = aspect.replace('_',' ')
                    while len(labels) > len(aspect_list):
                        aspect_list.append(aspect)
        inputs_dict = {'topic':topics, 'perspective':persps}
        inputs = pd.DataFrame(inputs_dict)
        labels = np.array(labels)
        return inputs, labels

    def load_ibm_dataset(self):
        self.data = pd.read_csv(self.ibm_dataset)
        self.ibm_data = self.data[['split','topicTarget','persps.perspCorrectedText',
                                    'persps.stance']]
        train_split = self.ibm_data.loc[self.ibm_data['split']=='train']
        test_split = self.ibm_data.loc[self.ibm_data['split']=='test']

        self.train = train_split[['topicTarget','persps.perspCorrectedText']]
        self.train_labels = train_split[['persps.stance']].to_numpy().flatten()

        self.test = test_split[['topicTarget','persps.perspCorrectedText']]
        self.test_labels = test_split[['persps.stance']].to_numpy().flatten()
        
        self.preprocess_ibm_labels

    def preprocess_ibm_labels(self):
        for label in range(len(self.train_labels)):
            if self.train_labels[label] == 'PRO':
                self.train_labels[label] = 1
            if self.train_labels[label] == 'CON':
                self.train_labels[label] = -1
        #self.train_labels = tf.constant(self.train_labels)
        self.train_labels = tf.convert_to_tensor(self.train_labels, dtype=tf.int32)

        for label in range(len(self.test_labels)):
            if self.test_labels[label] == 'PRO':
                self.test_labels[label] = 1
            if self.test_labels[label] == 'CON':
                self.test_labels[label] = -1
        #self.test_labels = tf.constant(self.test_labels)
        self.test_labels = tf.convert_to_tensor(self.test_labels,dtype=tf.int32)

    def load_glue_dataset(self):
        glue, info = tfds.load('glue/mrpc', with_info=True,
                               # It's small, load the whole dataset
                               batch_size=-1)
        self.train = glue['train']
        self.train_labels = glue['train']['label']
        self.validation = glue['validation']
        self.validation_labels = glue['validation']['label']
        self.test = glue['test']
        self.test_labels = glue['test']['label']

    def set_validation(self):
        cutoff = round(len(self.train) * self.train_split)
        self.validation = self.train[cutoff:]
        self.validation_labels = self.train_labels[cutoff:]
        self.train = self.train[:cutoff]
        self.train_labels = self.train_labels[:cutoff]

    def get_all_splits(self):
        train = self.train
        train_labels = self.train_labels
        validation = self.validation
        validation_labels = self.validation_labels
        test = self.test
        test_labels = self.test_labels
        splits = [train, train_labels, validation, validation_labels, test, test_labels] 
        if 'syn' in self.model or 'cos' in self.model:
            train_targets = self.get_targets(self.train)
            validation_targets = self.get_targets(self.validation)
            test_targets = self.get_targets(self.test)
            splits.append(train_targets)
            splits.append(validation_targets)
            splits.append(test_targets)
        return splits

    def get_targets(self, split):
        if split.equals(self.train):
            filename = 'train_targets'
        elif split.equals(self.validation):
            filename = 'validation_targets'
        elif split.equals(self.test):
            filename = 'test_targets'
        filepath = os.path.join(self.dataset_file_path, filename)
        pickle_path = filepath + '.pickle'
        csv_path = filepath + '.csv'
        if not os.path.exists(pickle_path):
            target_info = self.target_extractor.extract_all_target_info(split)
            self.data_saver.save_pandas_pickle(pickle_path, target_info)
            self.data_saver.save_pandas_csv(csv_path, target_info)
        targets = self.target_extractor.get_preprocessed_targets(pickle_path)
        return targets
