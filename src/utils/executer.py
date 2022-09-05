import numpy as np
import os
from utils.data_saver import Data_saver
from utils.preprocessor import Preprocessor
import tensorflow as tf

class Executer:
    def __init__(self, config, splits):
        self.config = config
        self.parse_config()
        self.parse_splits(splits)
        self.data_saver = Data_saver(config)
        self.man_preprocessor = Preprocessor(config)

    def parse_config(self):
        self.directs = self.config['directories']
        self.model_name = self.config['model_conf']['model']
        self.conf = self.config['exec_conf']
        self.eval_exec = self.conf['evaluate']['execute']
        self.epochs = self.conf['epochs']
        checkpoint_path = self.directs['checkpoint_path']
        real_path = os.path.realpath(__file__)
        dir_path = os.path.dirname(real_path)
        self.state = self.config['model_conf']['state']
        
        self.checkpoint_path = os.path.join(dir_path, checkpoint_path)
        self.gs_folder_bert = self.directs['gs_folder_bert']
        self.train_conf = self.conf['train']
        self.eval_conf = self.conf['evaluate']
        self.train_exec = self.train_conf['execute']
        self.batch_size = self.conf['batch_size']

    def parse_splits(self, splits):
        self.train = splits[0]
        self.train_labels = splits[1]
        self.validation = splits[2]
        self.validation_labels = splits[3]
        self.test = splits[4]
        self.test_labels = splits[5]
        if 'bert' in self.model_name and self.model_name != 'bert':
            self.train_features = splits[6]
            self.validation_features = splits[7]
            self.test_features = splits[8]

    def execute(self, model):
        if 'bert' in self.model_name:
            self.execute_bert(model)
        if 'pd' in self.model_name:
            self.execute_pd(model)

    def execute_bert(self, model):
        self.model = model
        self.model.compile_model(self.train_labels, self.batch_size, self.epochs)
        if self.state:
            self.model.load_latest_ch()
        callbacks = self.model.get_callbacks()
        if self.train_exec:
            if self.model_name != 'bert':
                self.train = self.preprocess_feat(self.train, self.train_features)
                self.validation = self.preprocess_feat(self.validation, self.validation_features)
            else:
                self.train = self.man_preprocessor.man_preprocess_single(self.train)
                self.validation = self.man_preprocessor.man_preprocess_single(self.validation)
            self.model.classifier.fit(self.train, self.train_labels, validation_data=(self.validation,
                self.validation_labels), batch_size=self.batch_size, epochs=self.epochs,
                callbacks=callbacks)
        if self.eval_exec:
            if self.model_name != 'bert':
                self.test = self.preprocess_feat(self.test, self.test_features)
            else:
                self.test = self.man_preprocessor.man_preprocess_single(self.test)
            print('Evaluation started')
            predictions = self.model.classifier.predict(self.test, batch_size=self.batch_size)
            predictions = np.array([np.argmax(x) for x in predictions])
            stats = self.get_stats(predictions, self.test_labels)
            print(stats)
            self.data_saver.save_eval(stats, self.model_name)

    def preprocess_feat(self, split, targets):
        split = self.man_preprocessor.man_preprocess_single(split)
        topic_inds = targets['topic_inds']
        persp_inds = targets['persp_inds']
        split['topic_inds'] = topic_inds
        split['persp_inds'] = persp_inds
        return split

    def execute_pd(self, model):
        sentence_topic_dicts = []
        self.model = model
        if self.eval_exec:
            for index, row in self.test.iterrows():
                st_dict = {'sentence': row[0], 'topic': row[1]}
                sentence_topic_dicts.append(st_dict)
            output = self.model.run(sentence_topic_dicts)
            output = np.array([0 if x < 0 else 1 for x in output])
            stats = self.get_stats(output, self.test_labels)
            accuracy = self.get_acc(output, self.test_labels)
            self.data_saver.save_eval(stats, self.model_name)

    def get_stats(self, predictions, labels):
        stats = {'model':self.model_name}
        accuracy = self.get_acc(predictions, labels)
        stats.update(accuracy)
        prediction_classes = self.get_prediction_classes(predictions, labels)
        stats.update(prediction_classes)
        classif_metrics = self.get_classif_metrics(prediction_classes)
        stats.update(classif_metrics)
        return stats

    def get_acc(self, output, labels):
        correct = 0
        for i in range(len(labels)):
            if output[i] == labels[i]:
                correct += 1
        accuracy = round(correct / len(labels),2)
        return {'accuracy':accuracy}

    def get_prediction_classes(self, predictions, labels):
        TP = tf.math.count_nonzero(predictions * labels).numpy()
        TN = tf.math.count_nonzero((predictions-1) * (labels-1)).numpy()
        FP = tf.math.count_nonzero(predictions * (labels -1)).numpy()
        FN = tf.math.count_nonzero((predictions -1) * labels).numpy()
        return {'TP':TP, 'TN':TN, 'FP':FP, 'FN':FN}

    def get_classif_metrics(self, prediction_classes):
        TP = prediction_classes['TP']
        FP = prediction_classes['FP']
        FN = prediction_classes['FN']
        prec = self.get_precision(TP, FP)
        recall = self.get_recall(TP, FN)
        f1 = self.get_f1(prec, recall)
        return {'prec.':prec, 'recall':recall, 'f1':f1}

    def get_precision(self, TP, FP):
        precision = np.round(TP / (TP + FP),2)
        return precision
        
    def get_recall(self, TP, FN):
        recall = np.round(TP / (TP + FN),2)
        return recall

    def get_f1(self, prec, recall):
        f1 = np.round(2 * ((prec * recall)/(prec+recall)),2)
        return f1
