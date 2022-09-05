import os
import yaml
import tensorflow as tf
from official import nlp
import official.nlp.optimization
import datetime
import os
from utils.preprocessor import Preprocessor
import official.nlp.bert.tokenization
from official.nlp import bert
from utils.data_saver import Data_saver
import json
import tensorflow as tf
from utils.preprocessor import Preprocessor
import official.nlp.bert.tokenization
import official.nlp.bert.bert_models
from official.nlp import bert
class Bert:
    def __init__(self, config):
        self.parse_config(config)
        self.preprocessor = Preprocessor(config)

    def parse_config(self, config):
        self.directs = config['directories']
        self.root = self.directs['root_direct']
        data_conf = config['data_conf']
        model_conf= config['model_conf']
        self.model_name = model_conf['model']
        self.dataset = data_conf['dataset']
        self.state = model_conf['state']
        self.rate = config['exec_conf']['train']['lr']
        self.checkpoint_path = os.path.join(self.root,self.directs['checkpoint_path'], 
                                            self.model_name, self.dataset)
        self.gs_folder_bert = self.directs['gs_folder_bert']

    # load the off-shelf model of BERT for further customization
    def load_shelf(self):
        pass


    def load_model(self):
        bert_cosonfig = self.get_pre_bert_cosonf()
        bert_coslassifier, bert_encoder = bert.bert_models.classifier_model(
                                        bert_cosonfig, num_labels=2)

        checkpoint = tf.train.Checkpoint(encoder=bert_encoder)
        checkpoint = tf.train.Checkpoint(encoder=bert_encoder)
        checkpoint.read(os.path.join(self.gs_folder_bert, 'bert_model.ckpt')).assert_consumed()

        self.encoder = bert_encoder
        self.classifier = bert_coslassifier

    def load_latest_ch(self):
        latest=tf.train.latest_checkpoint(self.checkpoint_path)
        self.classifier.load_weights(latest)

    def get_optimizer(self):
        train_data_size = len(self.train_labels)
        steps_per_epoch = int(train_data_size / self.batch_size)
        num_train_steps = steps_per_epoch * self.epochs
        warmup_steps = int(self.epochs * train_data_size * 0.1 / self.batch_size)
        optimizer = nlp.optimization.create_optimizer(
            float(self.rate), num_train_steps=num_train_steps, num_warmup_steps=warmup_steps)
        return optimizer

    def get_loss_func(self):
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        return loss

    def get_metrics(self):
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)]
        return metrics

    def get_callbacks(self):
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        cp_path = os.path.join(self.checkpoint_path, self.model_name, self.dataset, 'cp-{epoch:04}.ckpt')
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=cp_path, 
                save_weights_only=True, verbose=1)
        return [tensorboard_callback, cp_callback]

    def get_pre_bert_cosonf(self):
        bert_cosonfig = os.path.join(self.gs_folder_bert, "bert_cosonfig.json")
        config_dict = json.loads(tf.io.gfile.GFile(bert_cosonfig).read())
        bert_cosonfig = bert.configs.BertConfig.from_dict(config_dict)
        #checkpoint.read(os.path.join(gs_folder_bert, 'bert_model.ckpt')).assert_consumed()
        return bert_cosonfig

    def compile_model(self, train_labels, batch_size, epochs):
        self.train_labels = train_labels
        self.batch_size = batch_size
        self.epochs = epochs
        optimizer = self.get_optimizer()
        loss = self.get_loss_func()
        metrics = self.get_metrics()
        self.classifier.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics)

    def encode_data(self, train, validation, test):
        tokenizer = bert.tokenization.FullTokenizer(
            vocab_file=os.path.join(self.gs_folder_bert, "vocab.txt"),
             do_lower_case=True)
        train = self.preprocessor.bert_encode(train, tokenizer)
        validation = self.preprocessor.bert_encode(validation, tokenizer)
        test = self.preprocessor.bert_encode(test, tokenizer)
        return train, validation, test
