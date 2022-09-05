from posixpath import join
from typing_extensions import final
import numpy as np
import os
from tensorflow.python.eager.context import run_eager_op_as_function_enabled
from textblob import TextBlob
import tensorflow as tf
from official.nlp import bert
import tensorflow_hub as hub
import tensorflow_text as text
from numpy.linalg import norm
import official.nlp.optimization
from official import nlp
import official.nlp.bert.tokenization
import datetime
from utils.preprocessor import Preprocessor
from official import nlp

class Bert_model:
    def __init__(self, config):
        self.parse_config(config)
        self.num_classes = 2
        tf.get_logger().setLevel('ERROR')
        self.man_preprocessor = Preprocessor(config)
        self.log_dir = os.path.join(self.root, '..', "logs/fit/" + 
                                    datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    def parse_config(self, config):
        directories = config['directories']
        model_conf = config['model_conf']
        exec_conf = config['exec_conf']
        data_conf = config['data_conf']

        self.encoder_handle = directories['hub_url_bert']
        self.preprocess_handle = directories['hub_preprocess']
        self.root = directories['root_direct']
        self.gs_preprocessor = directories['gs_folder_bert']

        self.model = model_conf['model']
        self.rate = exec_conf['train']['lr']
        self.dataset = data_conf['dataset']
        self.checkpoint_path = os.path.join(self.root,'..',directories['checkpoint_path'], 
                                            self.model, self.dataset)

    def build_model(self):
        if self.model == 'bert':
            self.classifier = self.build_classifier(self.encoder_handle)
        else:
            pool_type = 'con' if 'con' in self.model else 'mul'
            self.classifier =  self.build_classifier(self.encoder_handle, pool_type, targets=True)

    def build_classifier(self, encoder_handle, pool_type=None, targets=False):
        class Target_pooling(tf.keras.layers.Layer):
            def __init__(self):
                super(Target_pooling, self).__init__()
                self.max_pooling = tf.keras.layers.GlobalMaxPooling1D()
                self.pool_type = pool_type

            def call(self, target_inds, encoder_out, sep_index=None):
                target_vecs = self.get_target_vecs(target_inds, encoder_out, sep_index)
                x = self.max_pooling(target_vecs)
                return x

            def get_target_vecs(self, target_inds, encoder_out, sep_index):
                vec_length = len(encoder_out[0][0])
                collected_vecs = []
                for i in range(len(encoder_out)):
                    if isinstance(sep_index, type(None)):
                        sep_index = [1 for x in range(len(target_inds))]
                    try:
                        target_vecs = [encoder_out[i][target_inds[i][j]+sep_index[i]] for j in 
                                        range(len(target_inds[i])) if target_inds[i][j] != -1]
                    except Exception as e:
                        print(e)
                        breakpoint()
                    if len(target_vecs)==0:
                        if self.pool_type == 'con':
                            target_vecs = [np.zeros(vec_length)]
                        elif self.pool_type == 'mul':
                            target_vecs = [np.full(vec_length,1)]
                    while len(target_vecs) < len(target_inds[0]):
                        target_vecs.append(np.full(vec_length,-1))
                    target_vecs = np.array(target_vecs)
                    collected_vecs.append(target_vecs)
                try:
                    collected_vecs = np.array(collected_vecs, dtype=np.float32)
                except Exception:
                    breakpoint()
                return collected_vecs

        class Joiner(tf.keras.layers.Layer):
            def __init__(self, pool_type=pool_type):
                super(Joiner, self).__init__()
                self.pool_type = pool_type

            def call(self, pooled_output, pooled_topic, pooled_persp):
                if self.pool_type == 'con':
                    joined = tf.concat([pooled_output, pooled_topic, pooled_persp],1)
                elif self.pool_type == 'mul':
                    join_first = tf.math.multiply(pooled_output, pooled_topic)
                    joined = tf.math.multiply(join_first, pooled_persp)
                return joined

        class Classifier(tf.keras.Model):
            def __init__(self, num_classes=2, targets=targets):
                super(Classifier, self).__init__()
                self.targets = targets
                self.encoder = hub.KerasLayer(encoder_handle, trainable=True)
                if self.targets:
                    self.target_pooling = Target_pooling()
                    self.joiner = Joiner(pool_type)
                self.dropout = tf.keras.layers.Dropout(0.1)
                self.dense = tf.keras.layers.Dense(num_classes)
                self.softmax = tf.keras.layers.Softmax()

            def call(self, inputs):
                if self.targets:
                    topic_inds = inputs.pop('topic_inds')
                    persp_inds = inputs.pop('persp_inds')
                encoder_outputs = self.encoder(inputs)
                sep_inds= self.get_sep_indices(inputs)
                pooled_output = encoder_outputs['pooled_output']
                sequence_output = encoder_outputs['sequence_output']
                final_output = pooled_output
                if self.targets:
                    pooled_topic = self.target_pooling(topic_inds, sequence_output)
                    pooled_persp = self.target_pooling(persp_inds, sequence_output, sep_index=sep_inds)
                    final_output = self.joiner(pooled_output, pooled_topic, pooled_persp)
                x = self.dropout(final_output)
                x = self.dense(x)
                x = self.softmax(x)
                return x

            def get_sep_indices(self, inputs):
                sep_inds = np.array([np.where(inp==102)[0][0] for inp in inputs['input_word_ids']])
                return sep_inds

        model = Classifier()
        return model

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
            metrics=metrics,
            run_eagerly=True)

    def get_optimizer(self):
        train_data_size = len(self.train_labels)
        steps_per_epoch = int(train_data_size / self.batch_size)
        num_train_steps = steps_per_epoch * self.epochs
        warmup_steps = int(self.epochs * train_data_size * 0.1 / self.batch_size)
        optimizer = nlp.optimization.create_optimizer(
            2e-5, num_train_steps=num_train_steps, num_warmup_steps=warmup_steps)
        #optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
        return optimizer

    def get_loss_func(self):
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        return loss

    def get_metrics(self):
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)]
        return metrics

    def get_callbacks(self):
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1)
        cp_path = os.path.join(self.checkpoint_path, 'cp-{epoch:04}.ckpt')
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=cp_path, 
                save_weights_only=True, verbose=1)
        return [tensorboard_callback, cp_callback]

    def load_latest_ch(self):
        latest=tf.train.latest_checkpoint(self.checkpoint_path)
        self.classifier.load_weights(latest)
