from typing_extensions import final
import numpy as np
import os
from tensorflow.python.eager.context import run_eager_op_as_function_enabled
from textblob import TextBlob
import tensorflow as tf
from official.nlp import bert
import tensorflow_hub as hub
import tensorflow as tf
import tensorflow_text as text
from numpy.linalg import norm
import official.nlp.optimization
from official import nlp
import official.nlp.bert.tokenization
import datetime
from utils.preprocessor import Preprocessor

class Bert_model:
    def __init__(self, config):
        self.parse_config(config)
        self.num_classes = 2
        tf.get_logger().setLevel('ERROR')
        self.man_preprocessor = Preprocessor(config)
        self.man_tokenizer = bert.tokenization.FullTokenizer(
                vocab_file=os.path.join(self.gs_preprocessor, "vocab.txt"),
             do_lower_case=True)

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
        self.checkpoint_path = os.path.join(self.root,directories['checkpoint_path'], 
                                            self.model, self.dataset)

    def build_model(self):
        breakpoint()
        if self.model == 'bert':
            # Standard FC classifier
            self.classifier = self.build_base(self.num_classes, self.encoder_handle)
        if self.model[:7] == 'bert_syn':
            # TD classifier with noun extraction
            pool_type = 'mul' if self.model[8:] == 'mul' else 'con'
            self.classifier =  self.build_et(self.num_classes, self.encoder_handle, pool_type)
        if self.model[:7] == 'bert_cosyn':
            pool_type = 'mul' if self.model[8:] == 'mul' else 'con'
            self.classifier =  self.build_tc(self.num_classes, self.encoder_handle, pool_type)
        if self.model[:6] == 'bert_cos':
            pool_type = 'mul' if self.model[8:] == 'mul' else 'con'
            self.classifier =  self.build_c(self.num_classes, self.encoder_handle, pool_type)

    def build_tc(self, num_classes, encoder_handle, pool_type):
        class Classifier(tf.keras.Model):
            def __init__(self, num_classes, pool_type):
                super(Classifier, self).__init__(name='prediction')
                self.pool_type = pool_type
                self.encoder = hub.KerasLayer(encoder_handle, trainable=True)
                self.dropout = tf.keras.layers.Dropout(0.1)
                self.dense = tf.keras.layers.Dense(num_classes)
                self.softmax = tf.keras.layers.Softmax()

            def call(self, inputs):
                topic_features = inputs.pop('topic_features')
                claim_features = inputs.pop('claim_features')

                scores = inputs.pop('scores')
                text = inputs
                ids = text['input_type_ids']
                encoder_outputs = self.encoder(text)
                pooled_output = encoder_outputs['pooled_output']
                sequence_output = encoder_outputs['sequence_output']
                # Pool and concattenate target vectors with final output
                mapped_topic_targets, mapped_claim_targets = self.select_targets(topic_features, 
                                                claim_features, scores, sequence_output, ids)
                final_pooled_output = []
                if self.pool_type == 'mul':
                    for i in range(len(pooled_output)):
                        multed_res = np.multiply(mapped_topic_targets[i], mapped_claim_targets[i], 
                                                pooled_output[i].numpy())
                        final_pooled_output.append(multed_res)
                elif self.pool_type == 'con':
                    for i in range(len(pooled_output)):
                        concatted_res = tf.concat([mapped_topic_targets[i], mapped_claim_targets[i], 
                                                pooled_output[i].numpy()], 0)
                        final_pooled_output.append(concatted_res)

                final_output = np.array(final_pooled_output)
                
                x = self.dropout(final_output)
                x = self.dense(x)
                x = self.softmax(x)
                return x

            def select_targets(self, topic_features, claim_features, scores, seq_output, ids):
                final_scores = []
                for i in range(len(seq_output)):
                    sentence_scores = []
                    sep_index = np.where(ids[i]==1)[0][0]
                    for j in range(len(topic_features[0])):
                        topic_inds = topic_features[i][j].numpy()
                        clean_topic_inds = topic_inds[topic_inds != -1]
                        claim_inds = claim_features[i][j].numpy()
                        clean_claim_inds = claim_inds[claim_inds != -1]
                        score = scores[i][j].numpy()
                        clean_scores = score[score != -1]
                        cosine_scores = []
                        for topic_ind in clean_topic_inds:
                            for claim_ind in clean_claim_inds:
                                topic_vec = seq_output[i][topic_ind + 1]
                                claim_vec = seq_output[i][claim_ind + sep_index]
                                cosine_sim = self.get_cosine_similarity(topic_vec, claim_vec)
                                cosine_scores.append(cosine_sim)
                        norm_scores = [score / sum(cosine_scores) for score in cosine_scores]
                        if len(norm_scores) != 0:
                            sentence_scores.append(score + max(norm_scores))
                    final_scores.append(sentence_scores)
                max_indices = []
                for sent_score in final_scores:
                    if len(sent_score) == 0:
                        max_indices.append(-1)
                    else:
                        max_indices.append(sent_score.index(max(sent_score)))
                final_topic_inds = []
                final_claim_inds = []
                final_topic_vectors = []
                final_claim_vectors = []
                for i in range(len(max_indices)):
                    if max_indices[i] == -1:
                        final_topic_inds.append(np.array([-1]))
                    else:
                        batch_topic_ind = topic_features[i][max_indices[i]].numpy()
                        final_topic_inds.append(batch_topic_ind[batch_topic_ind != -1])
                for i in range(len(max_indices)):
                    if max_indices[i] == -1:
                        final_claim_inds.append(np.array([-1]))
                    else:
                        batch_claim_ind = claim_features[i][max_indices[i]].numpy()
                        final_claim_inds.append(batch_claim_ind[batch_claim_ind != -1])
                for i in range(len(final_topic_inds)):
                    to_pool = []
                    for ind in final_topic_inds[i]:
                        if ind == -1:
                            to_pool.append(seq_output[i][0])
                        else:
                            to_pool.append(seq_output[i][ind+1])
                    pooled_topics = self.pooling(to_pool)
                    final_topic_vectors.append(pooled_topics)
                for i in range(len(final_claim_inds)):
                    sep_index = np.where(ids[i]==1)[0][0]
                    to_pool = []
                    for ind in final_claim_inds[i]:
                        if ind == -1:
                            to_pool.append(seq_output[i][0])
                        else:
                            to_pool.append(seq_output[i][ind+sep_index])
                    pooled_claims = self.pooling(to_pool)
                    final_claim_vectors.append(pooled_claims)
                return final_topic_vectors, final_topic_vectors

            def pooling(self, tensors):
                return np.max(tensors, axis=0)

            def get_cosine_similarity(self, topic, claim):
                cosine = np.dot(topic, claim)/(norm(topic)*norm(claim))
                return cosine

        model = Classifier(num_classes, pool_type)
        return model

    def build_c(self, num_classes, encoder_handle, pool_type):
        class Classifier(tf.keras.Model):
            def __init__(self, num_classes, pool_type):
                super(Classifier, self).__init__(name='prediction')
                self.pool_type = pool_type
                self.encoder = hub.KerasLayer(encoder_handle, trainable=True)
                self.dropout = tf.keras.layers.Dropout(0.1)
                self.dense = tf.keras.layers.Dense(num_classes)
                self.softmax = tf.keras.layers.Softmax()

            def call(self, inputs):
                topic_features = inputs.pop('topic_features')
                claim_features = inputs.pop('claim_features')

                scores = inputs.pop('scores')
                text = inputs
                ids = text['input_type_ids']
                encoder_outputs = self.encoder(text)
                pooled_output = encoder_outputs['pooled_output']
                sequence_output = encoder_outputs['sequence_output']
                # Pool and concattenate target vectors with final output
                mapped_topic_targets, mapped_claim_targets = self.select_targets(topic_features, 
                                                claim_features, scores, sequence_output, ids)
                final_pooled_output = []
                if self.pool_type == 'mul':
                    for i in range(len(pooled_output)):
                        multed_res = np.multiply(mapped_topic_targets[i], mapped_claim_targets[i], 
                                                pooled_output[i].numpy())
                        final_pooled_output.append(multed_res)
                elif self.pool_type == 'con':
                    for i in range(len(pooled_output)):
                        concatted_res = tf.concat([mapped_topic_targets[i], mapped_claim_targets[i], 
                                                pooled_output[i].numpy()], 0)
                        final_pooled_output.append(concatted_res)

                final_output = np.array(final_pooled_output)
                
                x = self.dropout(final_output)
                x = self.dense(x)
                x = self.softmax(x)
                return x

            def select_targets(self, topic_features, claim_features, scores, seq_output, ids):
                final_scores = []
                for i in range(len(seq_output)):
                    sentence_scores = []
                    sep_index = np.where(ids[i]==1)[0][0]
                    for j in range(len(topic_features[0])):
                        topic_inds = topic_features[i][j].numpy()
                        clean_topic_inds = topic_inds[topic_inds != -1]
                        claim_inds = claim_features[i][j].numpy()
                        clean_claim_inds = claim_inds[claim_inds != -1]
                        score = scores[i][j].numpy()
                        clean_scores = score[score != -1]
                        cosine_scores = []
                        for topic_ind in clean_topic_inds:
                            for claim_ind in clean_claim_inds:
                                topic_vec = seq_output[i][topic_ind + 1]
                                claim_vec = seq_output[i][claim_ind + sep_index]
                                cosine_sim = self.get_cosine_similarity(topic_vec, claim_vec)
                                cosine_scores.append(cosine_sim)
                        norm_scores = [score / sum(cosine_scores) for score in cosine_scores]
                        if len(norm_scores) != 0:
                            sentence_scores.append(max(norm_scores))
                    final_scores.append(sentence_scores)
                max_indices = []
                for sent_score in final_scores:
                    if len(sent_score) == 0:
                        max_indices.append(-1)
                    else:
                        max_indices.append(sent_score.index(max(sent_score)))
                final_topic_inds = []
                final_claim_inds = []
                final_topic_vectors = []
                final_claim_vectors = []
                for i in range(len(max_indices)):
                    if max_indices[i] == -1:
                        final_topic_inds.append(np.array([-1]))
                    else:
                        batch_topic_ind = topic_features[i][max_indices[i]].numpy()
                        final_topic_inds.append(batch_topic_ind[batch_topic_ind != -1])
                for i in range(len(max_indices)):
                    if max_indices[i] == -1:
                        final_claim_inds.append(np.array([-1]))
                    else:
                        batch_claim_ind = claim_features[i][max_indices[i]].numpy()
                        final_claim_inds.append(batch_claim_ind[batch_claim_ind != -1])
                for i in range(len(final_topic_inds)):
                    to_pool = []
                    for ind in final_topic_inds[i]:
                        if ind == -1:
                            to_pool.append(seq_output[i][0])
                        else:
                            to_pool.append(seq_output[i][ind+1])
                    pooled_topics = self.pooling(to_pool)
                    final_topic_vectors.append(pooled_topics)
                for i in range(len(final_claim_inds)):
                    sep_index = np.where(ids[i]==1)[0][0]
                    to_pool = []
                    for ind in final_claim_inds[i]:
                        if ind == -1:
                            to_pool.append(seq_output[i][0])
                        else:
                            to_pool.append(seq_output[i][ind+sep_index])
                    pooled_claims = self.pooling(to_pool)
                    final_claim_vectors.append(pooled_claims)
                return final_topic_vectors, final_topic_vectors

            def pooling(self, tensors):
                return np.max(tensors, axis=0)

            def get_cosine_similarity(self, topic, claim):
                cosine = np.dot(topic, claim)/(norm(topic)*norm(claim))
                return cosine

        model = Classifier(num_classes, pool_type)
        return model

    def build_base(self, num_classes, encoder_handle):
        class Classifier(tf.keras.Model):
            def __init__(self, num_classes):
                super(Classifier, self).__init__(name='prediction')
                self.encoder = hub.KerasLayer(encoder_handle, trainable=True)
                self.dropout = tf.keras.layers.Dropout(0.1)
                self.dense = tf.keras.layers.Dense(num_classes)
                self.softmax = tf.keras.layers.Softmax()

            def call(self, preprocessed_text):
                encoder_outputs = self.encoder(preprocessed_text)
                pooled_output = encoder_outputs['pooled_output']
                sequence_output = encoder_outputs['sequence_output']
                x = self.dropout(pooled_output)
                x = self.dense(x)
                x = self.softmax(x)
                return x

            #def get_cosine_similarity(self,topic, claim):
            #    cosine = np.dot(topic, claim)/(norm(topic)*norm(claim))
        model = Classifier(num_classes)
        return model

    def build_et(self, num_classes, encoder_handle, pool_type):
        class Classifier(tf.keras.Model):
            def __init__(self, num_classes, pool_type):
                super(Classifier, self).__init__(name='prediction')
                self.pool_type = pool_type
                self.encoder = hub.KerasLayer(encoder_handle, trainable=True)
                self.dropout = tf.keras.layers.Dropout(0.1)
                self.dense = tf.keras.layers.Dense(num_classes)
                self.softmax = tf.keras.layers.Softmax()

            def call(self, inputs):
                topic_targets = inputs.pop('topic_targets')
                claim_targets = inputs.pop('claim_targets')
                text = inputs
                ids = text['input_type_ids']
                encoder_outputs = self.encoder(text)
                pooled_output = encoder_outputs['pooled_output']
                sequence_output = encoder_outputs['sequence_output']
                # Pool and concattenate target vectors with final output
                mapped_topic_targets = []
                mapped_claim_targets = []
                for i in range(len(topic_targets.numpy())):
                    embedded_topic = [sequence_output[i][x]+1 for x in 
                                        topic_targets.numpy()[i] if x != -1]
                    if len(embedded_topic) == 0:
                        pooled_topic = np.ones(len(pooled_output[0]))
                    else:
                        pooled_topic = self.pooling(embedded_topic)
                    mapped_topic_targets.append(pooled_topic)
                for i in range(len(claim_targets.numpy())):
                    sep_index = np.where(ids[i]==1)[0][0]
                    embedded_claim = [sequence_output[i][x]+sep_index for x in 
                                        claim_targets.numpy()[i] if x != -1]
                    if len(embedded_claim) == 0:
                        pooled_claim = np.ones(len(pooled_output[0]))
                    else:
                        pooled_claim = self.pooling(embedded_claim)
                    mapped_claim_targets.append(pooled_claim)
                final_pooled_output = []
                if self.pool_type == 'mul':
                    for i in range(len(pooled_output)):
                        multed_res = np.multiply(mapped_topic_targets[i], mapped_claim_targets[i], 
                                                pooled_output[i].numpy())
                        final_pooled_output.append(multed_res)
                elif self.pool_type == 'con':
                    for i in range(len(pooled_output)):
                        concatted_res = tf.concat([mapped_topic_targets[i], mapped_claim_targets[i], 
                                                pooled_output[i].numpy()], 0)
                        final_pooled_output.append(concatted_res)

                final_output = np.array(final_pooled_output)
                
                x = self.dropout(final_output)
                x = self.dense(x)
                x = self.softmax(x)
                return x

            def pooling(self, tensors):
                return np.max(tensors, axis=0)


        model = Classifier(num_classes, pool_type)
        return model

    def make_bert_preprocess_model(self, sentence_features, seq_length=128):
        input_segments = [
                tf.keras.layers.Input(shape=(), dtype=tf.string, name=ft)
                for ft in sentence_features]

        # Tokenize the text to word pieces
        bert_preprocess = hub.load(self.preprocess_handle)
        self.tokenizer = hub.KerasLayer(bert_preprocess.tokenize, name='tokenizer')
        segments = [self.tokenizer(s) for s in input_segments]

        # Trim segments to fit seq_length
        truncated_segments = segments

        # Pack inputs
        packer = hub.KerasLayer(bert_preprocess.bert_pack_inputs,
                                arguments=dict(seq_length=seq_length),
                                name='packer')
        model_inputs = packer(truncated_segments)
        self.preprocessor = tf.keras.Model(input_segments, model_inputs)

    def get_loss_func(self):
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        return loss


    def get_metrics(self):
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)]
        return metrics

    def get_optimizer(self):
        train_data_size = len(self.train_labels)
        steps_per_epoch = int(train_data_size / self.batch_size)
        num_train_steps = steps_per_epoch * self.epochs
        warmup_steps = int(self.epochs * train_data_size * 0.1 / self.batch_size)
        optimizer = nlp.optimization.create_optimizer(
            2e-5, num_train_steps=num_train_steps, num_warmup_steps=warmup_steps)
        return optimizer
    
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

    def get_callbacks(self):
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        cp_path = os.path.join(self.checkpoint_path, 'cp-{epoch:04}.ckpt')
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=cp_path, 
                save_weights_only=True, verbose=1)
        return [tensorboard_callback, cp_callback]

    def load_latest_ch(self):
        latest=tf.train.latest_checkpoint(self.checkpoint_path)
        self.classifier.load_weights(latest)

    def preprocess_data(self, train, validation, test):
        train = self.man_preprocess.bert_encode(train, self.tokenizer)
        validation = self.man_preprocess.bert_encode(validation, self.tokenizer)
        test = self.man_preprocess.bert_encode(test, self.tokenizer)
        return train, validation, test

    def man_preprocess(self, train, validation, test):
        train = self.man_preprocessor.bert_encode(train, self.man_tokenizer)
        validation = self.man_preprocessor.bert_encode(validation, self.man_tokenizer)
        test = self.man_preprocessor.bert_encode(test, self.man_tokenizer)
        return train, validation, test

    def man_preprocess_single(self, inputs):
        inputs = self.man_preprocessor.bert_encode(inputs, self.man_tokenizer)
        return inputs
