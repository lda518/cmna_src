import tensorflow as tf
import official.nlp.bert.tokenization
from official.nlp import bert
import numpy as np
import os

class Preprocessor:
    def __init__(self, config):
        self.parse_config(config)
        self.tokenizer = bert.tokenization.FullTokenizer(
                vocab_file=os.path.join(self.gs_preprocessor, "vocab.txt"),
             do_lower_case=True)

    def parse_config(self, config):
        directories = config['directories']
        self.gs_preprocessor = directories['gs_folder_bert']

    def encode_sentence(self, s, tokenizer):
       tokens = list(self.tokenizer.tokenize(s))
       tokens.append('[SEP]')
       return tokenizer.convert_tokens_to_ids(tokens)

    def bert_encode(self, inputs):
        self.extract_sentences(inputs)

        num_examples = len(self.sent1_array)

        sentence1 = tf.ragged.constant([
            self.encode_sentence(s, self.tokenizer)
            for s in self.sent1_array])
        sentence2 = tf.ragged.constant([
            self.encode_sentence(s, self.tokenizer)
             for s in self.sent2_array])

        cls = [self.tokenizer.convert_tokens_to_ids(['[CLS]'])]*sentence1.shape[0]
        input_word_ids = tf.concat([cls, sentence1, sentence2], axis=-1)

        input_mask = tf.ones_like(input_word_ids).to_tensor()

        type_cls = tf.zeros_like(cls)
        type_s1 = tf.zeros_like(sentence1)
        type_s2 = tf.ones_like(sentence2)
        input_type_ids = tf.concat(
            [type_cls, type_s1, type_s2], axis=-1).to_tensor()

        inputs = {
            'input_word_ids': input_word_ids.to_tensor(),
            'input_mask': input_mask,
            'input_type_ids': input_type_ids}

        return inputs

    def extract_sentences(self, inputs):
        self.sent1_array = np.array(inputs[inputs.columns[0]])
        self.sent2_array = np.array(inputs[inputs.columns[1]])

    def preprocess_splits(self, train, validation, test):
        tokenizer = bert.tokenization.FullTokenizer(
            vocab_file=os.path.join(self.gs_folder_bert, "vocab.txt"),
             do_lower_case=True)
        train = self.bert_encode(train, tokenizer)
        validation = self.bert_encode(validation, tokenizer)
        test = self.bert_encode(test, tokenizer)
        return train, validation, test

    def man_preprocess(self, train, validation, test):
        train = self.bert_encode(train, self.tokenizer)
        validation = self.bert_encode(validation, self.tokenizer)
        test = self.bert_encode(test, self.tokenizer)
        return train, validation, test

    def man_preprocess_single(self, inputs):
        inputs = self.bert_encode(inputs)
        return inputs
