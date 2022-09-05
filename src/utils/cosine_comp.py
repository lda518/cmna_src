import tensorflow_hub as hub
import numpy as np
from official.nlp import bert
from utils.preprocessor import Preprocessor
from numpy.linalg import norm

class Cosine_comp:
    def __init__(self, config):
        self.parse_config(config)
        self.preprocessor = Preprocessor(config)
        self.encoder = hub.KerasLayer(self.encoder_handle, trainable=True)

    def parse_config(self, config):
        directories = config['directories']
        self.encoder_handle = directories['hub_url_bert']

    def preprocess_sentences(self, sentences):
        self.sentences = self.preprocessor.bert_encode(sentences)
        input_ids = self.sentences['input_word_ids']
        self.sep_index = np.where(input_ids==102)[1][0]

    def get_embeddings(self):
        embeddings = self.encoder(self.sentences)['sequence_output']
        return embeddings

    def get_cosine_phrases(self, pairs, embeddings):
        scores_total = 0
        for pair in pairs:
            topic_phrase = pair['topic_phrase']
            persp_phrase = pair['persp_phrase']
            scores = []
            for topic_noun_ind in topic_phrase:
                topic_vec = embeddings[0][topic_noun_ind+1]
                for persp_noun_ind in persp_phrase:
                    persp_vec = embeddings[0][persp_noun_ind+self.sep_index]
                    score = self.get_cosine_vecs(topic_vec, persp_vec)
                    scores.append(score)
                    topic_word = topic_phrase[topic_noun_ind]
                    persp_word = persp_phrase[persp_noun_ind]
            max_score = round(max(scores),3)
            scores_total += max_score
            pair['cosine_score'] = max_score
        normed_pairs = self.normalize_scores(pairs, scores_total)
        return normed_pairs

    def get_cosine_vecs(self, topic_vec, persp_vec):
        cosine = np.dot(topic_vec, persp_vec)/(norm(topic_vec)*norm(persp_vec))
        return cosine

    def normalize_scores(self, pairs, total):
        for pair in pairs:
            pair['norm_cosine_score'] = round(pair['cosine_score']/total,3)
        return pairs
