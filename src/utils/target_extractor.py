from utils.text_processor import Text_processor
from utils.cosine_comp import Cosine_comp
from utils.data_saver import Data_saver
import pandas as pd
import numpy as np

class Target_extractor:
    def __init__(self, config):
        self.text_processor = Text_processor()
        self.cosine_comp = Cosine_comp(config)
        self.data_saver = Data_saver(config)
        self.parse_config(config)
        self.types = ['combined_score', 'norm_syn_score', 'norm_cosine_score']
        if 'cosyn' in self.model:
            self.type = self.types[0]
        elif 'syn' in self.model:
            self.type = self.types[1]
        elif 'cos' in self.model:
            self.type = self.types[2]

    def parse_config(self, config):
        model_conf = config['model_conf']
        self.model = model_conf['model']

    def extract_all_target_info(self, split):
        self.to_csv_file = pd.DataFrame()
        target_dict = self.extract_target_dict_from_dataset(split)
        for _type in self.types:
            target_info = self.get_model_spec_targets(target_dict, _type)
        return target_info

    def extract_target_dict_from_dataset(self, split):
        evaluated_rows = []
        #split = split[:10]
        split = split.reset_index()  # make sure indexes pair with number of rows
        for index, row in split.iterrows():
            pair = pd.DataFrame({'topic':[row['topic']], 'perspective':[row['perspective']]})
            combined_scores = self.extract_combined_scores_dict(pair)
            evaluated_rows.append(combined_scores)
            print(index)
            self.to_csv_file = self.to_csv_file.append([pair])
        self.to_csv_file = self.to_csv_file.reset_index()
        return evaluated_rows

    def extract_combined_scores_dict(self, sent_pair):
        topic_phrases, persp_phrases = self.extract_phrases(sent_pair)
        if len(topic_phrases)==0 or len(persp_phrases)==0:
            return [-1]
        synset_scores = self.get_synset_scores(topic_phrases, persp_phrases)
        embeddings = self.get_sentence_embeddings(sent_pair)
        cosine_scores = self.get_cosine_scores(synset_scores, embeddings)
        combined_scores = self.get_combinded_scores(cosine_scores)
        return combined_scores

    def extract_phrases(self, sent_pair):
        topic_cands, persp_cands = self.text_processor.get_noun_candidates(sent_pair)
        topic_phrases = self.text_processor.get_phrase_candidates(topic_cands)
        persp_phrases = self.text_processor.get_phrase_candidates(persp_cands)
        return topic_phrases, persp_phrases

    def get_synset_scores(self, topic_phrases, persp_phrases):
        return self.text_processor.get_synsets_phrases(topic_phrases, persp_phrases)

    def get_sentence_embeddings(self, sent_pair):
        self.cosine_comp.preprocess_sentences(sent_pair)
        embeddings = self.cosine_comp.get_embeddings()
        return embeddings 

    def get_cosine_scores(self, synset_dict, embeddings):
        combined_dict = self.cosine_comp.get_cosine_phrases(synset_dict, embeddings)
        return combined_dict

    def get_combinded_scores(self, combined_dict):
        for pair in combined_dict:
            pair['combined_score'] = pair['norm_syn_score'] + pair['norm_cosine_score']
        return combined_dict

    def get_model_spec_targets(self, target_dict, model_type):
        phrase_dataframe = pd.DataFrame()
        topic_target_inds = []
        persp_target_inds = []
        for sent_pair in target_dict:
            if sent_pair == [-1]:
                topic_inds = [-1]
                persp_inds = [-1]
                topic_phrase_string = '-1'
                persp_phrase_string = '-1'
            else:
                best_match = max(sent_pair, key=lambda x:x[model_type])
                best_topic_phrase = list(best_match['topic_phrase'].values())
                topic_phrase_string = " ".join(map(str,best_topic_phrase))
                best_persp_phrase = list(best_match['persp_phrase'].values())
                persp_phrase_string = " ".join(map(str,best_persp_phrase))
                topic_inds = list(best_match['topic_phrase'].keys())
                persp_inds = list(best_match['persp_phrase'].keys())
            phrase_frame = pd.DataFrame({'topic_tar_{}'.format(model_type): [topic_phrase_string],
                                        'persp_tar_{}'.format(model_type): [persp_phrase_string]})
            phrase_dataframe = phrase_dataframe.append(phrase_frame)
            topic_target_inds.append(topic_inds)
            persp_target_inds.append(persp_inds)
        padded_topic_targets = self.preprocess_targets(topic_target_inds)
        padded_persp_targets = self.preprocess_targets(persp_target_inds)
        inds_dataframe = pd.DataFrame({'topic_inds_{}'.format(model_type):padded_topic_targets,
                        'persp_inds_{}'.format(model_type):padded_persp_targets})
        phrase_dataframe = phrase_dataframe.reset_index()
        self.to_csv_file = pd.concat([self.to_csv_file, phrase_dataframe], axis=1)
        self.to_csv_file = pd.concat([self.to_csv_file, inds_dataframe], axis=1)
        return self.to_csv_file

    def preprocess_targets(self, target_split):
        padded_length = max(map(len, target_split))
        for target in target_split:
            while len(target)<padded_length:
                target.append(-1)
        return target_split

    def get_preprocessed_targets(self,filepath):
        target_data = self.data_saver.load_pandas_pickle(filepath)
        topic_inds = target_data['topic_inds_{}'.format(self.type)]
        persp_inds = target_data['persp_inds_{}'.format(self.type)]
        topic_inds = np.array([np.array(x) for x in topic_inds])
        persp_inds = np.array([np.array(x) for x in persp_inds])
        target_dict = {'topic_inds':topic_inds, 'persp_inds':persp_inds}
        return target_dict
