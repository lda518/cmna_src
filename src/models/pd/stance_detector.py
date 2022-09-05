from debater_python_api.api.sentence_level_index.client.sentence_query_base import SimpleQuery
from debater_python_api.api.sentence_level_index.client.sentence_query_request import SentenceQueryRequest
from debater_python_api.api.debater_api import DebaterApi
from models.pd.data_manager import Data_manager
import os
import json
class Stance_detector:
    def __init__(self, debater_api):
        self.debater_api = debater_api
        self.data_manager = Data_manager()

    def pro_con_pd(self):
        self.pro_con_detector = self.debater_api.get_pro_con_client()
        return self.pro_con_detector

    def get_scores(self, sentence_topic_dicts):
        return self.pro_con_detector.run(sentence_topic_dicts)

    def save_scores(self, scores_filename):
        return self.data_manager.save(self.scores_filename, self.pro_con_scores)

if __name__=="__main__":
    api_key = os.environ['DEBATER_API_KEY']
    debater_api = DebaterApi(api_key)

