import yaml
import os
import json
import tensorflow as tf
from official.nlp import bert
from models.pd.stance_detector import Stance_detector
import official.nlp.bert.configs
import official.nlp.bert.bert_models
from debater_python_api.api.debater_api import DebaterApi
from models.subclassing_bert import Bert_model

class Model_loader:
    def __init__(self, config):
        api_key = os.environ['DEBATER_API_KEY']
        debater_api = DebaterApi(api_key)
        self.config = config
        self.parse_config()
        self.stance_detector = Stance_detector(debater_api)

    def parse_config(self):
        self.directs = self.config['directories']
        model_conf= self.config['model_conf']
        check_path = self.directs['checkpoint_path']
        self.check_direct = 'utils/training_1/'
        self.model = model_conf['model']
        self.state = model_conf['state']

    def load_model(self):
        if 'bert' in self.model:
            return self.load_bert()
        if 'pd' in self.model:
            return self.load_pd()

    def load_bert(self):
        bert_model = Bert_model(self.config)
        bert_model.build_model()
        return bert_model

    def load_pd(self):
        return self.stance_detector.pro_con_pd()
