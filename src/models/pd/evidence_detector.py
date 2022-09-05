import json
from data_manager import Data_manager

class Evidence_detector:
    def __init__(self, sentence_topic_dicts, evidence_filename, debater_api):
        self.sentence_topic_dicts = sentence_topic_dicts
        self.evidence_filename = evidence_filename
        self.debater_api = debater_api
        self.data_manager = Data_manager()

    def evidence_scores(self):
        evidence_scores = self.debater_api.get_evidence_detection_client().run(self.sentence_topic_dicts)
        self.data_manager.save(self.evidence_filename, evidence_scores)
