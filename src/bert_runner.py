import os
import tensorflow as tf
import numpy as np
import yaml
from models.bert_model import Bert_model
from utils.target_extractor import Target_extractor 

real_path = os.path.realpath(__file__)
dir_path = os.path.dirname(real_path)
config_file = os.path.join(dir_path, 'config.yaml')
with open(config_file) as file:
    config = yaml.safe_load(file)

topic = "Drones Should Be Used to Take Out Enemy Combatants"
claim = "Drone strikes are legal under international law."
target_extractor = Target_extractor()
phrase_candidates = target_extractor.get_phrase_candidates(claim, topic)
phrase_indices = target_extractor.identify_target_phrases(claim, topic)

model = Bert_model(config)
test_preprocess_model = model.make_bert_preprocess_model(['my_input1', 'my_input2'])
test_text = [np.array([topic]),
             np.array([claim])]

text_preprocessed = model.preprocessor(test_text)
breakpoint()
test_classifier_model = model.build_model()
bert_raw_result = model.classifier([text_preprocessed, phrase_indices, phrase_candidates])
print(bert_raw_result)
