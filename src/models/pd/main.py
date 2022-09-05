import os
from debater_python_api.api.clients.narrative_generation_client import Polarity
from debater_python_api.api.debater_api import DebaterApi
from miner import Miner
from stance_detector import Stance_detector
from evidence_detector import Evidence_detector
from speech_generator import Speech_generator
from analyzer import Analyzer
from data_manager import Data_manager
import json
api_key = os.environ['DEBATER_API_KEY']
debater_api = DebaterApi(api_key)

# Define topic of interest and areas of interest
dc = 'Roman Empire'
aspects = ['War','Economy','Technology']

# Additional parameters
articles_filename = dc + "_wikipedia.json"
scores_filename = 'pro_con_scores.json'
evidence_filename = 'evidence_scores.json'
mentions_filename = 'all_mentions.json'
matched_filename = 'matched_mentions.json'
dc_re = dc  # This can allow a more refined regular expression to find relevant sentences containing the topic concept.
            # For example 'Hybrid.*Cloud' would increase coverage.
topic = 'The ' + dc + ' was beneficial to the world'
max_articles = 10000
data_manager = Data_manager()

#################
# Wikipedia miner
miner = Miner(debater_api, dc, articles_filename)
miner.mine()
all_candidates = miner.clean_up()
sentence_topic_dicts = [{'sentence' : sentence['text'], 'topic' : topic} for sentence in all_candidates]


# Get pro/con scores
stance_detector = Stance_detector(debater_api)
pro_con_detector = stance_detector.pro_con_pd()
pro_con_detector.get_scores(sentence_topic_dicts)
pro_con_detector.save_scores(scores_filename)
pro_con_scores = data_manager.load(scores_filename)
for sentence, pro_con_score in zip(all_candidates, pro_con_scores):
    sentence['procon'] = round(pro_con_score, 3)

# Score evidence of arguments
evidence_detector = Evidence_detector(sentence_topic_dicts, evidence_filename, debater_api)
evidence_detector.evidence_scores()
evidence_scores = data_manager.load(evidence_filename)
for sentence, evidence_score in zip(all_candidates, evidence_scores):
    sentence['evidence'] = round(evidence_score, 3)


# Print top statements with high evidence and stance
#print_top_by_function("Most argumentative",all_candidates,
#                          function=lambda candidate: -abs(candidate['procon'])-candidate['evidence'],
#                                                       fields = ['html','procon','evidence'])


# Generate Narrative 
print('start speech gen')

speech_generator = Speech_generator(all_candidates, debater_api, topic, dc)
speech_generator.generate()
print('stop speech gen')
###################
# Analyze arguments by aspect
##################
# Extract all the wikipedia concepts mentioned in a sentence using the termwikifier service
analyzer = Analyzer(mentions_filename, matched_filename, all_candidates, aspects, debater_api)
#print(all_candidates[0]['mentions'])
analyzer.get_sentence_mentions()
analyzer.set_sentence_mentions()

print('Total number of unique concepts: {}'.format(len(analyzer.all_mentions)))

# Find all closely related wikipedia concepts that are mentioned in the given sentences
analyzer.get_matched_mentions()
matched_mentions = data_manager.load(matched_filename)
# Identify for each sentence all related concepts that appear in it for each aspect

for sentence in all_candidates:
    for aspect in aspects:
        sentence[aspect + '_concepts'] = sentence['mentions'].intersection(matched_mentions[aspect])

#############
# Print out analyzed sentences based on concepts and ranked by evidence / stance
#############
for aspect in aspects:
    matched_sentences = [sentence for sentence in all_candidates if len(sentence[aspect + '_concepts']) >= 1]

    for sentence in all_candidates:
        sentence['score'] = len(sentence[aspect + '_concepts'])/3 + sentence['evidence'] + abs(sentence['procon'])

