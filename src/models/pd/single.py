import os
from debater_python_api.api.clients.narrative_generation_client import Polarity
from debater_python_api.api.debater_api import DebaterApi
api_key = os.environ['DEBATER_API_KEY']
debater_api = DebaterApi(api_key)

# Define topic of interest and areas of interest
dc = 'Roman Empire'
aspects = ['War','Economy','Technology']

#####################
# Additional parameters
######################
articles_filename = dc + "_wikipedia.json"
dc_re = dc  # This can allow a more refined regular expression to find relevant sentences containing the topic concept.
            # For example 'Hybrid.*Cloud' would increase coverage.
topic = 'The ' + dc + ' was beneficial to the world'
max_sentence_length = 50
max_articles = 10000

###################
# Wikipedia miner
###################
from debater_python_api.api.sentence_level_index.client.sentence_query_base import SimpleQuery
from debater_python_api.api.sentence_level_index.client.sentence_query_request import SentenceQueryRequest
searcher = debater_api.get_index_searcher_client()
candidates = set()
query_size=10000
query = SimpleQuery(is_ordered=True, window_size=12)
query.add_normalized_element(['that'])
query.add_concept_element([dc])
query_request = SentenceQueryRequest(query=query.get_sentence_query(), size=query_size, sentenceLength=(7, 60))
results = searcher.run(query_request)
print("'that' followed by {} appears {} times. ".format(dc,len(results)))
candidates.update(results)


query = SimpleQuery(is_ordered=False, window_size=12)
query.add_concept_element([dc])
query.add_type_element(['Causality', 'Sentiment'])
query_request = SentenceQueryRequest(query=query.get_sentence_query(), size=query_size, sentenceLength=(7, 60))
results = searcher.run(query_request)
print("{} followed by sentiment or causality word appears {} times. ".format(dc,len(results)))
candidates.update(results)

texts = ([ { 'news_content': c, 'news_title' : c, 'id' : c } for c in candidates ])

######################
# Save results to file
######################
import json
with open(articles_filename, 'w') as outfile:
    json.dump(texts, outfile)

import json
with open(articles_filename, "r") as read_file:
    data = json.load(read_file)

#########################
# Read in the sentences, remove duplicates and create html snippets
##########################
from spacy.lang.en import English # updated
import re
nlp = English()

all_sentences = []
hashed_sentences = {}
total_sentences = 0
nlp.max_length = 1000000000
nlp.add_pipe(nlp.create_pipe('sentencizer')) # updated
for hit in data:
    doc_id = hit['id']
    doc_text = hit['news_content']
    doc_title = hit['news_title']
    doc = nlp(doc_text)
    prev = {'text' : '', 'html':''}
    for (i,sent) in enumerate(doc.sents):
        total_sentences += 1
        text = sent.string.strip()
        hash_of_text = hash(text)
        if hash_of_text in hashed_sentences:
            continue
        hashed_sentences[hash_of_text] = True    
        sentence_entry = { 'doc_id': doc_id , 'line' : i , 'text': text , 'prev': prev['text'] }
        sentence_entry['wordcount'] = len(re.findall(r'\w+', sentence_entry['text']))
        prev[ 'next' ] = sentence_entry['text']
        sentence_entry[ 'html'] = sentence_entry['prev'] + '<b><br>' + sentence_entry['text'] + '</b><br>' 
        prev[ 'html' ] += sentence_entry['text']
        prev = sentence_entry
        all_sentences.append(sentence_entry)
    sentence_entry['next'] = ''      
print('Number of removed duplicates: {}'.format(total_sentences - len(all_sentences)))
print('Total number of sentences remaining: {}'.format(len(all_sentences)))

#####################
# Filter sentences that contain the main concept using regex.
# Remove sentences which are too long
######################
import re
all_candidates = [sent for sent in all_sentences  if re.search(dc_re,sent['text'],re.IGNORECASE) and sent['wordcount'] < max_sentence_length]
print('Total number of sentences containing the topic:' + str(len(all_candidates)))

######################
# Print some random candidates
######################
#print(all_candidates[0:100])


#######################
# Get pro/con scores
#######################sentence_topic_dicts = [{'sentence' : sentence['text'], 'topic' : topic } for sentence in all_candidates]
sentence_topic_dicts = [{'sentence' : sentence['text'], 'topic' : topic } for sentence in all_candidates]
pro_con_scores = debater_api.get_pro_con_client().run(sentence_topic_dicts)
for sentence, pro_con_score in zip(all_candidates, pro_con_scores):
    sentence['procon'] = round(pro_con_score, 3)

#####################
# Score evidence of arguments
#####################
evidence_scores = debater_api.get_evidence_detection_client().run(sentence_topic_dicts)
for sentence, evidence_score in zip(all_candidates, evidence_scores):
    sentence['evidence'] = round(evidence_score, 3)

######################
# Print top statements with high evidence and stance
######################
#print_top_by_function("Most argumentative",all_candidates,
#                          function=lambda candidate: -abs(candidate['procon'])-candidate['evidence'],
#                                                       fields = ['html','procon','evidence'])

######################
# Generate Narrative 
######################
# customize opening statement
opening_statement_customization_dict = {
    'type' : 'openingStatement',
    'items':
        [
              {
                'key': 'Opening_statement',
                'value': ' The following speech is based on <NUM_SBC_ARGUMENTS_SUGGESTED> mined from Wikipedia supporting the notion that <MOTION>.',
                'itemType' : 'single_string'
              },
        ]
}

print('Generating speech:')

sentences = [ sentence['text'] for sentence in all_candidates if sentence['evidence'] > 0.2]
print('Number of sentences passed to speech generation: ' + str(len(sentences)))
narrative_generation_client = debater_api.get_narrative_generation_client()
sentence_topic_dicts = [{'sentence' : sentence, 'topic' : topic } for sentence in sentences]
pro_con_scores = debater_api.get_pro_con_client().run(sentence_topic_dicts)
speech = narrative_generation_client.run(topic=topic, dc=dc, sentences=sentences,
                                         pro_con_scores=pro_con_scores, polarity=Polarity.PRO,
                                         customizations=[opening_statement_customization_dict])
print('\n\nSpeech:' + speech.status)
if (speech.status == 'DONE'):
    print(speech)
else:
    print(speech.error_message)


###################
# Analyze arguments by aspect
##################
# Extract all the wikipedia concepts mentioned in a sentence using the termwikifier service
def add_sentence_mentions(sentences):
    term_wikifier_client = debater_api.get_term_wikifier_client()
    mentions_list = term_wikifier_client.run([sentence['text'] for sentence in sentences])
    for sentence, mentions in zip(sentences, mentions_list):
        sentence['mentions'] = set([mention['concept']['title'] for mention in mentions])
    all_mentions = set([mention for sentence in sentences for mention in sentence['mentions']])
    return set(all_mentions)

all_mentions = add_sentence_mentions(all_candidates)
print('Total number of unique concepts: {}'.format(len(all_mentions)))

# Find all closely related wikipedia concepts that are mentioned in the given sentences
def get_related_mentions(concept, threshold, mentions):
    term_relater_client = debater_api.get_term_relater_client()
    concept_mention_pairs = [[concept, mention] for mention in mentions]
    scores = term_relater_client.run(concept_mention_pairs)
    return [mention for mention, score in zip(all_mentions, scores) if score > threshold]

matched_mentions = {}
for aspect in aspects:
    matched_mentions[aspect] = get_related_mentions(aspect, 0.8, all_mentions)
    print(aspect,":",matched_mentions[aspect])

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

    #print_top_by_function(aspect,matched_sentences,
    #                      function=lambda candidate: -candidate['score'],
    #                                                   fields = ['html','procon', 'evidence',aspect + '_concepts'])
