from debater_python_api.api.sentence_level_index.client.sentence_query_base import SimpleQuery
from debater_python_api.api.sentence_level_index.client.sentence_query_request import SentenceQueryRequest
from debater_python_api.api.debater_api import DebaterApi
from data_manager import Data_manager
import json
import os
from spacy.lang.en import English # updated
import re
import re

class Miner:
    def __init__(self, debater_api, dc, articles_filename):
        self.data_manager = Data_manager()
        self.debater_api = debater_api
        self.dc = dc
        self.articles_filename = articles_filename
        self.dc_re = dc
        self.max_sentence_length = 50

    def mine(self):
        searcher = self.debater_api.get_index_searcher_client()
        candidates = set()
        query_size=10000
        query = SimpleQuery(is_ordered=True, window_size=12)
        query.add_normalized_element(['that'])
        query.add_concept_element([self.dc])
        query_request = SentenceQueryRequest(query=query.get_sentence_query(), size=query_size, sentenceLength=(7, 60))
        results = searcher.run(query_request)
        print("'that' followed by {} appears {} times. ".format(self.dc,len(results)))
        candidates.update(results)

        query = SimpleQuery(is_ordered=False, window_size=12)
        query.add_concept_element([self.dc])
        query.add_type_element(['Causality', 'Sentiment'])
        query_request = SentenceQueryRequest(query=query.get_sentence_query(), size=query_size, sentenceLength=(7, 60))
        results = searcher.run(query_request)
        print("{} followed by sentiment or causality word appears {} times. ".format(self.dc,len(results)))
        candidates.update(results)

        texts = ([ { 'news_content': c, 'news_title' : c, 'id' : c } for c in candidates ])
        self.data_manager.save(self.articles_filename, texts)

    def clean_up(self):
        data = self.data_manager.load(self.articles_filename)
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

        # Filter sentences that contain the main concept using regex.
        # Remove sentences which are too long
        all_candidates = [sent for sent in all_sentences  if re.search(self.dc_re,sent['text'],re.IGNORECASE) and sent['wordcount'] < self.max_sentence_length]
        print('Total number of sentences containing the topic:' + str(len(all_candidates)))
        return all_candidates

if __name__=="__main__":
    api_key = os.environ['DEBATER_API_KEY']
    debater_api = DebaterApi(api_key)
