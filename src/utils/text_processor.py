import nltk
from textblob import TextBlob
from textblob import Word

class Text_processor:
    def __init__(self):
        self.pad_size = 5
        nltk.download('wordnet')
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
    
    def get_noun_candidates(self,pair):
        topic, perspective = self.extract_from_pair(pair)
        topic_blob = TextBlob(topic).lower()
        persp_blob = TextBlob(perspective).lower()
        topic_cands = self.get_noun_dicts(topic_blob)
        persp_cands = self.get_noun_dicts(persp_blob)
        return topic_cands, persp_cands

    def get_phrase_candidates(self,noun_cands):
        return self.get_phrase_dicts(noun_cands)

    def extract_from_pair(self, pair):
        topic = pair['topic'].values[0]
        perspective = pair['perspective'].values[0]
        return topic, perspective

    def get_noun_dicts(self, blob):
        words = blob.words
        noun_dicts = {}
        nouns = [w for (w, pos) in blob.lower().pos_tags if pos[0] == 'N']
        for noun in nouns:
            if noun in words:
                noun_dicts[words.index(noun)] = noun 
        return noun_dicts

    def get_phrase_dicts(self, noun_dict):
        noun_phrases = []
        prev_index = -2
        phrase = {}
        for entry in noun_dict.items():
            if entry[0] == prev_index+1 or entry[0] == list(noun_dict.keys())[0]:
                phrase[entry[0]] = entry[1]
            elif entry[0] != prev_index+1 :
                noun_phrases.append(phrase)
                phrase = {entry[0]:entry[1]}
            if entry[0] == list(noun_dict.keys())[-1]:
                noun_phrases.append(phrase)
            prev_index = entry[0]
        return noun_phrases

    def get_synsets_phrases(self, topic_phrases, persp_phrases):
        running_total = 0
        phrase_scores = []
        for topic_phrase in topic_phrases:
            for persp_phrase in persp_phrases:
                score = []
                for topic_noun_ind in topic_phrase:
                    topic_noun = Word(topic_phrase[topic_noun_ind].lemmatize())
                    for persp_noun_ind in persp_phrase:
                        persp_noun = Word(persp_phrase[persp_noun_ind].lemmatize())
                        score.append(self.get_synsets_nouns(topic_noun, persp_noun))
                best_match = round(max(score),3)
                phrase_scores.append({'topic_phrase':topic_phrase, 'persp_phrase':persp_phrase,
                        'synset_score':best_match})
                running_total += best_match
        # When one of the phrase groups does not contain a synset, for eg with "BBC", 
        # return a score of 0 so that the cosine evaluator can do the rest
        if running_total == 0:
            running_total = 1
        if len(phrase_scores) == 0:
            breakpoint()
        normed_scores = self.normalize_scores(phrase_scores, running_total)
        return normed_scores

    def normalize_scores(self, pairs, total):
        for pair in pairs:
            pair['norm_syn_score'] = round(pair['synset_score']/total,3)
        return pairs

    def get_synsets_nouns(self, topic_noun, persp_noun):
        scores = []
        topic_synsets = topic_noun.get_synsets()
        persp_synsets = persp_noun.get_synsets()
        for topic_syn in topic_synsets:
            for persp_syn in persp_synsets:
                distance = topic_syn.path_similarity(persp_syn)
                if distance != None:
                    scores.append(distance)
        # If extracted noun contains no synsets, mark with -1
        if len(scores)==0:
            scores = [0]
        return max(scores)
