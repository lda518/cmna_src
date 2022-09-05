from data_manager import Data_manager
class Analyzer:
    def __init__(self, mentions_filename, matched_filename, sentences, aspects, debater_api):
        self.mentions_filename = mentions_filename
        self.matched_filename = matched_filename
        self.debater_api = debater_api
        self.data_manager = Data_manager()
        self.sentences = sentences
        self.aspects = aspects

    # Extract all the wikipedia concepts mentioned in a sentence using the termwikifier service
    def get_sentence_mentions(self):
        term_wikifier_client = self.debater_api.get_term_wikifier_client()
        mentions_list = term_wikifier_client.run([sentence['text'] for sentence in self.sentences])
        self.data_manager.save(self.mentions_filename, mentions_list)
        breakpoint()

    def set_sentence_mentions(self):
        mentions_list = self.data_manager.load(self.mentions_filename)
        for sentence, mentions in zip(self.sentences, mentions_list):
            sentence['mentions'] = set([mention['concept']['title'] for mention in mentions])
        all_mentions = set([mention for sentence in self.sentences for mention in sentence['mentions']])
        self.all_mentions = set(all_mentions)

    def get_related_mentions(self, concept, threshold, mentions):
        term_relater_client = self.debater_api.get_term_relater_client()
        concept_mention_pairs = [[concept, mention] for mention in mentions]
        scores = term_relater_client.run(concept_mention_pairs)
        return [mention for mention, score in zip(self.all_mentions, scores) if score > threshold]

    def get_matched_mentions(self):
        matched_mentions = {}
        for aspect in self.aspects:
            matched_mentions[aspect] = self.get_related_mentions(aspect, 0.8, self.all_mentions)
            print(aspect,":",matched_mentions[aspect])
        self.matched_mentions = matched_mentions
        self.data_manager.save(self.matched_filename, self.matched_mentions)
