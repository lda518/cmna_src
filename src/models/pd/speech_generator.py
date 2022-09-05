from debater_python_api.api.clients.narrative_generation_client import Polarity
class Speech_generator:
    def __init__(self, all_candidates, debater_api, topic, dc):
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

        self.opening_statement_customization_dict = opening_statement_customization_dict  
        self.all_candidates = all_candidates
        self.debater_api = debater_api
        self.topic = topic
        self.dc = dc

    def generate(self):
        print('Generating speech:')

        sentences = [ sentence['text'] for sentence in self.all_candidates if sentence['evidence'] > 0.2]
        print('Number of sentences passed to speech generation: ' + str(len(sentences)))
        narrative_generation_client = self.debater_api.get_narrative_generation_client()
        sentence_topic_dicts = [{'sentence' : sentence, 'topic' : self.topic } for sentence in sentences]
        pro_con_scores = self.debater_api.get_pro_con_client().run(sentence_topic_dicts)
        speech = narrative_generation_client.run(topic=self.topic, dc=self.dc, sentences=sentences,
                                                 pro_con_scores=pro_con_scores, polarity=Polarity.PRO,
                                                 customizations=[self.opening_statement_customization_dict])
        print('\n\nSpeech:' + speech.status)
        if (speech.status == 'DONE'):
            print(speech)
        else:
            print(speech.error_message)
