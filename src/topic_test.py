from debater_python_api.api.clients.narrative_generation_client import Polarity
from debater_python_api.api.debater_api import DebaterApi
import os

api_key = os.environ['DEBATER_API_KEY']
debater_api = DebaterApi(api_key)

sentence = 'We should take advantage of the fact that drone strikes are legal'
term_wikifier_client = debater_api.get_term_wikifier_client()
mentions_list = term_wikifier_client.run([sentence])
sentence_2 = 'Drones Should Be Used to Take Out Enemy Combatants'
mentions_list_2 = term_wikifier_client.run([sentence_2])

print(mentions_list)
print(mentions_list_2)
