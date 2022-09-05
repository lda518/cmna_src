
import os
from debater_python_api.api.clients.narrative_generation_client import Polarity
from debater_python_api.api.debater_api import DebaterApi
from miner import Miner
api_key = os.environ['DEBATER_API_KEY']
debater_api = DebaterApi(api_key)
