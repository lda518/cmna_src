#from utils.target_extractor import Target_extractor
from utils.new_target_extractor import Target_extractor 
import pandas as pd
import os
import yaml

topic = "Vaccination must be made compulsory"
perspective = "The state must keep it's community safe."
#topic = "The BBC should be free to blaspheme"
#perspective = "his was a piece of art, advertised and described as such, those likely to be offended were quite welcome not to watch it."
topic = "Drones Should Be Used to Take Out Enemy Combatants"
perspective = "Drone strikes are legal under international law."
#
#topic = "Raise the school leaving age to 18"
#perspective = "With more education comes more opportunities"

real_path = os.path.realpath(__file__)
dir_path = os.path.dirname(real_path)
config_file = os.path.join(dir_path, 'config.yaml')
with open(config_file) as file:
    config = yaml.safe_load(file)

pair = pd.DataFrame({'topic':[topic], 'perspective':[perspective]})

target_extractor = Target_extractor(config)
final_score_dict = target_extractor.extract_combined_scores_dict(pair)
