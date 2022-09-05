from utils.data_saver import Data_saver

config = {'directories':{'eval_path': 'stats/model_evals.csv'}}
saver = Data_saver(config)

saver.save_eval(0.6, 'test')
