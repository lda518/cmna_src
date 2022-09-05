import nltk
from utils.dataset_loader import Dataset_loader

lines = 'the coffee pot is very big'

is_noun = lambda pos: pos[:2] == 'NN'
tokenized = nltk.word_tokenize(lines)
nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)] 

print(nouns)
