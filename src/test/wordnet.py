from nltk.corpus import wordnet as wn

government = wn.synset('drone.n.01')
vaccine = wn.synset('strike.v.01')

similarity = government.path_similarity(vaccine)

print(similarity)

