import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np 
import pickle
import random
from collections import Counter

lemmatizer = WordNetLemmatizer()
hm_lines = 10000

def create_lexicon(pos, neg):
	lexicon = []
	for file in [pos,neg]:
		with open(file, 'r') as f:
			content = f.readlines()
			for l in content[:hm_lines]:
				file_lex = word_tokenize(l.lower())
				lexicon += list(file_lex) 

	lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
	dict_words_count = Counter(lexicon)                # shows output of wordin the format { 'the': 2345, 'and':45876, etc....}
	
	lex_final = []

	for w in dict_words_count:
		if 1000 < dict_words_count[w] < 5:
			lex_final.append(w)

	return lex_final 

def sample_handling(sample, lexicon, classification):
	featureset = []

	with open(sample,'r') as f:
		contents = f.readlines()
		for l in contents[:hm_lines]:
			content_words = word_tokenize(l.lower())
			content_words = [lemmatizer.lemmatize(i) for i in content_words]
			features = np.zeros(len(lexicon))
			for t in content_words:
				if t.lower() in lexicon:
					word_index = lexicon.index[t.lower()]
					features[index_values] = 1 #rememmber one hot array
			features = list(features)
			featureset.append([features, classification ])

	return featureset	


def create_features_and_labels(pos, neg, test_size = 0.1):
	lexicon = create_lexicon(pos, neg)
	features = []

	features += sample_handling('/Users/avnish/LearningNewstuff/Data_Analysis/NLP/pos.txt',lexicon,[1,0])
	features += sample_handling('/Users/avnish/LearningNewstuff/Data_Analysis/NLP/neg.txt',lexicon,[0,1])

	random.shuffle(features)

	features = np.array(features)
	print (features)

	testing_size = int(test_size*len(features))

	train_x = list(features[:,0][:-testing_size])
	train_y = list(features[:,1][:-testing_size])

	test_x = list(features[:,0][-testing_size:])
	test_y = list(features[:,1][-testing_size:])

	return train_x,train_y,test_x,test_y

if __name__ == '__main__':
	train_x,train_y,test_x,test_y = create_features_and_labels('/Users/avnish/LearningNewstuff/Data_Analysis/NLP/pos.txt','/Users/avnish/LearningNewstuff/Data_Analysis/NLP/neg.txt')

	with open('/Users/avnish/LearningNewstuff/Data_Analysis/NLP/sentiment_set_og.pickle','wb') as f:
		pickle.dump([train_x,train_y,test_x,test_y],f)
















