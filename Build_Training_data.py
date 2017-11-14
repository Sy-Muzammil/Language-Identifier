import sys
import os
import numpy as np
import gensim
eng = np.random.uniform(-10,10,size=(1,64))
tel = np.random.uniform(-10,10,size=(1,64))
ne = np.random.uniform(-10,10,size=(1,64))
univ = np.random.uniform(-10,10,size=(1,64))
dummy = np.random.uniform(-10,10,size=(1,64))
"""
python filename.py eng_tel.txt te
list1 =  word
list 2 = tag

"""
batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2
data_index = 0

BASE_PATH = "/home/muzammil/Desktop/Summer_Project/"
model = gensim.models.KeyedVectors.load_word2vec_format(
	BASE_PATH + "Training_data/ENG_ALL_ISCTOK_BNC_maskUN_C2SHFL_d64_m20_i1.vec",binary=True)

def use_embedding(trigrams_words,trigrams_tags):
	for i,chunk in enumerate(trigrams_words):
		#print chunk
		for j,word in enumerate(chunk):
			
			if(trigrams_words[i][j] == dummy):
				continue
			
			elif word in model:
				trigrams_words[i][j] = model[word]
			
			else:
				if(trigrams_tags[i][j] == "te"):
					trigrams_words[i][j] = tel

				elif (trigrams_tags[i][j] == "ne"):
					trigrams_words[i][j] = ne 

				elif (trigrams_tags[i][j] == "univ"):
					trigrams_words[i][j] = univ
				else:
					trigrams_words[i][j] = eng

	print trigrams_words

def generate_ngrams(words,tags):

	trigrams_words = []
	trigrams_tags = []
	for sent in words:
		for (x,y,z) in zip(sent[0:-1],sent[1:-1],sent[2:]):
			trigrams_words.append([x,y,z])

	
	for sent in tags:
		for (x,y,z) in zip(sent[0:-1],sent[1:-1],sent[2:]):
			trigrams_tags.append([x,y,z])
	
	# print trigrams_words
	# print "-------------------------------------------------------------------------------"
	# print trigrams_tags
	# print "-------------------------------------------------------------------------------"
	use_embedding(trigrams_words,trigrams_tags)


	

if __name__ == "__main__":
	list1 = []
	list2 = []
	words = [] #matrix of words each row containing sentence
	tags = []# matrix of tags corresponding to each word
	i = 0
	with open(sys.argv[1],"r+") as fp:
		for line in fp:
		
			if(i == 0):
				list1.append(dummy)
				list2.append(dummy)
				i+=1
			if len(line.strip()) != 0:
				#print line
				list1.append(line.split()[0])
				list2.append(line.split()[1])
			else:
				list1.append(dummy)
				list2.append(dummy)
				words.extend([list1])
				tags.extend([list2])
				list1 = []
				list2 = []
				i = 0
	#print words
	#print "------------------------------------------------------------------"
	generate_ngrams(words,tags)
	#generate_embedding(words,tags,)
