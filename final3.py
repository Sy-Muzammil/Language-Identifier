#!/usr/bin/env python 
from __future__ import print_function
import sys
import os
import numpy as np
import gensim                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
from sets import Set
d = str(5)
#BASE_PATH = "/home/muzammil/Desktop/Summer_Project/"
BASE_PATH = "/home/muzammil/Desktop/Summer_Project/"
eng = np.random.rand(64).astype(np.float32)
tel = np.random.rand(64).astype(np.float32)
ne = np.random.rand(64).astype(np.float32)
univ = np.random.rand(64).astype(np.float32)
dummy = np.random.rand(64).astype(np.float32)



model = gensim.models.KeyedVectors.load_word2vec_format(
    BASE_PATH + "Training_data/ENG_ALL_ISCTOK_BNC_maskUN_C2SHFL_d64_m20_i1.vec",binary=True,limit=10000)




import sys
from itertools import cycle, izip

import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report as cr
np.random.seed(42)


class MLPClassifier():
    """Multi Layered Perceptron with one hidden layer"""

    def __init__(self, n_hidden=50, learning_rate=0.1, SGD=False):
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.SGD = SGD

    def softmax(self, x):
        """softmax normalization"""
        np.exp(x, x)
        x /= np.sum(x, axis=1)[:, np.newaxis]

    def fit(self, X, y, max_epochs=10):
        # uniform labels
        print ("Training is started")
        self.lb = LabelBinarizer()
        y = self.lb.fit_transform(y)
        #print(y)
        # get all sizes

        n_samples, n_features = X.shape
        self.n_outs = y.shape[1]
        #print(self.n_outs)
        n_iterations = int(max_epochs * n_samples)

        # initialize weights #NOTE smart initialization
        nO = np.sqrt(n_features)
        nH = np.sqrt(self.n_hidden)
        self.weights1_ = np.random.uniform(-1/nO, 1/nO, size=(n_features, self.n_hidden))
        self.bias1_ = np.zeros(self.n_hidden)
        self.weights2_ = np.random.uniform(-1/nH, 1/nH, size=(self.n_hidden, self.n_outs))
        self.bias2_ = np.zeros(self.n_outs)

        if self.SGD:
            # NOTE Stochastic Gradient Descent
            # initialize hidden-layer and output layer matrices 
            x_hidden = np.empty((1, self.n_hidden))
            delta_h = np.empty((1, self.n_hidden))
            x_output = np.empty((1, self.n_outs))
            delta_o = np.empty((1, self.n_outs))

            for it in xrange(1, max_epochs+1):
                for j in xrange(n_samples):
                    self._forward(X[j, None], x_hidden, x_output)
                    self._backward(X[j, None], y[j, None], x_hidden, x_output, delta_o, delta_h)
                pred = self.predict(xtest)
                #print("p:",pred)
                #print("1: ",cr(ytest, pred))

        else:
            # NOTE Gradient Descent
            # initialize hidden-layer and output layer matrices 
            x_hidden = np.empty((n_samples, self.n_hidden))
            delta_h = np.empty((n_samples, self.n_hidden))
            x_output = np.empty((n_samples, self.n_outs))
            delta_o = np.empty((n_samples, self.n_outs))

            # adjust weights by a forward pass and a backward error propagation
            for i in xrange(max_epochs):
                self._forward(X, x_hidden, x_output)
                self._backward(X, y, x_hidden, x_output, delta_o, delta_h)
                pred = self.predict(X)
                print("2: ",cr(y1, pred))

    def sigmoid(self,x):
            return 1/(1+np.exp(-x/1))
        

    # predict test patterns
    def predict(self, X):
        #print("Testng started")
        n_samples = X.shape[0]
        x_hidden = np.empty((n_samples, self.n_hidden))
        x_output = np.empty((n_samples, self.n_outs))
        self._forward(X, x_hidden, x_output)
        return self.lb.inverse_transform(x_output)

    def _forward(self, X, x_hidden, x_output):
        """Forward pass through the network"""
        #print("XX: ",X.shape)
        x_hidden[:] = np.dot(X, self.weights1_)
        x_hidden += self.bias1_
        x_hidden = self.sigmoid(x_hidden)
        x_output[:] = np.dot(x_hidden, self.weights2_)
        x_output += self.bias2_

        # apply softmax normalization
        self.softmax(x_output)

    def _backward(self, X, y, x_hidden, x_output, delta_o, delta_h):
        """Backward error propagation to update the weights"""

        # calculate derivative of output layer
        delta_o[:] = y - x_output
        delta_h[:] = np.dot(delta_o, self.weights2_.T) 

        # update weights
        self.weights2_ += self.learning_rate * np.dot(x_hidden.T, delta_o)
        self.bias2_ += self.learning_rate * np.mean(delta_o, axis=0)
        self.weights1_ += self.learning_rate * np.dot(X.T, delta_h)
        self.bias1_ += self.learning_rate * np.mean(delta_h, axis=0)




class Build_W2V:

    def __init__(self,words,tags):
        self.words = words
        self.tags = tags
        self.trigrams_words = []
        self.trigrams_tags = []
        self.dataset_tags  = []
        self.generate_ngrams()


    def generate_ngrams(self):
        
        for sent in self.words:
            for (x,y,z) in zip(sent[0:-1],sent[1:-1],sent[2:]):
                self.trigrams_words.append([x,y,z])

        
        for sent in self.tags:
            for (x,y,z) in zip(sent[0:-1],sent[1:-1],sent[2:]):
                self.trigrams_tags.append([x,y,z])
        
        self.use_embed()
        self.use_embedding()
    
    def use_embed(self):
        wrd = self.trigrams_words
        tg = self.trigrams_tags

        self.prefix1 = set()
        self.sufix1 = set()
        self.prefix2 = set()
        self.sufix2 = set()

        for i in range(len(wrd)):
            #print(wrd[i])
            for j in range(len(wrd[i])):
                #print(temp)
                if j == 0:
                    temp = wrd[i][j]
                    print(temp)
                    pre = temp[:3]
                    suf = temp[-3:]
                
                    self.prefix1.add(pre)
                    self.sufix1.add(suf)
                if j == 2:
                    temp = wrd[i][j]
                    pre = temp[:3]
                    suf = temp[-3:]
                
                    self.prefix2.add(pre)
                    self.sufix2.add(suf)


        #print len(self.prefix)
        #print len(self.sufix)
        #print "----------------"
        #print(len(self.prefix))
        #print(len(self.sufix))
        self.pre_dict1 = dict(zip(self.prefix1, range(len(self.prefix1)+1)))  #NOTE index 0 for OOV affixes
        self.suf_dict1 = dict(zip(self.sufix1, range(len(self.sufix1)+1)))
        self.pfx_embd1 = np.random.uniform(-0.5, 0.5, (len(self.prefix1)+1, 32))
        self.sfx_embd1 = np.random.uniform(-0.5, 0.5, (len(self.sufix1)+1, 32))

        self.pre_dict2 = dict(zip(self.prefix2, range(len(self.prefix2)+1)))  #NOTE index 0 for OOV affixes
        self.suf_dict2 = dict(zip(self.sufix2, range(len(self.sufix2)+1)))
        self.pfx_embd2 = np.random.uniform(-0.5, 0.5, (len(self.prefix2)+1, 32))
        self.sfx_embd2 = np.random.uniform(-0.5, 0.5, (len(self.sufix2)+1, 32))

        print(self.pfx_embd1.shape)
        print(self.sfx_embd1.shape)
        print(self.pfx_embd2.shape)
        print(self.sfx_embd2.shape)

        #print(self.predic)
        #print("========================")
        #print(self.sufdic)
        #print("========================")


    def use_embedding(self):

        wrd = self.trigrams_words
        tg = self.trigrams_tags

        self.wordvec = []
        self.lisvec = []
        for i in range(len(wrd)):
            tempvec =[]
            if wrd[i][1] in model:
                tempvec.extend(model[wrd[i][1]])
            elif wrd[i][1] not in model:
                tempvec.extend(dummy)

            temp = wrd[i][0]
            pre = temp[:3]
            suf = temp[-3:]
            x = self.pfx_embd1[self.pre_dict1.get(pre, 0)]
            tempvec.extend(x)
            x = self.sfx_embd1[self.suf_dict1.get(suf, 0)]
            tempvec.extend(x)


            temp = wrd[i][2]
            pre = temp[:3]
            suf = temp[-3:]
            x = self.pfx_embd2[self.pre_dict2.get(pre, 0)]
            tempvec.extend(x)
            x = self.sfx_embd2[self.suf_dict2.get(suf, 0)]
            tempvec.extend(x)

            tempvec = np.array(tempvec)
            self.wordvec.append(tempvec)
            if tg[i][1] == d:
                self.lisvec.append(0)
            elif tg[i][1] == "te" or tg[i][1] == "te/dl":
                self.lisvec.append("te")
            elif tg[i][1] == "ne" or tg[i][1] == "ne/dl":
                self.lisvec.append("ne")
            elif tg[i][1] == "univ":
                self.lisvec.append("univ")
            else:
                self.lisvec.append("en")
            #tempvec = np.array(tempvec.tolist())
            #npa = np.asarray(tempvec, dtype=np.float32)
            #arr = np.vstack(npa)
            #print arr
        #print"final------------->"
        #print self.wordvec
        #print" ========================================="
        # print len(self.wordvec[0])
        # print len(self.wordvec)
        # print len(self.lisvec)
        
        #print "----------------"
        self.wordvec = np.array(self.wordvec)
        self.lisvec = np.array(self.lisvec)
        print(self.wordvec.shape)
        print(self.lisvec.shape)
    
def process_data(file,words,tags):
    list1 = [] #word 
    list2 = [] #tag
    
    i = 0
    with open(sys.argv[1],"r+") as fp:
        for line in fp:
        
            if(i == 0):
                list1.append(d)
                list2.append(d)
                i+=1
            if len(line.strip()) != 0:
                #print line
                list1.append(line.split()[0])
                list2.append(line.split()[1])
            else:
                list1.append(d)
                list2.append(d)
                words.extend([list1])
                tags.extend([list2])
                list1 = []
                list2 = []
                i = 0

if __name__ == '__main__':

    words = []  #matrix of words each row containing sentence
    tags = []   # matrix of tags corresponding to each word
    
    process_data(sys.argv[1],words,tags)
    W2V = Build_W2V(words,tags)
    #WV.TrainModel()
    
    data_input = W2V.wordvec
    data_output = W2V.lisvec
    xtrain = data_input[:9000]
    ytrain = data_output[:9000]
    xtest = data_input[9000:]
    ytest = data_output[9000:]
    #X,y1 = collect_data(file_tra)
    #print(X.shape)
    clf = MLPClassifier(n_hidden=50, learning_rate=0.01, SGD=True)
    clf.fit(xtrain, ytrain, max_epochs=200)
    #D,O = collect_data(file_test)
    pred = clf.predict(xtest)
    #print("O: ",O)
    #print("pred: ",pred)
    print("0: ",cr(ytest, pred))
