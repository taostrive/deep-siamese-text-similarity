import numpy as np
import re
import itertools
from collections import Counter
import numpy as np
import time
import gc
from tensorflow.contrib import learn
from gensim.models.word2vec import Word2Vec
import gzip
from random import random
from preprocess import MyVocabularyProcessor
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

class InputHelper(object):
    
    def getTsvData(self, filepath):
        print("Loading training data from "+filepath)
        x1=[]
        x2=[]
        y=[]
        # positive samples from file
        for line in open(filepath):
            l=line.strip().split("\t")
            if len(l)<2:
                continue
            if random() > 0.5:
               x1.append(l[0].lower())
               x2.append(l[1].lower())
            else:
               x1.append(l[1].lower())
               x2.append(l[0].lower())
            #Note here y is a list which will be filled with purely 1!!
            y.append(1)#np.array([0,1]))
        # generate random negative samples
        #negative samples' length is twice of positive ones!
        #The np.asarray converts a list into an array
        combined = np.asarray(x1+x2)
        #>>> np.arange(3)
        #array([0, 1, 2])
        #>>> np.random.permutation(10)
        #array([1, 7, 4, 3, 0, 9, 2, 5, 8, 6])
        shuffle_indices = np.random.permutation(np.arange(len(combined)))
        #NumPy arrays may be indexed with other arrays:
        #https://docs.scipy.org/doc/numpy-1.13.0/user/basics.indexing.html#index-arrays
        combined_shuff = combined[shuffle_indices]
        #Tao: the number of negative samples are twice of positive ones
        for i in xrange(len(combined)):
            x1.append(combined[i])
            x2.append(combined_shuff[i])
            y.append(0) #np.array([1,0]))
        return np.asarray(x1),np.asarray(x2),np.asarray(y)


    def getTsvTestData(self, filepath):
        print("Loading testing/labelled data from "+filepath)
        x1=[]
        x2=[]
        y=[]
        # positive samples from file
        for line in open(filepath):
            l=line.strip().split("\t")
            if len(l)<3:
                continue
            x1.append(l[1].lower())
            x2.append(l[2].lower())
            y.append(int(l[0])) #np.array([0,1]))
        return np.asarray(x1),np.asarray(x2),np.asarray(y)  
 
    def batch_iter(self, data, batch_size, num_epochs, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        data = np.asarray(data)
        print(data)
        print(data.shape)
        data_size = len(data)
        num_batches_per_epoch = int(len(data)/batch_size) + 1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            # The shuffle operation here only re-orders the tuples in the data list but not the elements in the tuple itself
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                #To avoid index overflow
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]
                
    def dumpValidation(self,x1_text,x2_text,y,shuffled_index,dev_idx,i):
        print("dumping validation "+str(i))
        x1_shuffled=x1_text[shuffled_index]
        x2_shuffled=x2_text[shuffled_index]
        y_shuffled=y[shuffled_index]
        x1_dev=x1_shuffled[dev_idx:]
        x2_dev=x2_shuffled[dev_idx:]
        y_dev=y_shuffled[dev_idx:]
        del x1_shuffled
        del y_shuffled
        with open('validation.txt'+str(i),'w') as f:
            for text1,text2,label in zip(x1_dev,x2_dev,y_dev):
                f.write(str(label)+"\t"+text1+"\t"+text2+"\n")
            f.close()
        del x1_dev
        del y_dev
    
    # Data Preparatopn
    # ==================================================
    
    
    def getDataSets(self, training_paths, max_document_length, percent_dev, batch_size):
        #Tao: x1_text, x2_text, y all are 1-D np arrays
        x1_text, x2_text, y=self.getTsvData(training_paths)
        
        # Build vocabulary
        print("Building vocabulary")
        vocab_processor = MyVocabularyProcessor(max_document_length,min_frequency=0)
        #Tao:
        vocab_processor.fit_transform(np.concatenate((x2_text,x1_text),axis=0))
        print("Length of loaded vocabulary ={}".format( len(vocab_processor.vocabulary_)))
        i1=0
        train_set=[]
        dev_set=[]
        sum_no_of_batches = 0
        #Tao: x1 and x2 are both 2-D arrays
        x1 = np.asarray(list(vocab_processor.transform(x1_text)))
        x2 = np.asarray(list(vocab_processor.transform(x2_text)))
        # Randomly shuffle data
        np.random.seed(131)
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x1_shuffled = x1[shuffle_indices]
        x2_shuffled = x2[shuffle_indices]
        y_shuffled = y[shuffle_indices]
        dev_idx = -1*len(y_shuffled)*percent_dev//100
        del x1
        del x2
        # Split train/test set
        self.dumpValidation(x1_text,x2_text,y,shuffle_indices,dev_idx,0)
        # TODO: This is very crude, should use cross-validation
        x1_train, x1_dev = x1_shuffled[:dev_idx], x1_shuffled[dev_idx:]
        x2_train, x2_dev = x2_shuffled[:dev_idx], x2_shuffled[dev_idx:]
        y_train, y_dev = y_shuffled[:dev_idx], y_shuffled[dev_idx:]
        print("Train/Dev split for {}: {:d}/{:d}".format(training_paths, len(y_train), len(y_dev)))
        #Tao:
        #root@docker:/opt/siamese_nn/deep-siamese-text-similarity-master# wc -l person_match.train2
        #13198 person_match.train2
        #Train/Dev split for person_match.train2: 35634/3960   -->13198 * 3=39594  35634+3960=39594
        sum_no_of_batches = sum_no_of_batches+(len(y_train)//batch_size)
        train_set=(x1_train,x2_train,y_train)
        dev_set=(x1_dev,x2_dev,y_dev)
        gc.collect()
        return train_set,dev_set,vocab_processor,sum_no_of_batches
    
    def getTestDataSet(self, data_path, vocab_path, max_document_length):
        x1_temp,x2_temp,y = self.getTsvTestData(data_path)

        # Build vocabulary
        vocab_processor = MyVocabularyProcessor(max_document_length,min_frequency=0)
        vocab_processor = vocab_processor.restore(vocab_path)
        print len(vocab_processor.vocabulary_)

        x1 = np.asarray(list(vocab_processor.transform(x1_temp)))
        x2 = np.asarray(list(vocab_processor.transform(x2_temp)))
        # Randomly shuffle data
        del vocab_processor
        gc.collect()
        return x1,x2, y

