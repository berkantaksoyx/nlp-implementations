# -*- coding: utf-8 -*-

import math
import random
import re
import codecs


# ngramLM CLASS
class ngramLM:
    """Ngram Language Model Class"""

    # Create Empty ngramLM
    def __init__(self):
        self.numOfTokens = 0
        self.sizeOfVocab = 0
        self.numOfSentences = 0
        self.sentences = []
        # TO DO
        self.vocab_dict = {}
        self.bigram_dict = {}

    # INSTANCE METHODS
    def trainFromFile(self, fn):
        pattern = r"""(?x)
(?:[A-ZÇĞIİÖŞÜ]\.)+
| \d+(?:\.\d*)?(?:\'\w+)?
| \w+(?:-\w+)*(?:\'\w+)?
| \.\.\.
| [][,;.?():_!#^+$%&><|/{()=}\"\'\\\"\`-]
"""
        with open(fn, 'r', encoding='utf8') as file:
            for line in file:
                line = line.strip()
                if len(line) == 0:
                    continue
                pattern2 = r'(?<=[?!]|(?<!\d)\.)(?=\s+|\[)'
                sentences = re.split(pattern2,line)

                for sentence in sentences:
                    sentence = sentence.replace("I","ı").replace("İ","i")
                    sentence = sentence.lower()
                    tokenized_sentence = re.findall(pattern, sentence)
                    tokenized_sentence = ["<s>"] + tokenized_sentence + ["</s>"]
                    self.sentences.append(tokenized_sentence)
                    self.numOfTokens += len(tokenized_sentence)
        self.numOfSentences = len(self.sentences)


        ## unigram training ##

        sentences = self.sentences
        vocab = self.vocab_dict
        for sentence in sentences:
            for token in sentence:
                vocab[token] = vocab.get(token, 0) + 1
        self.sizeOfVocab = len(vocab)



        ## bigram training
        bigrams = self.bigram_dict
        for sentence in sentences:
            for bigram in zip(sentence, sentence[1:]):
                bigrams[bigram] = bigrams.get(bigram, 0) + 1


        return



    

    def vocab(self):
        vocab = self.vocab_dict

        vocab_list = sorted(vocab.items(), key=lambda item: (-item[1], item[0]))


        return vocab_list

    

    def bigrams(self):
        bigrams = self.bigram_dict
        bigram_list = sorted(bigrams.items(), key=lambda item: (-item[1], item[0]))
        return bigram_list
    

    def unigramCount(self, word):
        unigram_list = self.vocab_dict
        unigram_count = unigram_list.get(word)
        if unigram_count is None:
            return 0
        return unigram_count

    

    def bigramCount(self, bigram):
        bigram_list = self.bigram_dict
        bigram_count = bigram_list.get(bigram)
        if bigram_count is None:
            return 0
        return bigram_count

    

    def unigramProb(self, word):
        unigram_count = self.unigramCount(word)
        unigram_prob = (unigram_count) / (self.numOfTokens)
        return unigram_prob

    
    # returns unsmoothed unigram probability value

    def bigramProb(self, bigram):
        bigram_count = self.bigramCount(bigram)
        w1 = bigram[0]
        w1_count = self.unigramCount(w1)
        if w1_count == 0:
            bigram_prob = 0
        else:
            bigram_prob = (bigram_count) / (w1_count)


        return bigram_prob



    
    # returns unsmoothed bigram probability value

    def unigramProb_SmoothingUNK(self, word):
        unigram_count = self.unigramCount(word)
        vocab_size = self.sizeOfVocab
        smoothed_probability = (unigram_count + 1) / (self.numOfTokens + (vocab_size +1))
        return smoothed_probability

    
    # returns smoothed unigram probability value

    def bigramProb_SmoothingUNK(self, bigram):
        bigram_count = self.bigramCount(bigram)
        vocab_size = self.sizeOfVocab
        w1 = bigram[0]
        w1_count = self.unigramCount(w1)
        smoothed_probability = float((bigram_count + 1) / (w1_count + vocab_size +1))
        return smoothed_probability

    
    # returns smoothed bigram probability value

    def sentenceProb(self, sent):
        sentenceProb = 1
        if len(sent) == 1:
            sentenceProb = self.unigramProb_SmoothingUNK(sent[0])
        else:
            for bigram in zip(sent, sent[1:]):
                bigram_prob = self.bigramProb_SmoothingUNK(bigram)
                sentenceProb *= bigram_prob
        return sentenceProb

    
    # sent is a list of tokens
    # returns the probability of sent using smoothed bigram probability values
    def topKsample(self,word,maxFollowWords=1):
        bigram_list = self.bigrams()
        bigram_of_word = []
        sampling_list = []
        total_freq = 0

        for bigram in bigram_list:
            if bigram[0][0] == word:
                if maxFollowWords > len(bigram_of_word):
                    bigram_of_word.append(bigram)
                    freq_of_word = bigram[1]
                    total_freq += freq_of_word
                    for i in range(freq_of_word):
                        sampling_list.append(bigram)
        x = random.randint(0,len(sampling_list)-1)
        next_word = sampling_list[x][0][1]
        return next_word







        return
    def generateSentence(self, sent=["<s>"], maxFollowWords=1, maxWordsInSent=20):

        generated_sentence = []
        next_word = sent[-1]
        generated_sentence = sent + generated_sentence
        while next_word != "</s>":
            if len(generated_sentence) == maxWordsInSent:
                break

            next_word = self.topKsample(next_word,maxFollowWords)
            generated_sentence.append(next_word)

        print(generated_sentence)


        return







            
            
        
        
