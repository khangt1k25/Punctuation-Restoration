import numpy as np
from keras.preprocessing import sequence
from nltk.corpus import gutenberg
from string import punctuation
import nltk
from underthesea import word_tokenize
from collections import defaultdict

def count_comma(sent):
    return sent.count(",")

def count_word(string):
    return(len(string.split()))

def cleaning(raw_texts):
    '''
    Clean other punct, and other simple processs
    '''
    data = []
    for sent in raw_texts:
        sent = sent.replace('\n','')
        sent = sent.replace(':',',')
        sent = sent.replace('!','.')
        sent = sent.replace('?','.')
        sent = sent.replace(';',',')
        sent = sent.replace('"','')
        sent = sent.replace(')','')
        sent = sent.replace('(','')
        sent = sent.replace('“','')
        sent = sent.replace('”','')
        sent = sent.replace('-','')
        sent = sent.replace('_','')
        sent = sent.replace('+','')
        sent = sent.replace('=','')
        sent = sent.replace('[','')
        sent = sent.replace(']','')
        sent = sent.replace('{','')
        sent = sent.replace('}','')
        sent = sent.replace('*','')
        sent = sent.replace('&','')
        sent = sent.replace('^','')
        sent = sent.replace('%','')
        sent = sent.replace('$','')
        sent = sent.replace('#','')
        sent = sent.replace('@','')
        sent = sent.replace('!','')
        sent = sent.replace('`','')
        sent = sent.replace('~','')
        sent = sent.replace('/','')
        sent = sent.replace('|','')
        sent = sent.replace('…','')
        sent = sent.replace(',.','')
        if count_comma(sent) >= 2:
            if count_word(sent) >= 20 and count_word(sent) <= 50:
                sent = word_tokenize(sent)
                sent = ' '.join(sent)
                sent = sent+'\n'
                data.append(sent)
    return data



def create_label(text):

    '''
    Take a string -> intext and label
    '''
    tokens = word_tokenize(text)
    words = []
    ids_punct = {',':[], '.':[]}
    i = 0
    for token in tokens:
        if token not in ids_punct.keys():
            words.append(token)
            i+=1
        else:
            ids_punct[token].append(i-1)

    label = [0]*len(words)
    for pun, ids in ids_punct.items():
        for index in ids:
            label[index] = 1 if pun == ',' else 2
    
    in_text = '<fff>'.join(words)
    return in_text, label



def preprocessing_train_data(RAW_PATH = './data/Data_byADuc.txt', IN_TEXT_PATH = './demo_data/text.txt', LABEL_PATH = './demo_data/label.txt'):
    # start processing
    with open(RAW_PATH, 'r') as f:
        lines = f.read().splitlines()


    lines = cleaning(lines[:1000])
    texts, labels = [], []
    for text in lines:
        in_text, label = create_label(text)
        texts.append(in_text)
        labels.append(label)


    with open(IN_TEXT_PATH, 'w') as f:
        for text in texts:
            f.write(text)
            f.write('\n')

    with open(LABEL_PATH, 'w') as f:
        for label in labels :
            label = [str(ele) for ele in label]
            label = ' '.join(label)
            f.write(label)
            f.write('\n')



preprocessing_train_data()