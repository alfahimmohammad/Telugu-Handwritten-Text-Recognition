# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 13:52:41 2020

@author: Kather
"""
import numpy as np
import editdistance as ed

unicodes = list(np.load('unicodes.npy',allow_pickle=True))
with open('lexicon.txt','r',encoding='utf-8') as f:
    lexicon = f.readlines()
    
dict_trainnval = {}
infile = open('Newtest.txt','r',encoding='utf-8')
integer = 0
for line in infile:
    integer += 1
    lineSplit = line.strip().split(' ')
    dict_trainnval[lineSplit[0]] = integer  
infile.close()
    
def decode(yPred):  #Best Path
    """
    texts = []
    for y in yPred:
        label = list(np.argmax(y[2:],1))
        label = [k for k, g in itertools.groupby(label)]
        text = labelsToText(label)
        texts.append(text)
    return texts
    """
    texts = []
    for i in range(yPred.shape[0]):
      y = yPred[i,2:,:]
      y = np.reshape(y,(1,30,97))
      pred =  K.get_value(K.ctc_decode(y, input_length=np.ones(y.shape[0])*30, greedy=False, beam_width=3, top_paths=1)[0][0])[0]
      word = ""
      for i in range(len(pred)):
        if pred[i] == len(unicodes):
          word += ""
        else:
          word += unicodes[pred[i]]
      lex = np.zeros(len(lexicon))
      for i in range(len(lex)):
          lex[i] = ed.eval(word,lexicon[i])
      predWord = lexicon[np.argmin(lex)]
      texts.append(predWord)
    return texts

def truncateLabel(text):
    cost = 0
    for i in range(len(text)):
        if i!=0 and text[i]==text[i-1]:
            cost+=2
        else:
            cost+=1
        if cost>29:
            return text[:i]
    return text

def getData():
    dataset = []
    f = open('Newtest.txt','r',encoding='utf-8')
    for line in f:
        if not line or line[0] =='#':
            continue
        lineSplit = line.strip().split(' ')
        assert len(lineSplit) >= 2
        fileName = lineSplit[0] 
        text = truncateLabel(' '.join(lineSplit[1:]))

        for ch in text:
            if ch not in unicodes:
                print(ch,('0'+hex(ord(ch))[2:]))


        dataset.append((text,fileName))
    f.close()
    return dataset

with open('TestPaper.pickle','rb') as f:
    imtest = pickle.load(f)
f.close()    

def Test():
    validation = getData()
    imgs = []
    trueText = []
    for (i,path) in validation:
        trueText.append(i)
        nn = dict_trainnval[path]
        imgs.append(np.reshape(imtest[:,:,nn-1],(32,128,1)))
    model = getModel(False)
    model.load_weights('Saved/model_83.hdf5')
    imgs = np.array(imgs)
    
    outputs = model.predict(imgs)
    predText = decode(outputs)
    
    wordOK = 0
    wordTot = 0
    charDist = 0
    charTot = 0
    for i in range(len(predText)):
        print(predText[i],trueText[i])
        wordOK += 1 if predText[i]==trueText[i] else 0
        wordTot += 1
        dist = editdistance.eval(predText[i],trueText[i])
        charDist += dist
        charTot += len(trueText[i])
        
    charDist = charDist/charTot
    wordOK = wordOK/wordTot
    print('Character Distance(CER):'+str(charDist*100))
    print('Word Accuracy:'+str(wordOK*100))    
    
Test()