# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 19:57:11 2020

@author: Kather
"""
import os
import numpy as np
from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda, BatchNormalization
from keras.layers.merge import average, concatenate
from keras.models import Model
from keras.layers.recurrent import LSTM
from keras.optimizers import Adadelta
import keras.callbacks
import random
import cv2
import editdistance
import itertools
import matplotlib.pyplot as plt
#from VizCallbackTrial import VizCallback

os.chdir('D:/OCR/NewDataset')
batchSize = 64
valBatchSize = 20
width = 128
height = 32
valSplit = 0.02
trainSamplesPerEpoch = 25000
valSamplesPerEpoch = 1000
noEpochs = 30
wordFile = 'newDataset.txt'
unicodes = list(np.load('unicodes.npy',allow_pickle=True))



class VizCallback(keras.callbacks.Callback):

    def __init__(self, test_func, nextVal, num_display_words=1):
        self.test_func = test_func
        self.text_img_gen = nextVal
        self.num_display_words = num_display_words
        
    def on_epoch_end(self, epoch, logs={}):
        wordBatch = next(self.text_img_gen)[0]
        imgs = wordBatch['inputX']
        trueText = np.uint8(wordBatch['label'])
        out = self.test_func([imgs])[0]
        predText = decode(out)
        wordOK = 0
        wordTot = 0
        charDist = 0
        charTot = 0
        for i in range(len(predText)):    
            true = labelsToText(list(trueText[i]))
            wordOK += 1 if predText[i]==true else 0
            wordTot += 1
            dist = editdistance.eval(predText[i],true)
            charDist += dist
            charTot += len(true)
            print(true,predText[i])
        charDist = charDist/charTot
        wordOK = wordOK/wordTot
        print('Character Distance(CER):'+str(charDist*100))
        print('Word Accuracy:'+str(wordOK*100))
        


def plotHistory(history, key = 'loss'):
    plt.figure()
    plt.plot(history.epoch, history.history['val_'+key],'--', label=' Val')
    plt.plot(history.epoch, history.history[key],label=' Train')
    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_',' ').title())
    plt.legend()
    plt.xlim([0,max(history.epoch)])

def preprocess(img, imgSize, dataAugmentation=False):
    if img is None:
        img = np.zeros((imgSize[0],imgSize[1],1))
    if dataAugmentation:
    	stretch = (random.random() - 0.5) # -0.5 .. +0.5
    	wStretched = max(int(img.shape[1] * (1 + stretch)), 1) # random width, but at least 1
    	img = cv2.resize(img, (wStretched, img.shape[0])) # stretch horizontally by factor 0.5 .. 1.5
    (wt, ht) = imgSize
    h = img.shape[0]
    w = img.shape[1]
    fx = w / wt
    fy = h / ht
    f = max(fx, fy)
    newSize = (max(min(wt, int(w / f)), 1), max(min(ht, int(h / f)), 1)) # scale according to f (result at least 1 and at most wt or ht)
    img = cv2.resize(img, newSize)
    target = np.ones([ht, wt]) * 255
    target[0:newSize[1], 0:newSize[0]] = img
    img = cv2.transpose(target)
    (m, s) = cv2.meanStdDev(img)
    m = m[0][0]
    s = s[0][0]
    img = img - m
    img = img / s if s>0 else img
    return np.reshape(img,(img.shape[0],img.shape[1],1))

def textToLabels(text):
    ret = []
    for c in text:
        ret.append(unicodes.index(c))
    return ret

def labelsToText(labels):
    ret = []
    for c in labels:
        if c == len(unicodes):
            ret.append("")
        else:
            ret.append(unicodes[c])
    return "".join(ret)
        

class DataGenerator(keras.callbacks.Callback):
    def __init__(self,batchSize,valBatchSize,width,height,valSplit,downsampleFactor = 4 , maxStringLen = 32):
        self.batchSize = batchSize
        self.imgW = width
        self.imgH = height
        self.downSampleFactor = downsampleFactor
        self.valSplit = valSplit
        self.blankLabel = len(unicodes)
        self.wordFile = wordFile
        self.maxStringLen = maxStringLen
        self.trainSet = None
        self.valSet = None
        self.dataset = []
        self.trainIndex = 0
        self.valBatchSize = valBatchSize
        self.valIndex = 0
        self.getData()
        
    def getData(self):
        f = open(self.wordFile,'r',encoding='utf-8')
        for line in f:
            if not line or line[0] =='#':
                continue
            lineSplit = line.strip().split(' ')
            assert len(lineSplit) >= 2
            fileName = lineSplit[0]
            text = self.truncateLabel(' '.join(lineSplit[1:]))
            
            for ch in text:
                if not ch in unicodes:
                    print(ch,('0'+hex(ord(ch))[2:]))
                    
            #text = text.replace('\n','')              #Removing LF newlines
            self.dataset.append((text,fileName))
        self.splitData()
        f.close()
            
    def splitData(self):
        random.shuffle(self.dataset)
        self.valSet = self.dataset[:int(self.valSplit*len(self.dataset))]
        random.shuffle(self.valSet)
        self.trainSet = self.dataset[int(self.valSplit*len(self.dataset)):]
        random.shuffle(self.trainSet)
        self.trainIndex = 0
        self.valIndex = 0
    
    def truncateLabel(self,text):
        cost = 0
        for i in range(len(text)):
            if i!=0 and text[i]==text[i-1]:
                cost+=2
            else:
                cost+=1
            if cost>self.maxStringLen:
                return text[:i]
        return text
    
    def getBatch(self,index,size,train):
        if train:
            batch = self.trainSet[index:index+size]
        else:
            batch = self.valSet[index:index+size]
        imgs = []
        labels = np.ones([size, self.maxStringLen])*(len(unicodes))
        inputLength = np.zeros([size, 1])
        labelLength = np.zeros([size, 1])
        for i in range(size):
            img = cv2.imread(batch[i][1],0)
            imgs.append(preprocess(img,(width,height),False))
            labels[i, 0:len(batch[i][0])] = textToLabels(batch[i][0])
            labelLength[i] = len(batch[i][0])
            inputLength[i] = self.imgW // self.downSampleFactor - 2
        imgs = np.asarray(imgs)
        inputs = {
                'inputX': imgs,
                'label': labels,
                'inputLen':inputLength,
                'labelLen':labelLength,
                    }
        outputs = {'ctc':np.zeros([size])}
        return (inputs,outputs)
            
    def nextTrain(self):
        while True:
            if self.trainIndex + self.batchSize >= len(self.trainSet):
                self.trainIndex = self.trainIndex % self.batchSize
                random.shuffle(self.trainSet)   
            ret = self.getBatch(self.trainIndex,self.batchSize,True)
            self.trainIndex += self.batchSize
            yield ret
            
    def nextVal(self):
        while True:
            if self.valIndex + self.valBatchSize >= len(self.valSet):
                self.valIndex = self.valIndex % self.valBatchSize
                random.shuffle(self.valSet)   
            ret = self.getBatch(self.valIndex,self.valBatchSize,False)
            self.valIndex += self.valBatchSize
            yield ret

def ctcLambdaFunc(args):
    yPred, labels, inputLength, labelLength = args
    yPred = yPred[:,2:,:]
    loss = K.ctc_batch_cost(labels,yPred,inputLength,labelLength)
    return loss

def getModel(training):
    inputShape = (128,32,1)
    kernelVals = [5,5,3,3,3]
    convFilters = [32,64,128,128,256]
    strideVals = [(2,2),(2,2),(1,2),(1,2),(1,2)]
    rnnUnits = 256
    maxStringLen = 32
    inputs = Input(name = 'inputX', shape = inputShape, dtype = 'float32')
    inner = inputs
    for i in range(len(kernelVals)):
        inner = Conv2D(convFilters[i],(kernelVals[i],kernelVals[i]),padding = 'same',\
                       name = 'conv'+str(i), kernel_initializer = 'he_normal')(inner)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = MaxPooling2D(pool_size = strideVals[i],name = 'max' + str(i+1))(inner)
    inner = Reshape(target_shape = (maxStringLen,rnnUnits), name = 'reshape')(inner)
    
    LSF = LSTM(rnnUnits,return_sequences=True,kernel_initializer='he_normal',name='LSTM1F')(inner)
    LSB = LSTM(rnnUnits,return_sequences=True, go_backwards = True, kernel_initializer='he_normal',name='LSTM1B')(inner)
    LSB = Lambda(lambda inputTensor: K.reverse(inputTensor,axes=1))(LSB)
    
    LS1 = average([LSF,LSB])
    LS1 = BatchNormalization()(LS1)
    
    LSF = LSTM(rnnUnits,return_sequences=True,kernel_initializer='he_normal',name='LSTM2F')(LS1)
    LSB = LSTM(rnnUnits,return_sequences=True, go_backwards = True, kernel_initializer='he_normal',name='LSTM2B')(LS1)
    LSB = Lambda(lambda inputTensor: K.reverse(inputTensor,axes=1))(LSB)
    
    LS2 = concatenate([LSF,LSB])
    LS2 = BatchNormalization()(LS2)
    yPred = Dense(len(unicodes)+1,kernel_initializer='he_normal',name='dense2')(LS2)
    yPred = Activation('softmax',name='softmax')(yPred)
    #Model(inputs = inputs,outputs = yPred).summary()
    
    labels = Input(name='label', shape=[32], dtype='float32')
    inputLength = Input(name='inputLen', shape=[1], dtype='int64')     # (None, 1)
    labelLength = Input(name='labelLen', shape=[1], dtype='int64')
    
    lossOut = Lambda(ctcLambdaFunc, output_shape=(1,), name='ctc')([yPred, labels, inputLength, labelLength])
    
    if training:
        return Model(inputs = [inputs, labels, inputLength, labelLength], outputs=[lossOut,yPred])
    return Model(inputs=[inputs], outputs=yPred) 
    
def decode(yPred):  #Best Path
    texts = []
    for y in yPred:
        label = list(np.argmax(y[2:],1))
        label = [k for k, g in itertools.groupby(label)]
        text = labelsToText(label)
        texts.append(text)
    return texts
    
def train(loader,transLearn = False):
    model = getModel(True)
    if transLearn:
        model.load_weights('D:/OCR/NewDataset/synthModel.h5')
     
    model.compile(loss={'ctc': lambda yTrue,yPred: yPred},optimizer = Adadelta())
    
    test_func = K.function([model.inputs[0]],[model.outputs[1]])
    viz_cb = VizCallback(test_func, loader.nextVal())
    
    h = model.fit_generator(generator=loader.nextTrain(),
                    steps_per_epoch=int(trainSamplesPerEpoch / batchSize),
                    epochs=noEpochs,
                    callbacks=[viz_cb,loader],
                    validation_data = loader.nextVal(),
                    validation_steps = int(valSamplesPerEpoch / valBatchSize)
                    )
    
    if transLearn:
        model.save('D:/OCR/Documents/[8]WordsSKCor/transModel.h5')
    else:
        model.save('D:/OCR/newDataset/synthModel.h5')
    plotHistory(h)
    return model,h

def plotAccChange(loader,transLearn = False):
    if transLearn:
        os.chdir('D:/OCR/Documents/Models')
    else:
        os.chdir('D:/OCR/NewDataset/Models')
    accs = np.zeros((noEpochs,1))
    model = getModel(False)
    validation = loader.valSet
    imgs = []
    trueText = []
    for (i,path) in validation:
        trueText.append(i)
        img = cv2.imread(path,0)
        imgs.append(preprocess(img,(width,height),False))
    imgs = np.array(imgs)    
    for j in range(1,noEpochs+1):
        model.load_weights(str(j)+'.hdf5')
        outputs = model.predict(imgs)
        predText = decode(outputs)
        wordOK = 0
        wordTot = 0
        charDist = 0
        charTot = 0
        for i in range(len(predText)):
            #print(predText[i],trueText[i])
            wordOK += 1 if predText[i]==trueText[i] else 0
            wordTot += 1
            dist = editdistance.eval(predText[i],trueText[i])
            charDist += dist
            charTot += len(trueText[i])
        charDist = charDist/charTot
        accs[j-1] = wordOK/wordTot
        if j == noEpochs:
            print('Character Distance(CER):'+str(charDist*100))
            print('Word Accuracy:'+str(accs[-1]*100))
    plt.figure()
    plt.plot(range(1,noEpochs+1), accs,label=' ValidationAcc')
    plt.xlabel('Epochs')
    plt.legend()
    plt.xlim([0,noEpochs+1])
    
        

def validate(loader,transAcc = False):
    validation = loader.valSet
    imgs = []
    trueText = []
    for (i,path) in validation:
        trueText.append(i)
        img = cv2.imread(path,0)
        imgs.append(preprocess(img,(width,height),False))
    model = getModel(False)
    if transAcc:
        model.load_weights('D:/OCR/Documents/[8]WordsSKCor/transModel.h5')
    else:
        model.load_weights('D:/OCR/NewDataset/synthModel.h5')
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
    
    
def predict(imgPath):
    img2 = cv2.imread(imgPath,0)
    img2= preprocess(img2,(width,height),False)
    img2 = np.reshape(img2,(1,img2.shape[0],img2.shape[1],1))
    model2 = getModel(False)
    model2.load_weights('synthModel.h5')
    out = model2.predict(img2)
    pred = decode(out)
    print('Recognized Word: '+str(pred))


#loader = DataGenerator(batchSize,valBatchSize,width,height,valSplit)
#model,h = train(loader)
#validate(loader)

os.chdir('D:/OCR/Documents/[8]WordsSKCor')
wordFile = 'realWords.txt'

#valSplit = 1
#loader2 = DataGenerator(batchSize,valBatchSize,width,height,valSplit)
#validate(loader2)

noEpochs = 20
trainSamplesPerEpoch = 1900
valTransSplit = 0.02
valSamplesPerEpoch = 40
valBatchSize = 32
loader2 = DataGenerator(batchSize,valBatchSize,width,height,valTransSplit)
model2,h2 = train(loader2,transLearn = True)
#plotAccChange(loader2,True)
