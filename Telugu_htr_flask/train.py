# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 13:02:34 2020

@author: Kather
"""
#works only with tensorflow==1.13.1 and keras==2.2.4 for compatibilty with flask
import os
import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation, Flatten, LeakyReLU
from keras.layers import Reshape, Lambda, BatchNormalization
from keras.layers import Input, Conv2D, MaxPool2D,BatchNormalization, Bidirectional, LSTM, Dense, Lambda
from keras.layers.merge import average, concatenate
from keras.models import Model
from keras.layers.recurrent import LSTM
from keras.optimizers import Adadelta
import keras.callbacks
from keras.callbacks import ModelCheckpoint
import random
import cv2
import editdistance
import itertools
import matplotlib.pyplot as plt
import pickle

#os.chdir('/content/drive/My Drive/Colab Notebooks/CRNN')
batchSize = 64
valBatchSize = 20
width = 128
height = 32
valSplit = 0.02
trainSamplesPerEpoch = 25000
valSamplesPerEpoch = 1000
noEpochs = 50
wordFile = 'Newtrainnval.txt'
unicodes = list(np.load('unicodes.npy',allow_pickle=True))

with open('Newtrainnval_data.pickle','rb') as f:
  im = pickle.load(f)

dict_trainnval = {}
infile = open('/content/drive/My Drive/Colab Notebooks/CRNN/Newtrainnval.txt','r',encoding='utf-8')
integer = 0
for line in infile:
  integer += 1
  lineSplit = line.strip().split(' ')
  dict_trainnval[lineSplit[0]] = integer

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

def Preprocess(imag, imgSize=(128,32)):
    imag = cv2.medianBlur(imag,3)
    J = cv2.resize(imag,(128,32))
    rt3,_ = cv2.threshold(J,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    N,_ = np.histogram(J,[i for i in range(0,256)])
    f = np.argmax(N[int(rt3):]) #distance from threshold to max value greater than threshold
    g = np.argmax(N[:int(rt3)]) #distance from max value lesser than
    gg = int(rt3)-g             # threshold to to threshold
    """
    if blacks are more than whites, it should reduce them by kk amount,
    else increase them by kk amount
    """
    kk = ((np.sum(N[int(rt3):])-np.sum(N[:int(rt3)]))/np.sum(N)) 
    rt4 = int(rt3+kk*f) if kk>0 else int(rt3+kk*gg)
    _,k = cv2.threshold(J,rt4,255,cv2.THRESH_BINARY) 
    (wt, ht) = imgSize
    h = imag.shape[0]
    w = imag.shape[1]
    fx = w / wt #4
    fy = h / ht #3
    f = max(fx, fy)
    newSize = (max(min(wt, int(w / f)), 1), max(min(ht, int(h / f)), 1)) # scale according to f (result at least 1 and at most wt or ht)
    img = cv2.resize(k, newSize)#, cv2.INTER_AREA
    n,_ = np.histogram(img,[i for i in range(257)])
    nn = n[1:255]
    for j in range(len(nn)):
        if (np.sum(nn[:j])-np.sum(nn[j:]))/100 >= 0.25*np.sum(nn):
            break
    rt5 = j+1
    _,ii = cv2.threshold(img,rt5,255,cv2.THRESH_BINARY)
    H,W = ii.shape
    l,b = 32 - H, 128 - W
    if l%2 == 1:
        top, bottom = int(l/2) + 1, int(l/2)
    else:
        top, bottom = int(l/2), int(l/2)
    
    if b%2 == 1:
        left, right = int(b/2) + 1, int(b/2)
    else:
        left, right = int(b/2), int(b/2)
    FinalImage = cv2.copyMakeBorder(ii, top, bottom, left, right, cv2.BORDER_CONSTANT,255,255)
    (m, s) = cv2.meanStdDev(FinalImage)
    m = m[0][0]
    s = s[0][0]
    img = FinalImage - m
    img = img / s if s>0 else img
    return np.reshape(img,(32,128))

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
            fileName = lineSplit[0] # '/content/drive/My Drive/Colab Notebooks/CRNN/' +
            text = self.truncateLabel(' '.join(lineSplit[1:]))
            
            for ch in text:
                if ch not in unicodes:
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
            #img = cv2.imread(batch[i][1],0)
            kk = dict_trainnval[batch[i][1]]
            imgs.append(np.reshape(im[:,:,kk-1],(32,128,1)))
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

def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal')

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
            #x = LeakyReLU()(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            #x = LeakyReLU()(x)
            x = Activation(activation)(x)
        x = conv(x)
    return x

def res_block(inputs,
              num_filters=16,
              kernel_size=3,
              strides=1,
              activation='relu',
              batch_normalization=True,
              conv_first=True,
              BN=True,
              A=True):
    x = inputs
    y = resnet_layer(inputs=x,
                    num_filters=num_filters,
                    strides=strides)
    y = resnet_layer(inputs=y,
                    num_filters=num_filters,
                    activation=None)
    x = resnet_layer(inputs=x,
                    num_filters=num_filters,
                    strides=strides,
                    activation=None,
                    batch_normalization=False)
    x = keras.layers.add([x, y])
    if BN:
      x = BatchNormalization()(x)
    if A:
      x = Activation(activation)(x)
      #x = LeakyReLU()(x)
    return x

def getModel(training):
    inputShape = (32,128,1)
    rnnUnits = 256
    maxStringLen = 32
    inputs = Input(name = 'inputX', shape = inputShape, dtype = 'float32')
    inner = res_block(inputs,64)
    inner = res_block(inner,64)
    inner = MaxPooling2D(pool_size = (2,2),name = 'MaxPoolName1')(inner)
    inner = res_block(inner,128)
    inner = res_block(inner,128)
    inner = MaxPooling2D(pool_size = (2,2),name = 'MaxPoolName2')(inner)
    inner = res_block(inner,256)
    inner = res_block(inner,256)
    inner = MaxPooling2D(pool_size = (1,2),strides = (2,2), name = 'MaxPoolName4')(inner)
    inner = res_block(inner,512)
    inner = res_block(inner,512)
    inner = MaxPooling2D(pool_size = (1,2), strides = (2,2), name = 'MaxPoolName6')(inner)
    inner = res_block(inner,512)

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
    yPred = Activation('softmax')(yPred)
    #Model(inputs = inputs,outputs = yPred).summary()

    labels = Input(name='label', shape=[32], dtype='float32')
    inputLength = Input(name='inputLen', shape=[1], dtype='int64')     # (None, 1)
    labelLength = Input(name='labelLen', shape=[1], dtype='int64')
    
    lossOut = Lambda(ctcLambdaFunc, output_shape=(1,), name='ctc')([yPred, labels, inputLength, labelLength])
    
    if training:
        return Model(inputs = [inputs, labels, inputLength, labelLength], outputs=[lossOut,yPred])
    return Model(inputs=[inputs], outputs=yPred)

def decode(yPred):#Best Path or Beam Search
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
          word+= ""
        else:
          word += unicodes[pred[i]]
      texts.append(word)
    return texts
    
def train(loader,transLearn = False):
    model = getModel(True)
    if transLearn:
        model.load_weights('/content/drive/My Drive/Colab Notebooks/CRNN/withoutDil/model_89.hdf5')
    model.compile(loss={'ctc': lambda yTrue,yPred: yPred},optimizer = Adadelta())
    
    checkpoint = ModelCheckpoint('/content/drive/My Drive/Colab Notebooks/CRNN/WDaugmented/augmodel-{epoch:03d}-{val_loss:03f}.hdf5',verbose=1, monitor='val_loss',save_best_only=False, mode='auto')    
    test_func = K.function([model.inputs[0]],[model.outputs[1]])
    viz_cb = VizCallback(test_func, loader.nextVal())      

    h = model.fit_generator(generator=loader.nextTrain(),
                            workers=1,use_multiprocessing=False,
                    steps_per_epoch= trainSamplesPerEpoch // batchSize,
                    epochs=noEpochs,
                    callbacks=[viz_cb,loader, checkpoint],
                    validation_data = loader.nextVal(),
                    validation_steps = valSamplesPerEpoch // valBatchSize
                    )

    if transLearn:
        model.save('/content/drive/My Drive/Colab Notebooks/CRNN/Model/TransModel.h5') #transModel.h5
    else:
        model.save('/content/drive/My Drive/Shared/CRNN/New_Models/latestPaperModel.h5') #synthModel.h5
    
    plotHistory(h)
    return model,h

def plotAccChange(loader,transLearn = False):
    if transLearn:
        os.chdir('/content/drive/My Drive/Colab Notebooks/CRNN/Models')
    else:
        os.chdir('/content/drive/My Drive/Colab Notebooks/CRNN/New_Models')
    accs = np.zeros((noEpochs,1))
    model = getModel(False)
    validation = loader.valSet
    imgs = []
    trueText = []
    for (i,path) in validation:
        trueText.append(i)
        #img = cv2.imread(path,0)
        jj = dict_trainnval[path]
        imgs.append(np.reshape(im[:,:,jj-1],(32,128,1)))
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
        #img = cv2.imread(path,0)
        nn = dict_trainnval[path]
        imgs.append(np.reshape(im[:,:,nn-1],(32,128,1)))
        #imgs.append(preprocess(img,(width,height),0))
    model = getModel(False)
    """
    if transAcc:
        model.load_weights('/content/drive/My Drive/Colab Notebooks/CRNN/Models/transModel.h5')
    else:
        model.load_weights('/content/drive/My Drive/Colab Notebooks/CRNN/New_Models/paperModel2.h5')
    """
    model.load_weights('WDaugmented/model_27.hdf5')
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
    img2= Preprocess(img2,(128,32),0)
    img2 = np.reshape(img2,(1,img2.shape[0],img2.shape[1],1))
    model2 = getModel(False)
    model2.load_weights('synthModel.h5')
    out = model2.predict(img2)
    pred = decode(out)
    print('Recognized Word: '+str(pred))


loader = DataGenerator(batchSize,valBatchSize,width,height,valSplit)
model,h = train(loader,transLearn = True)
#validate(loader)

#os.chdir('D:/OCR/Documents/[8]WordsSKCor')
#wordFile = 'realWords.txt'

#valSplit = 1
#loader2 = DataGenerator(batchSize,valBatchSize,width,height,valSplit)
#validate(loader2)

#noEpochs = 20
#trainSamplesPerEpoch = 1900
#valTransSplit = 0.02
#valSamplesPerEpoch = 40
#valBatchSize = 32
#loader2 = DataGenerator(batchSize,valBatchSize,width,height,valTransSplit)
#model2,h2 = train(loader2,transLearn = True)
#plotAccChange(loader2,True)