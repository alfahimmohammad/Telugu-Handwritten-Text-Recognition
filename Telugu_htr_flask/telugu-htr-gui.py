# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 11:17:32 2020

@author: Kather
"""
#%%
from tkinter import *
from tkinter.filedialog import askopenfilename, asksaveasfile
import cv2 #install
from skimage.transform import rotate#, hough_line, hough_line_peaks #install
#from skimage.feature import canny
from deskew import determine_skew #install
import numpy as np #install
import keras.layers #install keras==2.2.4 and tensorflow==1.13.1
from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda, BatchNormalization
from keras.layers.merge import average, concatenate
from keras.models import Model
from keras.layers.recurrent import LSTM
#import itertools
#import tensorflow as tf

width = 128
height = 32
unicodes = list(np.load('unicodes.npy',allow_pickle=True))

def predict(img2,model):
    img2= Preprocess(img2)
    img2 = np.reshape(img2,(1,img2.shape[0],img2.shape[1],1))
    out = model.predict(img2)
    pred = decode(out)
    for word in pred:
        return str(word)
    
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
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
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
    return x

def getModel():
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
    return Model(inputs=[inputs], outputs=yPred)

def labelsToText(labels):
    ret = []
    for c in labels:
        if c == len(unicodes):
            ret.append("")
        else:
            ret.append(unicodes[c])
    return "".join(ret)

def decode(yPred):  #Best Path, now beam search=3
    texts = []
    """
    for y in yPred:
        label = list(np.argmax(y[2:],1))
        label = [k for k, g in itertools.groupby(label)]
        text = labelsToText(label)
        texts.append(text)
    return texts
    """
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
      texts.append(word)
    return texts

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

model2 = getModel()
model2.load_weights('ResNetBestNew_weights.h5') #change model here

def htr(filepath):
    image = cv2.imread(filepath,0) #change filename
    rows,cols = image.shape
    kernel = np.ones((9,9),np.uint8)
    erode = cv2.erode(image,kernel,iterations = 1)
    angle = determine_skew(erode)
    img = rotate(image, angle, resize=True) * 255
    img = np.uint8(img)
    print('\ngot image')
    # mser properties
    _delta=5
    _min_area=60
    _max_area=14400
    _max_variation=0.25
    _min_diversity=.2
    _max_evolution=200
    _area_threshold=1.01
    _min_margin=0.003
    _edge_blur_size=5
    
    mser = cv2.MSER_create(_delta,_min_area,_max_area,_max_variation,_min_diversity,_max_evolution,_area_threshold,_min_margin,_edge_blur_size)
    
    regions, boundingBoxes = mser.detectRegions(img)
    
    out_image_2 = np.zeros(img.shape,dtype='uint8')
    regions2 = []
    area_regions = []
    for region in regions:
        region = np.asarray(region)
        min1 = np.amin(region[:,0])
        max1 = np.amax(region[:,0])
        min2 = np.amin(region[:,1])
        max2 = np.amax(region[:,1])
        if max1 != min1 and max2 != min2:
            e = float(max2 - min2)/float(max1 - min1)
            ac = float(len(region))/((max2 - min2)*(max1 - min1))
            if e>0.1 and e<10 and ac>0.2:
                regions2.append(region)
                area_regions.append((max2 - min2)*(max1 - min1))
                out_image_2[ region[:,1] , region[:,0] ] = 255
    
    area_regions = np.asarray(area_regions)
    
    regions = regions2
    
    n, bins = np.histogram(area_regions,bins="auto")
    
    avg = 0
    num = 0
    
    a, b = bins[np.argmax(n)], bins[np.argmax(n)+1]
    for i in range(len(area_regions)):
        if area_regions[i]>a and area_regions[i]<b:
            avg += area_regions[i]
            num += 1
    avg = avg/float(num)
    
    kernell = np.ones((1,int(0.7*np.sqrt(avg))),np.uint8)
    appx_size = int(0.7*np.sqrt(avg))
    out_image_3 = cv2.dilate(out_image_2,kernell,iterations=1)
    kernel2 = np.ones((int(0.2*np.sqrt(avg)),1),np.uint8)
    out_image_4 = cv2.dilate(out_image_3,kernel2,iterations=1)
    
    cnts, _ = cv2.findContours(out_image_4.astype(np.uint8).copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    regions1 = []
    
    for i in range(len(cnts)):
        x,y,w,h = cv2.boundingRect(cnts[i])
        
        include = True
        
        for j in range(len(cnts)):
            if j!= i:
                x1,y1,w1,h1 = cv2.boundingRect(cnts[j])
                if x>=x1 and y>=y1 and x+w<=x1+w1 and y+h<=y1+h1:
                    include = False
    
        if (h>2*appx_size or w>2*appx_size or w*h>100) and include:
            regions1.append([x,y,w,h])
            
    regions1 = np.array(regions1)
    area = regions1[:,2] * regions1[:,3]
    area = np.array(sorted(area))/(rows*cols)
    
    regions2 = [[] for i in range(len(regions1))]
    regions2[0].append(regions1[0])
    line_idx = 0
    
    for i in range(1,len(regions1)):
        x,y,w,h = regions1[i]
        xa,ya,wa,ha = regions1[i-1]
        a = max(y,ya)
        b = min(h+y,ha+ya)
        if(b-a)>0:
            regions2[line_idx].append(regions1[i])
        else:
            line_idx = line_idx + 1
            regions2[line_idx].append(regions1[i]) 
    regions2 = np.array(regions2)
    regions2 = [x for x in regions2 if x != []]
    
    regions3 = []
    for i in range(len(regions2)-1,-1,-1):
        array = np.array(regions2[i])
        g = np.argsort(array[:,0])
        lin = array[g,:]
        regions3.append(lin)
    
    content = u''
    for line in regions3:
        LineString = ''
        for i in range(len(line[:,0])):
            x,y,w,h = line[i,:]
            w = img[y:y+h,x:x+w]
            Word = predict(w,model2)
            LineString += Word + '  '
        LineString += '\n'
        content += LineString
    
    return content

root = Tk()
def save_as(text):
    f = asksaveasfile(mode="w",defaultextension=".txt")
    if f is None:
        return
    f.write(text)
    f.close()

def select_file():
    filename = askopenfilename()
    txt = htr(filename)
    #label.pack_forget()
    label = Text(root)
    label.insert(END,txt)
    label.pack()
    #button2 = Button(root,text='save text',command= lambda: save_as(txt)).pack()

button = Button(root,text='open image',command=select_file)
button.pack()
label = Label(root,text="telugu text will be printed here").pack()

root.mainloop()
#%%
