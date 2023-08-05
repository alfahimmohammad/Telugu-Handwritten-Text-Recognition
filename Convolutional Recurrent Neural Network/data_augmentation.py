# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 18:28:05 2020

@author: Kather
"""
#%%
import numpy as np
import cv2
import matplotlib.pyplot as plt
#from IPython.core.interactiveshell import InteractiveShell
#InteractiveShell.ast_node_interactivity = "all"
import os
from skimage.util import random_noise
from collections import Counter
import random
import pickle

os.chdir('C:/Users/Kather/Desktop/fahim/ML project/sent by surya/CRNN') 
#os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

def crop(image):
    H,W = image.shape
    _,img = cv2.threshold(image,0,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY)#cv2.THRESH_OTSU+
    vertical = np.sum(img,axis=1)
    p, q, r, s = 0, H, 0, W
    if np.amin(vertical) == 0 and np.argmin(vertical) < int(H/2):
        for m in range(np.argmin(vertical),len(vertical)):
            if vertical[m] != 0:
                break
        p = m+np.argmin(vertical)+1
    if np.amin(vertical[p:]) == 0 and np.argmin(vertical[p:]) > int(H/2):
        q = np.argmin(vertical[p:])-1
    horizontal = np.sum(img,axis=0)
    if np.amin(horizontal) == 0 and np.argmin(horizontal) < int(W/2):
        for n in range(np.argmin(horizontal),len(horizontal)):
            if horizontal[n] != 0:
                break
        r = n+np.argmin(horizontal)+1
    if np.amin(horizontal[r:]) == 0 and np.argmin(horizontal[r:]) > int(W/2):
        s = np.argmin(horizontal[r:])-1
    im = img[p:q,r:s]    
    kernel = np.ones((4,4),dtype='uint8')
    Img = cv2.dilate(im,kernel,iterations=1)
    h, w = Img.shape
    for i in range(w):
        array = Img[:,i]
        l = array.shape[0]
        if np.amin(array) == 0 and np.argmin(array) < 3*int(l/100) and i >= int(w/100):
            for ii in range(np.argmin(array)+1,l):
                if array[ii] != array[ii-1]:
                    #arr = array[ii:]
                    break
            if np.amin(array[ii:]) == 0 and np.argmin(array) < l - 3*int(l/100) and i >= int(w/100):
                a = max(i-1,0)
                break
        elif np.amin(array) == 0 and np.argmin(array) > 3*int(l/100) and np.argmin(array) < l-3*int(l/100) and i >= int(w/100):
            a = max(i-1,0)
            break  
    for j in range(w-1,-1,-1):
        jarray = Img[:,j]
        l = jarray.shape[0]
        if np.amin(jarray) == 0 and np.argmin(jarray) < 3*int(l/100) and j <= w-int(w/100):
            for jj in range(np.argmin(jarray)+1,l):
                if jarray[jj] != jarray[jj-1]:
                    #arr = array[jj:]
                    break
            if np.amin(jarray[jj:]) == 0 and np.argmin(jarray) < l - 3*int(l/100) and j <= w-int(w/100):
                b = j+2 if j>0 else w-1
                break
        elif np.amin(jarray) == 0 and np.argmin(jarray) > 3*int(l/100) and np.argmin(jarray) < l-3*int(l/100) and j <= w-int(w/100):
            b=j+2 if j>0 else w-1
            break           
    imag = Img[:,a:b]
    for k in range(h):
        vector = imag[k,:]
        if np.amin(vector) == 0 and k > 7*int(h/100):
            c = max(k-1,0)
            break    
    for kk in range(h-1,-1,-1):
        array = imag[kk,:]
        if np.amin(array) == 0 and kk < h-4*int(h/100):
            d = kk+1 if kk>0 else h-1
            break
            
    #if d > c and b > a and im[c:d,a:b].size and image[p+c:min(p+d,h),r+a:min(r+b,w)].size:
    #I = im[c:d,a:b]
    J = image[p+c:min(p+d,h),r+a:min(r+b,w)]
    #else:
     #   I = im
      #  J = image
    return J
"""
def preprocess(imag, imgSize, dataAugmentation=0,num):
    imag = cv2.medianBlur(imag,3)
    J = cv2.resize(imag,(128,32))
    rt3,_ = cv2.threshold(J,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    N,_ = np.histogram(J,[i for i in range(0,256)])
    f = np.argmax(N[int(rt3):])
    rt4 = int(rt3+0.7*f)
    _,k = cv2.threshold(J,rt4,255,cv2.THRESH_BINARY) 
    if dataAugmentation==1:
    	stretch = (random.random() - 0.5) # -0.5 .. +0.5
    	wStretched = max(int(k.shape[1] * (1 + stretch)), 1) # random width, but at least 1
    	k = cv2.resize(k, (wStretched, k.shape[0])) # stretch horizontally by factor 0.5 .. 1.5
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
    #rotatedImg = cv2.rotate(FinalImage,cv2.ROTATE_90_CLOCKWISE)   
    #rotatedImg = cv2.transpose(FinalImage)
    (m, s) = cv2.meanStdDev(FinalImage)
    m = m[0][0]
    s = s[0][0]
    img = FinalImage - m
    img = img / s if s>0 else img
    #
    return np.reshape(img,(32,128))#.T[::-1,:]
"""
def salt_pepper(img):
    noise = random_noise(img, mode = 's&p', amount = 0.05)
    a = Counter(noise.flatten())
    s = a.most_common(1)[0][0]
    noise[noise==s] = 1
    final = noise.copy()
    """
    (m,s) = cv2.meanStdDev(final)
    m = m[0][0]
    s = s[0][0]
    final = final - m
    final = final/s if s>0 else final
    final = final.astype('float32')
    #final -= np.mean(noise)
    #final /= np.std(noise)
    #final = final.astype('float32')
    """
    return final

def rotate_bound(image, Range):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # get the random angle
    angle = np.random.randint(-Range,Range+1,1)
    # grab the rotation matrix 
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    im = cv2.warpAffine(image, M, (nW, nH), borderValue = 255)
    im = cv2.resize(im, (128,32))
    #Changing from float32 to uint8
    img_n = cv2.normalize(src=im, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    #OTsu thresholding
    _,im = cv2.threshold(img_n,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    Final = im.copy()
    """
    (M,S) = cv2.meanStdDev(Final)
    M = M[0][0]
    S = S[0][0]
    Final = Final - M
    Final = Final/S if S>0 else Final
    Final.astype('float32')
    #Final -= np.mean(im)
    #Final /= np.std(im)
    """
    return Final

def Preprocess(imag,  numb, imgSize=(128,32), dataAugmentation=0):
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
    if dataAugmentation==1:
    	stretch = (random.random() - 0.5) # -0.5 .. +0.5
    	wStretched = max(int(k.shape[1] * (1 + stretch)), 1) # random width, but at least 1
    	k = cv2.resize(k, (wStretched, k.shape[0])) # stretch horizontally by factor 0.5 .. 1.5
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
    if numb == 0:
        FinalImage = rotate_bound(FinalImage,5)
    elif numb == 1:
        FinalImage = salt_pepper(FinalImage)
    else:
        FinalImage = rotate_bound(FinalImage,5)
        FinalImage = salt_pepper(FinalImage)
    (m, s) = cv2.meanStdDev(FinalImage)
    m = m[0][0]
    s = s[0][0]
    img = FinalImage - m
    img = img / s if s>0 else img
    return np.reshape(img,(32,128))

infile = open('Newtrainnval.txt','r',encoding="utf-8")
Line = infile.readlines()
infile.close()
im = np.zeros((32,128,len(Line)))

for i in range(0,len(Line)):
    num = np.random.randint(0,4,1)
    lineSplit = Line[i].strip().split(' ')
    imgpath = lineSplit[0]
    I = cv2.imread(imgpath,0)
    Ic = crop(I)
    Ic = Preprocess(Ic,numb=num)
    im[:,:,i] = Ic
    print(i+1)

print('over')
#%%
with open('Newaugmented_data.pickle','wb') as f:
    pickle.dump(im,f)

print('over')

#%%
import pickle
with open('augmented_data.pickle','rb') as f:
    Im = pickle.load(f)