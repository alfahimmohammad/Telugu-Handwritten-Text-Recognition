# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 10:29:47 2020

@author: Kather
"""
#%%
from tkinter import *

root = Tk()

mylabel = Label(root, text="hello")
mylabel2 = Label(root,text="watcha doin")
mylabel.grid(row=0,column=1)
mylabel2.grid(row=1,column=0)

root.mainloop()
#%%
from tkinter import *

root = Tk()

def pressed():
    mylabel = Label(root, text="hello")
    mylabel.pack()

button = Button(root,text='press me',command=pressed)
button.pack()

root.mainloop()
#%%
from tkinter.filedialog import askopenfilename
root = Tk()
def select_file():
    filename = askopenfilename()
    label = Label(root,text=filename)
    label.pack()
    
button = Button(root,text='select file',command=select_file)
button.pack()

root.mainloop()