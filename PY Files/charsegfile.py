import cv2 as cv
import numpy as np
import os
import CharSegmentation

try:
    cseg=CharSegmentation.Model('C:/Users/akroc/Desktop/ocr-handwriting-models/gap-clas/RNN/Bi-RNN-new', 'prediction')
    print("Successfully loaded.")
except:
    print("Couldn't Load the model!")

path=os.getcwd()+'/Desktop/Dataset'

filename= os.listdir(path+'/NewWords')

for name in filename:
    file=path+'/NewWords/'+name
    print(file)
    newwrd =cv.imread(file)
    CharSegmentation.Cycler(newwrd,1,cseg,f"{name[4:-4]}")
