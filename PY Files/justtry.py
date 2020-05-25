import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from normalization import letter_normalization
from tensorflow.python.ops.rnn_cell_impl import LSTMCell, ResidualWrapper, DropoutWrapper, MultiRNNCell


class Model():
    """Loading and running isolated tf graph."""
    def __init__(self, loc, operation='activation', input_name='x'):
        """
        loc: location of file containing saved model
        operation: name of operation for running the model
        input_name: name of input placeholder
        """
        self.input = input_name + ":0"
        self.graph = tf.Graph()
        self.sess = tf.compat.v1.Session(graph=self.graph)
        with self.graph.as_default():
            saver = tf.compat.v1.train.import_meta_graph(loc + '.meta', clear_devices=True)
            saver.restore(self.sess, loc)
            self.op = self.graph.get_operation_by_name(operation).outputs[0]

    def run(self, data):
        """Run the specified operation on given data."""
        return self.sess.run(self.op, feed_dict={self.input: data})
    
    def eval_feed(self, feed):
        """Run the specified operation with given feed."""
        return self.sess.run(self.op, feed_dict=feed)
    
    def run_op(self, op, feed, output=True):
        """Run given operation with the feed."""
        if output:
            return self.sess.run(
                self.graph.get_operation_by_name(op).outputs[0],
                feed_dict=feed)
        else:
            self.sess.run(
                self.graph.get_operation_by_name(op),
                feed_dict=feed)
        
    
    
def _create_single_cell(cell_fn, num_units, is_residual=False, is_dropout=False, keep_prob=None):
    """Create single RNN cell based on cell_fn."""
    cell = cell_fn(num_units)
    if is_dropout:
        cell = DropoutWrapper(cell, input_keep_prob=keep_prob)
    if is_residual:
        cell = ResidualWrapper(cell)
    return cell


def create_cell(num_units, num_layers, num_residual_layers, is_dropout=False, keep_prob=None, cell_fn=LSTMCell):
    """Create corresponding number of RNN cells with given wrappers."""
    cell_list = []
    
    for i in range(num_layers):
        cell_list.append(_create_single_cell(
            cell_fn=cell_fn,
            num_units=num_units,
            is_residual=(i >= num_layers - num_residual_layers),
            is_dropout=is_dropout,
            keep_prob=keep_prob
        ))

    if num_layers == 1:
        return cell_list[0]
    return MultiRNNCell(cell_list)








MODEL_LOC_CHARS = 'C:/Users/akroc/Desktop/ocr-handwriting-models/char-clas/en/CharClassifier'

CHARACTER_MODEL = Model(MODEL_LOC_CHARS)

CHARS = ['', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
         'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
         'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c',
         'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
         'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
         'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6',
         '7', '8', '9', '.', '-', '+', "'"]



CHAR_SIZE = len(CHARS)

idxs = [i for i in range(len(CHARS))]
idx_2_chars = dict(zip(idxs, CHARS))
chars_2_idx = dict(zip(CHARS, idxs))

def char2idx(c, sequence=False):
    if sequence:
        return chars_2_idx[c] + 1
    return chars_2_idx[c]

def idx2char(idx, sequence=False):
    if sequence:
        return idx_2_chars[idx-1]
    return idx_2_chars[idx]
    

def recognise(img,counter):
    
    chars=[]
    char, dim = letter_normalization(img, is_thresh=True, dim=True)
    # TODO Test different values
    if dim[0] > 4 and dim[1] > 4:
        chars.append(char.flatten())

    chars = np.array(chars)
    word = ''        
    if len(chars) != 0:
        pred = CHARACTER_MODEL.run(chars)                
        for c in pred:
            cv2.imwrite(f"C:/Users/akroc/Desktop/LabeledChar/{CHARS[c]}_{counter}.png",char)
            # print(CHARS[c])
            

import os
path=os.getcwd()+'/Desktop/Dataset'
filename= os.listdir(path+'/Char')
counter=1
for name in filename:
    file=path+'/Char/'+name
    print(file)
    img=cv2.imread(file)
    img = cv2.bilateralFilter(img, 10, 30, 30)
    gray = 255 - cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    ret,th = cv2.threshold(norm, 50, 255, cv2.THRESH_TOZERO)
    # cv2.namedWindow("dd",0)
    # cv2.imshow("dd",th)
    # cv2.waitKey(0)
    recognise(th,counter)
    counter+=1