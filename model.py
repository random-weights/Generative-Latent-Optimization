import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

X_TRAIN_FNAME = "data/x_train.csv"
one_hot_length = 60000

def image():
    with open(X_TRAIN_FNAME,'r') as fh:
        line = fh.readline().strip("\n")
        pixels = line.split(",")
        norm_pixels = [int(pixel)/255.0 for pixel in pixels]
        yield norm_pixels



kern_init = tf.random_normal_initializer(mean = 0.0, stddev = 0.01)
bias_init = tf.zeros_initializer()

with tf.variable_scope("inputs"):
    y_true = tf.placeholder(tf.float32,shape = [1,28,28,1],name = "y_true")
    x = tf.placeholder(tf.float32, shape = [1,one_hot_length], name = "one_hot_vector")
with tf.variable_scope("noise_selector"):
    noise_vector = tf.get_variable("noise_1d_vector",shape = [one_hot_length,64])
    noise_selector = tf.matmul(x,noise_vector)
    noise_mass =tf.reshape(noise_selector,shape = [4,4,4],name = "reshaper")


