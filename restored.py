import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

one_hot_length = 60000


def get_ohvector(index):
    """
    :param index: the index of the image in csv file
    :return: a one hot vector that indicates position of image in csv fie
    """
    vector = [0] * one_hot_length
    vector[index] = 1
    vector = np.array(vector).reshape(1, one_hot_length)
    return vector

tf.reset_default_graph()
#initializers for weights and biases.
kern_init = tf.random_normal_initializer(mean = 0.0, stddev = 0.01)
bias_init = tf.zeros_initializer()

with tf.variable_scope("inputs"):
    y_true = tf.placeholder(tf.float32,shape = [1,28,28,1],name = "y_true")
    x = tf.placeholder(tf.float32, shape = [1,one_hot_length], name = "one_hot_vector")

#onehot will essentially select a noise vector of the corresponding image.
with tf.variable_scope("noise_selector"):
    noise_vector = tf.get_variable("noise_1d_vector",shape = [one_hot_length,64])
    noise_selector = tf.matmul(x,noise_vector)
    noise_mass =tf.reshape(noise_selector,shape = [1,4,4,4],name = "reshaper")

tf.summary.histogram("noise_vector1d",noise_vector)

#convolution transpose layers start from here.
ls_layer_units = [64, 32, 16, 8, 4, 1]
ls_layer_names = ["glayer" + str(i + 1) for i in range(6)]
layer = noise_mass
for i in range(len(ls_layer_units)):
    with tf.variable_scope(ls_layer_names[i]):
        layer = tf.layers.conv2d_transpose(layer, ls_layer_units[i], [5, 5],
                                       strides=[1, 1], padding='valid',
                                       activation=None,
                                       use_bias=False,
                                       kernel_initializer=kern_init,
                                       trainable=True,name = "conv_transpose")
        layer = tf.contrib.layers.batch_norm(inputs=layer, center=True, scale=True, is_training=True,
                                             scope = "batch_norm")
        layer = tf.nn.relu(layer,name = "relu")

gen_image = tf.get_default_graph().get_tensor_by_name("glayer6/relu:0")

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess,"checkpoint/glo")
    print("model restored")
    sample_index = 64
    sample_img = sess.run(gen_image,feed_dict={x: get_ohvector(sample_index)})
    sample_img = sample_img.reshape(28,28)
    plt.imshow(sample_img)
    plt.show()