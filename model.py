import tensorflow as tf
import numpy as np

X_TRAIN_FNAME = "data/x_train.csv"
one_hot_length = 60000

def image_array(file_handler):
    """
    get the next image in csv file.
    :return: an image as numpy array with dims (1,28,28,1)
    """
    while True:
        line = file_handler.readline().strip("\n")
        pixels = line.split(",")
        norm_pixels = [int(pixel)/255.0 for pixel in pixels]
        norm_pixels = np.array(norm_pixels).astype(np.float32).reshape(1,28,28,1)
        yield norm_pixels

def get_ohvector(index):
    """
    :param index: the index of the image in csv file
    :return: a one hot vector that indicates position of image in csv fie
    """
    vector = [0]*one_hot_length
    vector[index] = 1
    vector = np.array(vector).reshape(1,one_hot_length)
    return vector


gph = tf.Graph()
with gph.as_default():

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

    gen_image = gph.get_tensor_by_name("glayer6/relu:0")
    cost = tf.reduce_mean((gen_image - y_true)**2,name = "cost")
    tf.summary.scalar("cost",cost)

    train = tf.train.AdamOptimizer(1e-2).minimize(cost,name = "optimizer")

    saver = tf.train.Saver()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("checkpoint",graph=tf.get_default_graph())

with tf.Session(graph = gph) as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    for image_index in range(2):
        print("\rimage: ".format(image_index)+str(image_index),end = "")

        fh = open(X_TRAIN_FNAME,'r')
        image = image_array(fh)

        feed_dict = {x: get_ohvector(image_index),
                     y_true: next(image)}
        _,summary = sess.run([train,merged],feed_dict)

        if (image_index+1)%100 == 0:
            writer.add_summary(summary,image_index)

    saver.save(sess,"checkpoint/glo")
    print("Training done")