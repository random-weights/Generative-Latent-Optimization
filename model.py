import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

one_hot_length = 60000

class Data():
    def __init__(self,train_fname):
        self.X_TRAIN_FNAME = train_fname

    def index_to_oh(self,index):
        """
        :param index: the index of the image in csv file
        :return: a one hot vector that indicates position of image in csv fie
        """
        vector = [0]*one_hot_length
        vector[index] = 1
        vector = np.array(vector).reshape(1,one_hot_length)
        return vector

    def image(self):
        """
        get the next image in csv file.
        :return: an image as numpy array with dims (1,28,28,1)
        """
        fh = open(self.X_TRAIN_FNAME,'r')
        for line in fh:
            line = line.strip("\n")
            pixels = line.split(",")
            norm_pixels = [int(pixel)/255.0 for pixel in pixels]
            norm_pixels = np.array(norm_pixels).astype(np.float32).reshape(1,28,28,1)
            yield norm_pixels

    def onehot(self):
        """
        get the next onehot array
        :return: onehot array od dims [1,one_hot_length]
        """
        indices = range(one_hot_length)
        for index in indices:
            yield self.index_to_oh(index)

    def image_batch(self,batch_size):
        """
        Uses the image generator.
        :param batch_size: recommended in power 2
        :return: a numpy array that contains <batch_size> images
        """
        image_gen = itertools.cycle(self.image())
        while True:
            image_arr = []
            for _ in range(batch_size):
                image_arr.append(next(image_gen))
            yield np.array(image_arr).reshape(batch_size,28,28,1)

    def onehot_batch(self,batch_size):
        """
        Uses onehot() generator.
        :param batch_size: recommended in power of 2
        :return: a numpy array of <batch_size> one_hot arrays.
        """
        oh_array = itertools.cycle(self.onehot())
        while True:
            onehot_batch = []
            for _ in range(batch_size):
                onehot_batch.append(next(oh_array))
            yield np.array(onehot_batch).reshape(batch_size,one_hot_length)

class Laplacian():

    def __init__(self,x_dim,sigma):
        self.g_filter = self.get_2d_gaussian(x_dim,sigma)

    def get_2d_gaussian(self,x_dim,sigma):
        """
        :param x_dim: square dimension of gaussian filter
        :param sigma: the standard deviation of gaussian(applied in all directions)
        :return: a 2d numpy gaussian filter
        """
        def gaussian(x,y):
            """
            x and y indices of gaussian filters should be zero centered.
            Example: for a 3x3 filter, x_dims are -1,0,1 and y_dims are -1,0,1
            :param x: x index of cell of gaussian filter
            :param y: y index of cell of gaussian filter
            :return: float that is defined by gaussian function
            """
            denom = np.sqrt((2*np.pi*(sigma**2)))
            sos = (x**2) + (y**2)
            num = np.exp(-sos/(2*(sigma**2)))
            return  num/denom

        if x_dim%2 == 0:
            raise ValueError
        else:
            start = -int(x_dim/2)
            end = int(x_dim/2) + 1
            index_offset = int(x_dim/2)
            filter = np.zeros(shape = [x_dim,x_dim])
            for x in range(start,end):
                for y in range(start,end):
                    x_index = x + index_offset
                    y_index = y + index_offset
                    filter[x_index][y_index] = gaussian(x,y)
            return filter

    def laplace_pyramid(self,image):
        """
        :param image: image for which pyramid is needed. Input shape must be 1,28,28,1
        If using any other size, change the no. of iterations as described below.

        yields a level of pyramid. Since this is dependent on the image size,
        function has to be modified for images other than MNIST.

        What changes: change the number of for loop iterations that depends on the image dims.
        """

        filt = self.g_filter
        with tf.variable_scope("gaussianKernel"):
            gkernel = tf.convert_to_tensor(filt,dtype = tf.float32)
            kern_shape = tf.shape(gkernel)[0]
            gkernel = tf.reshape(gkernel,shape = [kern_shape,kern_shape,1,1],name = "gkernel")

        sub_filter = tf.constant(1.0,tf.float32,shape = [1,1,1,1],name = "subsampling_filter")

        with tf.variable_scope("LaplacePyramid"):
            g0 = image
            for i in range(3):
                g0_shape = tf.shape(g0)
                g1 = tf.nn.conv2d(input=g0, filter=gkernel, strides=[1, 1, 1, 1], padding="VALID")
                g1_extended = tf.nn.conv2d_transpose(value=g1, filter=gkernel, output_shape=g0_shape, strides=[1, 1, 1, 1],
                                                     padding="VALID")
                yield tf.subtract(g0, g1_extended)
                g1_subsampled = tf.nn.conv2d(input=g1, filter=sub_filter, strides=[1, 2, 2, 1], padding="VALID")
                g0 = g1_subsampled

    def laplace_loss(self,img1,img2):
        """
        as described in https://arxiv.org/abs/1707.05776
        :param img1: first image(generally the original image)
        :param img2: second image(generally the image generated from noise
        :return: Given two images: img1 and img2, it computes the laplacian pyramid loss
        """
        loss = 0
        lp1 = self.laplace_pyramid(img1)
        lp2 = self.laplace_pyramid(img2)
        for i in range(3):
            l1 = next(lp1)
            l2 = next(lp2)
            loss += tf.reduce_mean(l1-l2)/4**(i+1)
        return loss

lpl = Laplacian()

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

    #tf.summary.histogram("noise_vector1d",noise_vector)

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

    cost = lpl.laplace_loss(y_true,gen_image)
    #tf.summary.scalar("cost",cost)

    train = tf.train.AdamOptimizer(1e-2).minimize(cost,name = "optimizer")

    #saver = tf.train.Saver()
    merged = tf.summary.merge_all()
    #writer = tf.summary.FileWriter("checkpoint",graph=tf.get_default_graph())

epochs = 1000

with tf.Session(graph = gph) as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    train_data = Data("data/x_train.csv")
    img_batch_gen = train_data.image_batch(batch_size = 32)
    oh_batch_gen = train_data.onehot_batch(batch_size=32)
    for epoch in range(epochs):
        print("\rEpoch: ".format(epoch)+str(epoch),end = "")

        feed_dict = {x: next(oh_batch_gen),
                     y_true: next(img_batch_gen)}

        _,summary = sess.run([train,merged],feed_dict)

        if (epoch+1)%100 == 0:
            #writer.add_summary(summary,epoch)
            pass

    #saver.save(sess,"checkpoint/glo")
    print("Training done")