import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

X_TRAIN_FNAME = "data/x_train.csv"

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

def get_2d_gaussian(x_dim,sigma):
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

def laplace_pyramid(image):
    filt = get_2d_gaussian(3, 1.0)
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

def laplace_loss(img1,img2):
    loss = 0
    lp1 = laplace_pyramid(img1)
    lp2 = laplace_pyramid(img2)
    for i in range(3):
        l1 = next(lp1)
        l2 = next(lp2)
        loss += tf.reduce_mean(l1-l2)
    return loss

gph = tf.Graph()
with gph.as_default():
    x1 = tf.placeholder(tf.float32, shape = [1,28,28,1], name = "img1")
    x2 = tf.placeholder(tf.float32, shape = [1,28,28,1], name = "img2")
    cost = laplace_loss(x1,x2)

with tf.Session(graph = gph) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    image = image_array(open(X_TRAIN_FNAME,'r'))
    feed_dict = {x1:next(image),
                 x2:next(image)}
    loss  = sess.run(cost,feed_dict)
    print(loss)
    #exp_image = exp_image.reshape(28, 28)
    #plt.imshow(exp_image,cmap = "binary")
    #plt.show()
