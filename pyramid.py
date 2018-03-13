import tensorflow as tf
import numpy as np

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
