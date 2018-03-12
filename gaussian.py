import numpy as np
import matplotlib.pyplot as plt

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