import tensorflow as tf
import math
# Spatial Pyramid Pooling block
# https://arxiv.org/abs/1406.4729
def spatial_pyramid_pool(previous_conv, num_sample, previous_conv_size, out_pool_size):
    '''
    previous_conv: a tensor vector of previous convolution layer
    num_sample: an int number of image in the batch
    previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
    out_pool_size: a int vector of expected output size of max pooling layer

    returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
    '''

    for i in range(len(out_pool_size)):
        h_strd = math.ceil(previous_conv_size[0] / out_pool_size[i])
        w_strd = math.ceil(previous_conv_size[1] / out_pool_size[i])

        max_pool = tf.nn.max_pool(previous_conv,
                                  ksize=[1, h_strd, w_strd, 1],
                                  strides=[1, h_strd, w_strd, 1],
                                  padding='SAME')
        if (i == 0):
            spp = tf.reshape(max_pool, [num_sample, -1])
        else:
            spp = tf.concat(axis=1, values=[spp, tf.reshape(max_pool, [num_sample, -1])])

    return spp
