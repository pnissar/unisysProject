import tensorflow as tf
import numpy as np
import cv2

def tf_box_filter(x, r):
    """ Applies a box filter to the input tensor x with a given radius r """
    k_size = int(2*r + 1)
    ch = x.get_shape().as_list()[-1]
    weight = 1 / (k_size ** 2)
    
    # Create box kernel
    box_kernel = np.ones((k_size, k_size, ch, 1), dtype=np.float32) * weight
    box_kernel = tf.constant(box_kernel, dtype=tf.float32)
    
    # Apply depthwise convolution (for filtering)
    output = tf.nn.depthwise_conv2d(x, box_kernel, strides=[1, 1, 1, 1], padding='SAME')
    return output


def guided_filter(x, y, r, eps=1e-2):
    """ Applies guided filtering to the input tensors """
    
    x_shape = tf.shape(x)
    N = tf_box_filter(tf.ones_like(x[:, :, :, :1]), r)

    mean_x = tf_box_filter(x, r) / N
    mean_y = tf_box_filter(y, r) / N
    cov_xy = tf_box_filter(x * y, r) / N - mean_x * mean_y
    var_x  = tf_box_filter(x * x, r) / N - mean_x * mean_x

    A = cov_xy / (var_x + eps)
    b = mean_y - A * mean_x

    mean_A = tf_box_filter(A, r) / N
    mean_b = tf_box_filter(b, r) / N

    output = mean_A * x + mean_b
    return output


def fast_guided_filter(lr_x, lr_y, hr_x, r=1, eps=1e-8):
    """ Applies fast guided filtering using downsampled input """
    
    hr_x_shape = tf.shape(hr_x)
    N = tf_box_filter(tf.ones_like(lr_x[:, :, :, :1]), r)

    mean_x = tf_box_filter(lr_x, r) / N
    mean_y = tf_box_filter(lr_y, r) / N
    cov_xy = tf_box_filter(lr_x * lr_y, r) / N - mean_x * mean_y
    var_x  = tf_box_filter(lr_x * lr_x, r) / N - mean_x * mean_x

    A = cov_xy / (var_x + eps)
    b = mean_y - A * mean_x

    mean_A = tf.image.resize(A, hr_x_shape[1:3])
    mean_b = tf.image.resize(b, hr_x_shape[1:3])

    output = mean_A * hr_x + mean_b
    return output


if __name__ == '__main__':
    # Load image
    image = cv2.imread('output_figure1/cartoon2.jpg')
    image = image / 127.5 - 1  # Normalize to [-1, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Placeholder input (Graph Mode for TensorFlow 1.x)
    input_photo = tf.placeholder(tf.float32, [1, None, None, 3])

    # Apply guided filter
    output = guided_filter(input_photo, input_photo, r=5, eps=1)

    # Setup TensorFlow session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        out = sess.run(output, feed_dict={input_photo: image})

    # Convert output image back to range [0, 255]
    out = (np.squeeze(out) + 1) * 127.5
    out = np.clip(out, 0, 255).astype(np.uint8)

    # Save the output
    cv2.imwrite('output_figure1/cartoon2_filter.jpg', out)

    print("Filtered image saved successfully!")
