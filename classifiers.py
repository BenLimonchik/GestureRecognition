import tensorflow as tf


def my_model(X,y,is_training):
    # setup variables
    Wconv1 = tf.get_variable("Wconv1", shape=[3, 3, 3, 64])
    bconv1 = tf.get_variable("bconv1", shape=[64])
    Wconv2 = tf.get_variable("Wconv2", shape=[5, 5, 64, 32])
    bconv2 = tf.get_variable("bconv2", shape=[32])
    W1 = tf.get_variable("W1", shape=[33600, 1024])
    b1 = tf.get_variable("b1", shape=[1024])
    W2 = tf.get_variable("W2", shape=[1024, 512])
    b2 = tf.get_variable("b2", shape=[512])
    W3 = tf.get_variable("W3", shape=[512, 128])
    b3 = tf.get_variable("b3", shape=[128])
    W4 = tf.get_variable("W4", shape=[128, 10])
    b4 = tf.get_variable("b4", shape=[10])

    # first conv layer (with relu and bn)
    a1 = tf.nn.conv2d(X, Wconv1, strides=[1,1,1,1], padding="VALID") + bconv1
    h1 = tf.nn.relu(a1)
    h2 = tf.contrib.layers.batch_norm(h1, center=True, scale=True, is_training=is_training, scope='bn')
    
    # second conv layer (with only relu)
    a11 = tf.nn.conv2d(h2, Wconv2, strides=[1,1,1,1], padding="VALID") + bconv2
    h11 = tf.nn.relu(a11)
    
    # pool layer
    h3 = tf.layers.max_pooling2d(inputs=h11, pool_size=[2, 2], strides=2) 
    
    # three affine layers
    h3_flat = tf.reshape(h3,[-1,33600])
    h4 = tf.matmul(h3_flat,W1) + b1
    h5 = tf.nn.relu(h4)
    
    h5_flat = tf.reshape(h5,[-1,1024])
    h6 = tf.matmul(h5_flat,W2) + b2
    h7 = tf.nn.relu(h6)
    
    h7_flat = tf.reshape(h7,[-1,512])
    h8 = tf.matmul(h7_flat,W3) + b3
    h9 = tf.nn.relu(h8)
    
    # output layer (softmax)
    h9_flat = tf.reshape(h9,[-1,128])
    y_out = tf.matmul(h9_flat,W4) + b4
    
    return y_out