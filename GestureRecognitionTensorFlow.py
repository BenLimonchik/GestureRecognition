
# coding: utf-8

# ## What's this TensorFlow business?
# 
# You've written a lot of code in this assignment to provide a whole host of neural network functionality. Dropout, Batch Norm, and 2D convolutions are some of the workhorses of deep learning in computer vision. You've also worked hard to make your code efficient and vectorized.
# 
# For the last part of this assignment, though, we're going to leave behind your beautiful codebase and instead migrate to one of two popular deep learning frameworks: in this instance, TensorFlow (or PyTorch, if you switch over to that notebook)
# 
# #### What is it?
# TensorFlow is a system for executing computational graphs over Tensor objects, with native support for performing backpropogation for its Variables. In it, we work with Tensors which are n-dimensional arrays analogous to the numpy ndarray.
# 
# #### Why?
# 
# * Our code will now run on GPUs! Much faster training. Writing your own modules to run on GPUs is beyond the scope of this class, unfortunately.
# * We want you to be ready to use one of these frameworks for your project so you can experiment more efficiently than if you were writing every feature you want to use by hand. 
# * We want you to stand on the shoulders of giants! TensorFlow and PyTorch are both excellent frameworks that will make your lives a lot easier, and now that you understand their guts, you are free to use them :) 
# * We want you to be exposed to the sort of deep learning code you might run into in academia or industry. 

# ## How will I learn TensorFlow?
# 
# TensorFlow has many excellent tutorials available, including those from [Google themselves](https://www.tensorflow.org/get_started/get_started).
# 
# Otherwise, this notebook will walk you through much of what you need to do to train models in TensorFlow. See the end of the notebook for some links to helpful tutorials if you want to learn more or need further clarification on topics that aren't fully explained here.

# ## Load Datasets
# 

# In[2]:


import tensorflow as tf
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[7]:


# from cs231n.data_utils import load_CIFAR10
from cs231n.data_utils import get_MARCEL_data

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

data = get_MARCEL_data()
for k, v in data.items():
  print('%s: ' % k, v.shape)

X_train = data['X_train']
y_train = data['y_train']
X_val = data['X_val']
y_val = data['y_val']


# ## Example Model
# 
# ### Some useful utilities
# 
# . Remember that our image data is initially N x H x W x C, where:
# * N is the number of datapoints
# * H is the height of each image in pixels
# * W is the height of each image in pixels
# * C is the number of channels (usually 3: R, G, B)
# 
# This is the right way to represent the data when we are doing something like a 2D convolution, which needs spatial understanding of where the pixels are relative to each other. When we input image data into fully connected affine layers, however, we want each data example to be represented by a single vector -- it's no longer useful to segregate the different channels, rows, and columns of the data.

# ### The example model itself
# 
# The first step to training your own model is defining its architecture.
# 
# Here's an example of a convolutional neural network defined in TensorFlow -- try to understand what each line is doing, remembering that each layer is composed upon the previous layer. We haven't trained anything yet - that'll come next - for now, we want you to understand how everything gets set up. 
# 
# In that example, you see 2D convolutional layers (Conv2d), ReLU activations, and fully-connected layers (Linear). You also see the Hinge loss function, and the Adam optimizer being used. 
# 
# Make sure you understand why the parameters of the Linear layer are 5408 and 10.
# 
# ### TensorFlow Details
# In TensorFlow, much like in our previous notebooks, we'll first specifically initialize our variables, and then our network model.

# In[12]:


# clear old variables
tf.reset_default_graph()

# setup input (e.g. the data that changes every batch)
# The first dim is None, and gets sets automatically based on batch size fed in
X = tf.placeholder(tf.float32, [None, 76, 66, 3])
y = tf.placeholder(tf.int64, [None])
is_training = tf.placeholder(tf.bool)

def simple_model(X,y):
    # define our weights (e.g. init_two_layer_convnet)
    
    # setup variables
    Wconv1 = tf.get_variable("Wconv1", shape=[7, 7, 3, 32])
    bconv1 = tf.get_variable("bconv1", shape=[32])
    W1 = tf.get_variable("W1", shape=[33600, 10])
    b1 = tf.get_variable("b1", shape=[10])

    # define our graph (e.g. two_layer_convnet)
    a1 = tf.nn.conv2d(X, Wconv1, strides=[1,2,2,1], padding='VALID') + bconv1
    h1 = tf.nn.relu(a1)
    h1_flat = tf.reshape(h1,[-1,33600])
    y_out = tf.matmul(h1_flat,W1) + b1
    return y_out

y_out = simple_model(X,y)

# define our loss
total_loss = tf.losses.hinge_loss(tf.one_hot(y,10),logits=y_out)
mean_loss = tf.reduce_mean(total_loss)

# define our optimizer
optimizer = tf.train.AdamOptimizer(5e-4) # select optimizer and set learning rate
train_step = optimizer.minimize(mean_loss)


# TensorFlow supports many other layer types, loss functions, and optimizers - you will experiment with these next. Here's the official API documentation for these (if any of the parameters used above were unclear, this resource will also be helpful). 
# 
# * Layers, Activations, Loss functions : https://www.tensorflow.org/api_guides/python/nn
# * Optimizers: https://www.tensorflow.org/api_guides/python/train#Optimizers
# * BatchNorm: https://www.tensorflow.org/api_docs/python/tf/contrib/layers/batch_norm

# ### Training the model on one epoch
# While we have defined a graph of operations above, in order to execute TensorFlow Graphs, by feeding them input data and computing the results, we first need to create a `tf.Session` object. A session encapsulates the control and state of the TensorFlow runtime. For more information, see the TensorFlow [Getting started](https://www.tensorflow.org/get_started/get_started) guide.
# 
# Optionally we can also specify a device context such as `/cpu:0` or `/gpu:0`. For documentation on this behavior see [this TensorFlow guide](https://www.tensorflow.org/tutorials/using_gpu)
# 
# You should see a validation loss of around 0.4 to 0.6 and an accuracy of 0.30 to 0.35 below

# In[17]:


def run_model(session, predict, loss_val, Xd, yd,
              epochs=5, batch_size=64, print_every=100,
              training=None, plot_losses=False):
    # have tensorflow compute accuracy
    correct_prediction = tf.equal(tf.argmax(predict,1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # shuffle indicies
    train_indicies = np.arange(Xd.shape[0])
    np.random.shuffle(train_indicies)

    training_now = training is not None
    
    # setting up variables we want to compute (and optimizing)
    # if we have a training function, add that to things we compute
    variables = [mean_loss,correct_prediction,accuracy]
    if training_now:
        variables[-1] = training
    
    # counter 
    iter_cnt = 0
    for e in range(epochs):
        # keep track of losses and accuracy
        correct = 0
        losses = []
        # make sure we iterate over the dataset once
        for i in range(int(math.ceil(Xd.shape[0]/batch_size))):
            # generate indicies for the batch
#             start_idx = (i*batch_size)%X_train.shape[0]
            start_idx = (i*batch_size)%Xd.shape[0]
            idx = train_indicies[start_idx:start_idx+batch_size]
            
            # create a feed dictionary for this batch
            feed_dict = {X: Xd[idx,:],
                         y: yd[idx],
                         is_training: training_now }
            # get batch size
#             actual_batch_size = yd[i:i+batch_size].shape[0]
            actual_batch_size = yd[idx].shape[0]
            
            # have tensorflow compute loss and correct predictions
            # and (if given) perform a training step
            loss, corr, _ = session.run(variables,feed_dict=feed_dict)
            
            # aggregate performance stats
            losses.append(loss*actual_batch_size)
            correct += np.sum(corr)
            
            # print every now and then
            if training_now and (iter_cnt % print_every) == 0:
                print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}"                      .format(iter_cnt,loss,np.sum(corr)/actual_batch_size))
            iter_cnt += 1
        total_correct = correct/Xd.shape[0]
        total_loss = np.sum(losses)/Xd.shape[0]
        print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}"              .format(total_loss,total_correct,e+1))
        if plot_losses:
            plt.plot(losses)
            plt.grid(True)
            plt.title('Epoch {} Loss'.format(e+1))
            plt.xlabel('minibatch number')
            plt.ylabel('minibatch loss')
            plt.show()
    return total_loss,total_correct

with tf.Session() as sess:
    with tf.device("/cpu:0"): #"/cpu:0" or "/gpu:0" 
        sess.run(tf.global_variables_initializer())
        print('Training')
        run_model(sess,y_out,mean_loss,X_train,y_train,1,64,100,train_step,True)
        print('Validation')
        run_model(sess,y_out,mean_loss,X_val,y_val,1,64)


# ## Training a specific model
# 
# In this section, we're going to specify a model for you to construct. The goal here isn't to get good performance (that'll be next), but instead to get comfortable with understanding the TensorFlow documentation and configuring your own model. 
# 
# Using the code provided above as guidance, and using the following TensorFlow documentation, specify a model with the following architecture:
# 
# * 7x7 Convolutional Layer with 32 filters and stride of 1
# * ReLU Activation Layer
# * Spatial Batch Normalization Layer (trainable parameters, with scale and centering)
# * 2x2 Max Pooling layer with a stride of 2
# * Affine layer with 1024 output units
# * ReLU Activation Layer
# * Affine layer from 1024 input units to 10 outputs
# 
# 

# In[18]:


# clear old variables
tf.reset_default_graph()

# define our input (e.g. the data that changes every batch)
# The first dim is None, and gets sets automatically based on batch size fed in
X = tf.placeholder(tf.float32, [None, 76, 66, 3])
y = tf.placeholder(tf.int64, [None])
is_training = tf.placeholder(tf.bool)

# define model
def complex_model(X,y,is_training):    
    # setup variables
    Wconv1 = tf.get_variable("Wconv1", shape=[7, 7, 3, 32])
    bconv1 = tf.get_variable("bconv1", shape=[32])
    W1 = tf.get_variable("W1", shape=[33600, 1024])
    b1 = tf.get_variable("b1", shape=[1024])
    W2 = tf.get_variable("W2", shape=[1024, 10])
    b2 = tf.get_variable("b2", shape=[10])

    # define our graph
    a1 = tf.nn.conv2d(X, Wconv1, strides=[1,1,1,1], padding="VALID") + bconv1
    h1 = tf.nn.relu(a1)
    h2 = tf.contrib.layers.batch_norm(h1, center=True, scale=True, is_training=is_training)
    h3 = tf.layers.max_pooling2d(inputs=h2, pool_size=[2, 2], strides=2) 
    h3_flat = tf.reshape(h3,[-1,33600])
    h4 = tf.matmul(h3_flat,W1) + b1
    h5 = tf.nn.relu(h4)
    h5_flat = tf.reshape(h5,[-1,1024])
    y_out = tf.matmul(h5_flat,W2) + b2
    return y_out


y_out = complex_model(X,y,is_training)


# To make sure you're doing the right thing, use the following tool to check the dimensionality of your output (it should be 64 x 10, since our batches have size 64 and the output of the final affine layer should be 10, corresponding to our 10 classes):

# In[19]:


# Now we're going to feed a random batch into the model 
# and make sure the output is the right size
x = np.random.randn(64, 76, 66, 3)
with tf.Session() as sess:
    with tf.device("/cpu:0"): #"/cpu:0" or "/gpu:0"
        tf.global_variables_initializer().run()

        ans = sess.run(y_out,feed_dict={X:x,is_training:True})
        get_ipython().magic('timeit sess.run(y_out,feed_dict={X:x,is_training:True})')
        print(ans.shape)
        print(np.array_equal(ans.shape, np.array([64, 10])))


# You should see the following from the run above 
# 
# `(64, 10)`
# 
# `True`

# ### GPU!
# 
# Now, we're going to try and start the model under the GPU device, the rest of the code stays unchanged and all our variables and operations will be computed using accelerated code paths. However, if there is no GPU, we get a Python exception and have to rebuild our graph. On a dual-core CPU, you might see around 50-80ms/batch running the above, while the Google Cloud GPUs (run below) should be around 2-5ms/batch.

# In[87]:


try:
    with tf.Session() as sess:
        with tf.device("/cpu:0") as dev: #"/cpu:0" or "/gpu:0"
            tf.global_variables_initializer().run()

            ans = sess.run(y_out,feed_dict={X:x,is_training:True})
            get_ipython().magic('timeit sess.run(y_out,feed_dict={X:x,is_training:True})')
except tf.errors.InvalidArgumentError:
    print("no gpu found, please use Google Cloud if you want GPU acceleration")    
    # rebuild the graph
    # trying to start a GPU throws an exception 
    # and also trashes the original graph
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y = tf.placeholder(tf.int64, [None])
    is_training = tf.placeholder(tf.bool)
    y_out = complex_model(X,y,is_training)


# You should observe that even a simple forward pass like this is significantly faster on the GPU. So for the rest of the assignment (and when you go train your models in assignment 3 and your project!), you should use GPU devices. However, with TensorFlow, the default device is a GPU if one is available, and a CPU otherwise, so we can skip the device specification from now on.

# ### Train the model.
# 
# Now that you've seen how to define a model and do a single forward pass of some data through it, let's  walk through how you'd actually train one whole epoch over your training data (using the complex_model you created provided above).
# 
# Make sure you understand how each TensorFlow function used below corresponds to what you implemented in your custom neural network implementation.
# 
# First, set up an **RMSprop optimizer** (using a 1e-3 learning rate) and a **cross-entropy loss** function. See the TensorFlow documentation for more information
# * Layers, Activations, Loss functions : https://www.tensorflow.org/api_guides/python/nn
# * Optimizers: https://www.tensorflow.org/api_guides/python/train#Optimizers

# In[20]:


# Inputs
#     y_out: is what your model computes
#     y: is your TensorFlow variable with label information
# Outputs
#    mean_loss: a TensorFlow variable (scalar) with numerical loss
#    optimizer: a TensorFlow optimizer
# This should be ~3 lines of code!
total_loss = tf.losses.softmax_cross_entropy(logits=y_out, onehot_labels=tf.one_hot(y,10))
mean_loss = tf.reduce_mean(total_loss)
optimizer = tf.train.RMSPropOptimizer(1e-3)


# In[21]:


# batch normalization in tensorflow requires this extra dependency
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):
    train_step = optimizer.minimize(mean_loss)


# ### Train the model
# Below we'll create a session and train the model over one epoch. You should see a loss of 1.4 to 1.8 and an accuracy of 0.4 to 0.5. There will be some variation due to random seeds and differences in initialization

# In[27]:


sess = tf.Session()

sess.run(tf.global_variables_initializer())
print('Training')
run_model(sess,y_out,mean_loss,X_train,y_train,2,64,100,train_step)


# ### Check the accuracy of the model.
# 
# Let's see the train and test code in action -- feel free to use these methods when evaluating the models you develop below. You should see a loss of 1.3 to 1.5 with an accuracy of 0.45 to 0.55.

# In[28]:


print('Validation')
run_model(sess,y_out,mean_loss,X_val,y_val,1,64)


# ## Train a _great_ model on CIFAR-10!
# 
# Now it's your job to experiment with architectures, hyperparameters, loss functions, and optimizers to train a model that achieves ** >= 70% accuracy on the validation set** of CIFAR-10. You can use the `run_model` function from above.

# ### Things you should try:
# - **Filter size**: Above we used 7x7; this makes pretty pictures but smaller filters may be more efficient
# - **Number of filters**: Above we used 32 filters. Do more or fewer do better?
# - **Pooling vs Strided Convolution**: Do you use max pooling or just stride convolutions?
# - **Batch normalization**: Try adding spatial batch normalization after convolution layers and vanilla batch normalization after affine layers. Do your networks train faster?
# - **Network architecture**: The network above has two layers of trainable parameters. Can you do better with a deep network? Good architectures to try include:
#     - [conv-relu-pool]xN -> [affine]xM -> [softmax or SVM]
#     - [conv-relu-conv-relu-pool]xN -> [affine]xM -> [softmax or SVM]
#     - [batchnorm-relu-conv]xN -> [affine]xM -> [softmax or SVM]
# - **Use TensorFlow Scope**: Use TensorFlow scope and/or [tf.layers](https://www.tensorflow.org/api_docs/python/tf/layers) to make it easier to write deeper networks. See [this tutorial](https://www.tensorflow.org/tutorials/layers) for making how to use `tf.layers`. 
# - **Use Learning Rate Decay**: [As the notes point out](http://cs231n.github.io/neural-networks-3/#anneal), decaying the learning rate might help the model converge. Feel free to decay every epoch, when loss doesn't change over an entire epoch, or any other heuristic you find appropriate. See the [Tensorflow documentation](https://www.tensorflow.org/versions/master/api_guides/python/train#Decaying_the_learning_rate) for learning rate decay.
# - **Global Average Pooling**: Instead of flattening and then having multiple affine layers, perform convolutions until your image gets small (7x7 or so) and then perform an average pooling operation to get to a 1x1 image picture (1, 1 , Filter#), which is then reshaped into a (Filter#) vector. This is used in [Google's Inception Network](https://arxiv.org/abs/1512.00567) (See Table 1 for their architecture).
# - **Regularization**: Add l2 weight regularization, or perhaps use [Dropout as in the TensorFlow MNIST tutorial](https://www.tensorflow.org/get_started/mnist/pros)
# 
# ### Tips for training
# For each network architecture that you try, you should tune the learning rate and regularization strength. When doing this there are a couple important things to keep in mind:
# 
# - If the parameters are working well, you should see improvement within a few hundred iterations
# - Remember the coarse-to-fine approach for hyperparameter tuning: start by testing a large range of hyperparameters for just a few training iterations to find the combinations of parameters that are working at all.
# - Once you have found some sets of parameters that seem to work, search more finely around these parameters. You may need to train for more epochs.
# - You should use the validation set for hyperparameter search, and we'll save the test set for evaluating your architecture on the best parameters as selected by the validation set.
# 
# ### Going above and beyond
# If you are feeling adventurous there are many other features you can implement to try and improve your performance. You are **not required** to implement any of these; however they would be good things to try for extra credit.
# 
# - Alternative update steps: For the assignment we implemented SGD+momentum, RMSprop, and Adam; you could try alternatives like AdaGrad or AdaDelta.
# - Alternative activation functions such as leaky ReLU, parametric ReLU, ELU, or MaxOut.
# - Model ensembles
# - Data augmentation
# - New Architectures
#   - [ResNets](https://arxiv.org/abs/1512.03385) where the input from the previous layer is added to the output.
#   - [DenseNets](https://arxiv.org/abs/1608.06993) where inputs into previous layers are concatenated together.
#   - [This blog has an in-depth overview](https://chatbotslife.com/resnets-highwaynets-and-densenets-oh-my-9bb15918ee32)
# 
# If you do decide to implement something extra, clearly describe it in the "Extra Credit Description" cell below.
# 
# ### What we expect
# At the very least, you should be able to train a ConvNet that gets at **>= 70% accuracy on the validation set**. This is just a lower bound - if you are careful it should be possible to get accuracies much higher than that! Extra credit points will be awarded for particularly high-scoring models or unique approaches.
# 
# You should use the space below to experiment and train your network. The final cell in this notebook should contain the training and validation set accuracies for your final trained network.
# 
# Have fun and happy training!

# In[97]:


# Feel free to play with this cell

def my_model(X,y,is_training):
    # setup variables
    Wconv1 = tf.get_variable("Wconv1", shape=[3, 3, 3, 64])
    bconv1 = tf.get_variable("bconv1", shape=[64])
    Wconv2 = tf.get_variable("Wconv2", shape=[5, 5, 64, 32])
    bconv2 = tf.get_variable("bconv2", shape=[32])
    W1 = tf.get_variable("W1", shape=[5408, 1024])
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
    h3_flat = tf.reshape(h3,[-1,5408])
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

tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.int64, [None])
is_training = tf.placeholder(tf.bool)

# softmax loss function with RMSProp optimizer
y_out = my_model(X,y,is_training)
total_loss = tf.losses.softmax_cross_entropy(logits=y_out, onehot_labels=tf.one_hot(y,10))
mean_loss = tf.reduce_mean(total_loss)
optimizer = tf.train.RMSPropOptimizer(1e-3)

# batch normalization in tensorflow requires this extra dependency
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):
    train_step = optimizer.minimize(mean_loss)


# In[98]:


# Feel free to play with this cell
# This default code creates a session
# and trains your model for 10 epochs
# then prints the validation set accuracy
sess = tf.Session()

sess.run(tf.global_variables_initializer())
print('Training')
run_model(sess,y_out,mean_loss,X_train,y_train,20,64,100,train_step,True)
print('Validation')
run_model(sess,y_out,mean_loss,X_val,y_val,1,64)


# In[100]:


# Test your model here, and make sure 
# the output of this cell is the accuracy
# of your best model on the training and val sets
# We're looking for >= 70% accuracy on Validation
print('Training')
run_model(sess,y_out,mean_loss,X_train,y_train,5,64,100,train_step,True)
print('Validation')
run_model(sess,y_out,mean_loss,X_val,y_val,1,64)


# ### Describe what you did here
# In this cell you should also write an explanation of what you did, any additional features that you implemented, and any visualizations or graphs that you make in the process of training and evaluating your network

# I designed a deep network of the following architecture: [conv-relu-bn-conv-relu-pool] -> [affine]x3 -> [softmax]. For the first convolutional layer I used a small filter size of 3x3 to capture low level features such as edges. Then, in the next convolutional layer I increased the filter size to 5x5 to capture more abstract features from the first conv layer (like colored blobs or small object parts). In addition I used less filters in the second convolutional layer. After the convolutional layers, I used a pooling layer to reduce the data size by 75% to farther manipulate it in a reasonable time. After the pooling layer, I used three affine layers to smoothly reduce the number of neurons to the final number of classes. I used a reduction factor of x4 for the affine layers. I used loss graphs and loss/accuracy reports to visualize and evaluate my model's performance and tune the different hyperparameters accordingly.  

# ### Test Set - Do this only once
# Now that we've gotten a result that we're happy with, we test our final model on the test set. This would be the score we would achieve on a competition. Think about how this compares to your validation set accuracy.

# In[101]:


print('Test')
run_model(sess,y_out,mean_loss,X_test,y_test,1,64)


# ## Going further with TensorFlow
# 
# The next assignment will make heavy use of TensorFlow. You might also find it useful for your projects. 
# 

# # Extra Credit Description
# If you implement any additional features for extra credit, clearly describe them here with pointers to any code in this or other files if applicable.
