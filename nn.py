import numpy as np
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!

# Q 2.1
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):
    # X: (batch_size, in_size)
    # W: (in_size, out_size)
    # b: (out_size)
    # output: (batch_size, out_size)
    
    low = -(6**0.5) / np.sqrt(in_size + out_size)
    high = -low

    W = np.random.uniform(low, high, (in_size, out_size))
    b = np.zeros((1, out_size))

    params['W' + name] = W
    params['b' + name] = b

# Q 2.2.1
# x is a matrix
# a sigmoid activation function
def sigmoid(x):
    
    res = 1.0 / (1.0 + np.exp(-x))

    return res

# Q 2.2.2
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    pre_act, post_act = None, None
    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]

    # your code here
    pre_act = X @ W + b
    post_act = activation(pre_act)

    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act

# Q 2.2.2 
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    maxs = np.max(x, axis=1, keepdims=True) # (examples, 1)
    exps = np.exp(x-maxs)
    res = exps / np.sum(exps, axis=1, keepdims=True) # (examples, classes)    
    return res

# Q 2.2.3
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    assert 'int' in y.dtype.name
    mask = y.astype(np.bool)
    true_probs = probs[mask] # (examples, )
    loss = 0.0 - np.sum(np.log(true_probs))
    
    # loss /= probs.shape[0]

    cls_pred = np.argmax(probs, axis=1) #(examples, )

    # boolean array, (examples, ) True for correctly predicting
    # the corresponding example
    example_correct = mask[np.arange(cls_pred.shape[0]), cls_pred]

    correct_cnt = np.sum(example_correct)

    acc = float(correct_cnt) / example_correct.shape[0]
    
    return loss, acc 

# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop (examples, dimension)
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    # everything you may need for this layer
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]
    # your code here
    # do the derivative through activation first
    # then compute the derivative W,b, and X
    
    delta_pre = activation_deriv(post_act) * delta
    # (in_dim, out_dim) = (in_dim, examples) @ (examples, out_dim)
    grad_W = X.transpose() @ delta_pre
    grad_b = np.sum(delta_pre, axis=0, keepdims=True) # (1, out_dim)
    # (examples, in_dim) = (examples, out_dim) @ (out_dim, in_dim)
    grad_X = delta_pre @ W.transpose()

    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b
    return grad_X

# Q 2.4
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):
    # x: (total_examples, dim)
    # y: (total_examples, num_classes), one-hot
    
    batches = list()
    N = x.shape[0]

    assert batch_size <= N
    
    idxs = list(np.random.permutation(N))
    
    remainer_num = N % batch_size
    if remainer_num > 0:
        append_on_num = batch_size - remainer_num
        for i in range(append_on_num):
            idxs.append(i)
    
    p = 0
    while p < N:
        batch_idxs = idxs[p:p+batch_size]
        
        batches.append((x[batch_idxs, :], y[batch_idxs, :]))

        p += batch_size

    return batches
