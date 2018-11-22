import numpy as np
import scipy.io
from nn import *
from collections import Counter
import matplotlib.pyplot as plt

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

max_iters = 100
# pick a batch size, learning rate
batch_size = 36 
learning_rate =  3e-5
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter()

# initialize layers here
initialize_weights(1024, hidden_size, params, 'layer1')
initialize_weights(hidden_size, hidden_size, params, 'layer2')
initialize_weights(hidden_size, hidden_size, params, 'layer3')
initialize_weights(hidden_size, 1024, params, 'layer4')

momentum_dict = dict()

for k, v in params.items():
    momentum_tensor = np.zeros(v.shape)
    momentum_dict[k+'_momentum'] = momentum_tensor

params.update(momentum_dict)

epochs = list()
losses = list()

# should look like your previous training loops
for itr in range(max_iters):
    total_loss = 0
    for xb,_ in batches:
        # training loop can be exactly the same as q2!
        # your loss is now squared error
        # delta is the d/dx of (x-y)^2
        # to implement momentum
        #   just use 'm_'+name variables
        #   to keep a saved value over timestamps
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions
        h1 = forward(xb, params, name='layer1', activation=relu)
        h2 = forward(h1, params, name='layer2', activation=relu)
        h3 = forward(h2, params, name='layer3', activation=relu)
        out = forward(h3, params, name='layer4', activation=sigmoid)

        loss = np.sum((out-xb)**2)
        total_loss += loss

        d_out = 2*(out-xb)
        
        d_h3 = backwards(d_out, params, name='layer4', activation_deriv=sigmoid_deriv)
        d_h2 = backwards(d_h3, params, name='layer3', activation_deriv=relu_deriv)
        d_h1 = backwards(d_h2, params, name='layer2', activation_deriv=relu_deriv)
        d_xb = backwards(d_h1, params, name='layer1', activation_deriv=relu_deriv)

        # apply gradient
        for k,v in params.items():
            if 'grad' in k:
                name = k.split('_')[1]
                param_tensor = params[name]
                grad_tensor = v
                
                momentum_tensor = params[name+'_momentum']
                momentum_tensor[:, :] = 0.9*momentum_tensor - learning_rate * grad_tensor

                param_tensor += momentum_tensor

    total_loss /= len(batches)
    print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss))

    epochs.append(itr)
    losses.append(total_loss)

    
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9
    

plt.subplot(1, 1, 1)
plt.plot(epochs, losses)
plt.ylabel('loss')
plt.xlabel('epochs')
plt.show()

# visualize some results
# Q5.3.1
import matplotlib.pyplot as plt
h1 = forward(xb,params,'layer1',relu)
h2 = forward(h1,params,'layer2',relu)
h3 = forward(h2,params,'layer3',relu)
out = forward(h3,params,'layer4',sigmoid)
for i in range(5):
    plt.subplot(2,1,1)
    plt.imshow(xb[i].reshape(32,32).T)
    plt.subplot(2,1,2)
    plt.imshow(out[i].reshape(32,32).T)
    plt.show()


from skimage.measure import compare_psnr as psnr
# evaluate PSNR
# Q5.3.2

