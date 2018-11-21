import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt
import pickle

'''

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']

max_iters = 100
# pick a batch size, learning rate
batch_size = 16
learning_rate = 1e-3
hidden_size = 64

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

params = {}

# initialize layers here
initialize_weights(1024, hidden_size, params, 'layer1')
initialize_weights(hidden_size, 36, params, 'output')

epochs = list()
losses = list()
train_accs = list()
val_accs = list()

# with default settings, you should get loss < 150 and accuracy > 80%
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    for xb,yb in batches:
        # forward
        h1 = forward(xb, params,'layer1')
        probs = forward(h1, params,'output',softmax)

        # loss
        # be sure to add loss and accuracy to epoch totals 
        loss, acc = compute_loss_and_acc(yb, probs)
        total_loss += loss
        total_acc += acc

        # backward
        delta1 = probs
        delta1 -= yb
        delta2 = backwards(delta1,params,'output',linear_deriv)
        backwards(delta2,params,'layer1',sigmoid_deriv)

        # apply gradient
        for k,v in params.items():
            if 'grad' in k:
                name = k.split('_')[1]
                param_tensor = params[name]
                grad_tensor = v
                param_tensor -= learning_rate * grad_tensor
    
    total_loss /= len(batches)
    total_acc /= len(batches)

    print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))
    # run on validation set and report accuracy! should be above 75%
    h1 = forward(valid_x, params,'layer1')
    probs = forward(h1, params,'output',softmax)
    valid_loss, valid_acc = compute_loss_and_acc(valid_y, probs)
    print('Validation accuracy: ', valid_acc)

    epochs.append(itr)
    train_accs.append(total_acc)
    val_accs.append(valid_acc)
    losses.append(total_loss)

h1 = forward(test_x, params,'layer1')
test_probs = forward(h1, params,'output',softmax)
test_loss, test_acc = compute_loss_and_acc(test_y, test_probs)
print('test accuracy: ', test_acc)

plt.subplot(2, 1, 1)
plt.plot(epochs, train_accs)
plt.plot(epochs, val_accs)
plt.ylabel('train/val accuracy')

plt.subplot(2, 1, 2)
plt.plot(epochs, losses)
plt.ylabel('loss')

plt.xlabel('epochs')

plt.show()

if False: # view the data
    for crop in xb:
        plt.imshow(crop.reshape(32,32).T)
        plt.show()

saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

'''

# Q3.1.3
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

saved_params = None
with open('q3_weights.pickle', 'rb') as handle:
    saved_params = pickle.load(handle)

weight_tensor = saved_params['Wlayer1']

fig = plt.figure()
grid = ImageGrid(fig, 111, (8, 8))
for i in range(64):
    filter_i = weight_tensor[:, i]
    filter_i = filter_i.reshape(32, 32)
    grid[i].imshow(filter_i)
    grid[i].axis('off')

plt.show()

initialize_weights(1024, 64, saved_params, 'layer1raw')

weight_tensor = saved_params['Wlayer1raw']

fig = plt.figure()
grid = ImageGrid(fig, 111, (8, 8))
for i in range(64):
    filter_i = weight_tensor[:, i]
    filter_i = filter_i.reshape(32, 32)
    grid[i].imshow(filter_i)
    grid[i].axis('off')

plt.show()

# Q3.1.3
confusion_matrix = np.zeros((36, 36))

import string
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()