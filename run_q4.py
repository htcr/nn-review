import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)



for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, bw = findLetters(im1)

    grouped_bboxes = group_lines(bboxes)
    
    fig, ax = plt.subplots()
    ax.imshow(bw)
    for line_bboxes in grouped_bboxes:
        link_x, link_y = list(), list()
        for bbox in line_bboxes:
            minr, minc, maxr, maxc = bbox
            rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                    fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)

            link_x.append(minc)
            link_y.append(minr)
            ax.plot(link_x, link_y, linewidth=2, color='green')

    plt.show()
    
    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    
    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle','rb'))

    ans_char_list = list()

    for line_bboxes in grouped_bboxes:
        # record the size and left bound of previous box
        # to decide when to insert space
        prev_left = -1
        prev_width = -1
        for idx, box in enumerate(line_bboxes):
            r1, c1, r2, c2 = box
            left = c1
            width = c2 - c1
            if idx > 0:
                offset = left - prev_left
                if offset > prev_width * 1.5:
                    ans_char_list.append(' ')
            
            prev_left = left
            prev_width = width
            
            char_crop = crop_img(bw, box)
            char_crop = skimage.transform.resize(char_crop, (32, 32))
            char_crop_flat = char_crop.transpose().reshape((1, -1))
            
            # forward
            h1 = forward(char_crop_flat, params, 'layer1')
            probs = forward(h1, params,'output',softmax)

            cls_pred = np.argmax(probs, axis=1)[0]
            ans_char_list.append(letters[cls_pred])
        ans_char_list.append('\n')
    
    print('===== image {} content ====='.format(img))
    print(''.join(ans_char_list))
    refined_char_list = context_refine(ans_char_list)
    print('refined: ')
    print(''.join(refined_char_list))
    
