import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation
import matplotlib.pyplot as plt

# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions
    
    '''
    print(image.dtype.name)
    print(image.shape)
    print(np.mean(image))
    '''

    height, width, channel = image.shape

    # plt.imshow(image)
    # plt.show()

    gray_img = skimage.color.rgb2gray(image)
    # plt.imshow(gray_img)
    # plt.show()
    # print(gray_img.shape)
    
    max_edge = max(height, width)

    threshold_block_size = max_edge // 20 * 2 + 1

    thresh_adapt_img = skimage.filters.threshold_adaptive(gray_img, threshold_block_size, offset=0.1)

    processed_img = thresh_adapt_img.copy()

    erosion_selem = np.ones((9, 9), dtype=np.bool)
    thresh_adapt_img = skimage.morphology.binary_erosion(thresh_adapt_img, selem=erosion_selem)

    '''
    global_thresh = skimage.filters.threshold_otsu(gray_img)
    thresh_global_img = image > global_thresh

    plt.imshow(thresh_global_img)
    plt.show()
    '''

    thresh_adapt_img = 1 - thresh_adapt_img.astype(np.uint8)

    #plt.imshow(thresh_adapt_img)
    #plt.show()

    labeled_img, region_num = skimage.measure.label(thresh_adapt_img, background=0, return_num=True)

    #plt.imshow(labeled_img)
    #plt.show()

    min_box_size = 0.01 * max(height, width)
    max_box_size = 0.5 * max(height, width)
    min_hw_ratio = 0.6
    max_hw_ratio = 1.0 / min_hw_ratio

    bboxes = list()
    for label in range(1, region_num+1):
        rs, cs = np.where(labeled_img==label)
        minr, minc = np.min(rs), np.min(cs)
        maxr, maxc = np.max(rs), np.max(cs)

        cr, cc = (minr+maxr)//2, (minc+maxc)//2

        boxh, boxw = maxr - minr, maxc - minc

        hw_ratio = float(boxh) / float(boxw)

        if hw_ratio < min_hw_ratio:
            boxh = int(boxw*min_hw_ratio)
        elif hw_ratio > max_hw_ratio:
            boxw = int(boxh/max_hw_ratio)

        minr, maxr = cr - boxh // 2, cr + boxh // 2
        minc, maxc = cc - boxw // 2, cc + boxw // 2

        padh, padw = 0.2*boxh, 0.2*boxw

        box_size = max(boxh, boxw)
        
        if min_box_size < box_size < max_box_size:
            bboxes.append((minr-padh, minc-padw, maxr+padh, maxc+padw))

    bw = processed_img

    return bboxes, bw