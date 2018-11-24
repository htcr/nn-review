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
    erosion_selem = np.ones((5, 5), dtype=np.bool)
    processed_img = skimage.morphology.binary_erosion(processed_img, selem=erosion_selem)

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


def group_lines(bboxes):
    # build graph
    N = len(bboxes)
    graph = list()
    for i in range(N):
        graph.append(list())
    
    for i in range(N):
        for j in range(i+1, N):
            ra1, ca1, ra2, ca2 = bboxes[i]
            rb1, cb1, rb2, cb2 = bboxes[j]
            
            vcenter_a = (ra1+ra2) // 2
            vcenter_b = (rb1+rb2) // 2
            
            dist = abs(vcenter_a - vcenter_b)
            max_h = max(ra2 - ra1, rb2 - rb1)
            
            if dist <= max_h // 2:
                graph[i].append(j)
                graph[j].append(i)
    
    lines = list() # list of (avg_row_coord, [box, box...])
    
    visited = [False]*N
    
    sort_boxes_in_line_key = lambda bbox: bbox[1] # sort by left bound
    sort_line_ley = lambda line: line[0] # sort by avg row coord

    for i in range(N):
        if not visited[i]:
            bfsq = list()
            bfsq.append(i)
            q_head = 0
            visited[i] = True
            while len(bfsq) - q_head > 0:
                cur_box_id = bfsq[q_head]
                q_head += 1
                for adj in graph[cur_box_id]:
                    if not visited[adj]:
                        bfsq.append(adj)
                        visited[adj] = True
            line_boxes = [bboxes[box_id] for box_id in bfsq]
            line_boxes.sort(key=sort_boxes_in_line_key)
            avg_row_coord = sum([b[0] for b in line_boxes]) / float(len(bfsq))
            lines.append((avg_row_coord, line_boxes))
    
    lines.sort(key=sort_line_ley)
    
    grouped_bboxes = [line[1] for line in lines]

    return grouped_bboxes

def crop_img(image, bbox):
    imgh, imgw = image.shape[0:2]
    r1, c1, r2, c2 = bbox
    r1 = int(max(0, r1))
    r2 = int(min(imgh-1, r2))
    c1 = int(max(0, c1))
    c2 = int(min(imgw-1, c2))
    
    return image[r1:r2, c1:c2]
    
def is_digit(c):
    return c != None and ord('0') <= ord(c) <= ord('9')

def is_letter(c):
    return c != None and (ord('a') <= ord(c) <= ord('z') or ord('A') <= ord(c) <= ord('Z'))

def context_refine(parsed_chars):

    refined_chars = list()

    prev_char = None
    next_char = None

    N = len(parsed_chars)
    
    for i in range(N):
        prev_char = parsed_chars[i-1] if i > 0 else None
        next_char = parsed_chars[i+1] if i < N-1 else None
        if parsed_chars[i] == '0' and (is_letter(prev_char) or is_letter(next_char)):
            refined_chars.append('O')
        else:
            refined_chars.append(parsed_chars[i])

    return refined_chars

def context_refine2(parsed_chars):

    refined_chars = list()

    prev_char = None
    next_char = None

    N = len(parsed_chars)
    
    for i in range(N):
        prev_char = parsed_chars[i-1] if i > 0 else None
        next_char = parsed_chars[i+1] if i < N-1 else None
        if parsed_chars[i] == '0' and (is_letter(prev_char) or is_letter(next_char)):
            refined_chars.append('O')
        elif parsed_chars[i] == 'Z' and (is_digit(prev_char) or is_digit(next_char)):
            refined_chars.append('2')
        elif parsed_chars[i] == 'G' and (is_digit(prev_char) or is_digit(next_char)):
            refined_chars.append('6')
        elif parsed_chars[i] == 'g' and (is_digit(prev_char) or is_digit(next_char)):
            refined_chars.append('9')
        elif parsed_chars[i] == 'A' and (is_digit(prev_char) or is_digit(next_char)):
            refined_chars.append('4')
        else:
            refined_chars.append(parsed_chars[i])

    return refined_chars

