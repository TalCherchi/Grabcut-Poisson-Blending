import numpy as np
import cv2
import argparse
import igraph as ig
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
import time

GC_BGD = 0 # Hard bg pixel
GC_FGD = 1 # Hard fg pixel, will not be used
GC_PR_BGD = 2 # Soft bg pixel
GC_PR_FGD = 3 # Soft fg pixel
N_edge = []
N_w = []


# Define the GrabCut algorithm function
def grabcut(img, rect, n_iter=5):
    # Assign initial labels to the pixels based on the bounding box
    #img = cv2.blur(img, (30,30)) #low blur
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask.fill(GC_BGD)
    x, y, w, h = rect
    w -=x
    h -= y
    #Initalize the inner square to Foreground
    mask[y:y+h, x:x+w] = GC_PR_FGD
    mask[rect[1]+rect[3]//2, rect[0]+rect[2]//2] = GC_FGD

    bgGMM, fgGMM = initalize_GMMs(img, mask)
    K = 0
    #Nlinks - need to calculate once
    b = calc_beta(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            curr_w = 0
            # all possible neighbours
            dxs = [-1 , -1 , -1 , 0 , 0 , 1, 1, 1]
            dys = [-1 , 0 , 1 , -1 , 1 , -1, 0, 1]
            for dx, dy in zip(dxs, dys):
                # check if out of range
                if ( not (0<=i+dx < img.shape[0] and 0 <= j+dy<img.shape[1]) ):
                    continue
                N_edge.append((get_pixel_id(i, j, img), get_pixel_id(i+dx,j+ dy, img)))
                w = get_N_link(i, j, i+dx, j+dy, img,  b)
                curr_w = curr_w + w
                N_w.append(w)

            if curr_w > K: # find max
                K = curr_w

                
    prev_energy = 0
    for i in range(n_iter):
        bgGMM, fgGMM = update_GMMs(img, mask, bgGMM, fgGMM)
        mincut_sets, energy = calculate_mincut(img, mask, bgGMM, fgGMM , K)
        mask = update_mask(mincut_sets, mask)
        if check_convergence((prev_energy, energy)):
            break
        prev_energy = energy
    # Return the final mask and the GMMs
    return mask, bgGMM, fgGMM

def calc_beta(img):
    beta = 0
    neighbors_count=0
    img=img/255
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            curr = 0
            # all possible neighbours
            dxs = [-1, -1, -1, 0, 0, 1, 1, 1]
            dys = [-1, 0, 1, -1, 1, -1, 0, 1]
            for dx, dy in zip(dxs, dys):
                if (not (0 <= i + dx < img.shape[0] and 0 <= j + dy < img.shape[1])):
                    continue
                neighbors_count +=1
                diff = img[i, j] - img[i + dx, j + dy]
                curr += diff.dot(diff)
            beta += curr
    beta = beta / neighbors_count
    beta = 2 * beta
    beta = 1 / beta
    return beta


def initalize_GMMs(img, mask):
    
    n_components = 5
    rows, cols = img.shape[:2]
    # CHANGE
    fg_mask = np.logical_or(( mask == GC_PR_FGD) , (mask == GC_FGD))
    bg_mask = np.logical_or(( mask == GC_PR_BGD) , (mask == GC_BGD))
    fg_pixels = img[fg_mask]
    bg_pixels = img[bg_mask]
 
 
    fgGMM = GaussianMixture(n_components=n_components).fit(fg_pixels.reshape((-1, img.shape[-1])))
    bgGMM = GaussianMixture(n_components=n_components).fit(bg_pixels.reshape((-1, img.shape[-1])))

    return bgGMM, fgGMM


# Define helper functions for the GrabCut algorithm
def update_GMMs(img, mask, bgGMM, fgGMM):

    fg_mask = np.logical_or(( mask == GC_PR_FGD) , (mask == GC_FGD))
    bg_mask = np.logical_or(( mask == GC_PR_BGD) , (mask == GC_BGD))
    
    fgGMM.fit(img[fg_mask].reshape((-1, img.shape[-1])))

    bgGMM.fit(img[bg_mask].reshape((-1, img.shape[-1])))

    return bgGMM, fgGMM

def get_N_link(i, j, oi, oj , img, beta):
        diff = img[i, j] - img[oi, oj]
        d = ((i - oi)**2 + (j-oj)**2)**0.5
        return 50 /d  * np.exp(- beta * diff.dot(diff))


def get_pixel_id(i, j, img): # fro cordinant to id
    return (img.shape[1] * i) + j


def calculate_mincut(img, mask, bgGMM, fgGMM, K):

    min_cut = [[], []]

    rows, cols = img.shape[:2]
    num_pixels = rows * cols
    # Initialize the graph with all the pixels as nodes and add the source and sink nodes
    graph = ig.Graph(directed=False)
    graph.add_vertices(num_pixels + 2)
    source_node = num_pixels #background
    sink_node = num_pixels + 1 #forgroumd
    img_flat = img.reshape(-1, 3)
    neg_log_likelihoods = np.zeros(img_flat.shape[0])
    neg_log_likelihoods2 = np.zeros(img_flat.shape[0])

    # Compute the negative log-likelihood of each pixel under the GMM
    for i in range(fgGMM.n_components):
        # Compute the probability density function of the i-th Gaussian component for each pixel
        pdf_i = fgGMM.weights_[i] * multivariate_normal.pdf(img_flat, mean=fgGMM.means_[i], cov=fgGMM.covariances_[i])
        pdf2_i = bgGMM.weights_[i] * multivariate_normal.pdf(img_flat, mean=bgGMM.means_[i], cov=bgGMM.covariances_[i])
        
        # Add the log probability of the i-th component to the negative log-likelihoods array
        neg_log_likelihoods -= np.log(pdf_i + 1*10**-12)
        neg_log_likelihoods2 -= np.log(pdf2_i+ 1*10**-12)

    fg_D = neg_log_likelihoods.reshape(img.shape[:2])
    bg_D = neg_log_likelihoods2.reshape(img.shape[:2])

    edges = []
    weights = []

    # TLinks
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            edges.append((get_pixel_id(i, j, img), source_node))
            edges.append((get_pixel_id(i, j, img), sink_node))
            if mask[i][j] == GC_BGD:
                weights.append(0)
                weights.append(K)
            elif mask[i][j] == GC_FGD:
                weights.append(K)
                weights.append(0)
            else:
                weights.append(fg_D[i, j] )
                weights.append(bg_D[i, j] )


    weights.extend(N_w)
    edges.extend(N_edge)
    graph.add_edges(edges, attributes={'weight': weights})
    cut = graph.st_mincut(source_node, sink_node , capacity='weight')
    bgd_cut , fgd_cut = cut.partition
    min_cut = [[bgd_cut], [fgd_cut]]
    return min_cut, cut.value



def update_mask(mincut_sets, mask):
    updated_mask = np.copy(mask)
    updated_mask = updated_mask.reshape(-1,)

    background_set, foreground_set = mincut_sets

    #remove source and sink
    foreground_set = np.delete(foreground_set, [-1])
    background_set = np.delete(background_set, [-1])

    for index in background_set:
        if updated_mask[index] == GC_PR_FGD :
            updated_mask[index] = GC_PR_BGD

    for index in foreground_set:
        if updated_mask[index] == GC_PR_BGD:
            updated_mask[index] = GC_PR_FGD


    updated_mask = updated_mask.reshape(mask.shape)

    return updated_mask


def check_convergence(energy):
    convergence = False
    if len(energy) < 2:
        return False

    energy_diff = abs(energy[-1] - energy[-2])
    # threshold of 5% diff 
    convergence = (energy_diff / energy[1]) < 0.05
    return convergence


def cal_metric(predicted_mask, gt_mask):
    rows, cols = gt_mask.shape[:2]
    counter = 0
    intersection = np.logical_and(predicted_mask, gt_mask)
    union = np.logical_or(predicted_mask, gt_mask)
    jaccard_similarity_score = intersection.sum() / float(union.sum())
    for r in range(rows):
        for c in range(cols):
            if gt_mask[r][c] == predicted_mask[r][c]:
                counter += 1

    return counter / (rows * cols), jaccard_similarity_score

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_name', type=str, default='bush', help='name of image from the course files')
    parser.add_argument('--eval', type=int, default=1, help='calculate the metrics')
    parser.add_argument('--input_img_path', type=str, default='', help='if you wish to use your own img_path')
    parser.add_argument('--use_file_rect', type=int, default=1, help='Read rect from course files')
    parser.add_argument('--rect', type=str, default='1,1,100,100', help='if you wish change the rect (x,y,w,h')
    return parser.parse_args()

if __name__ == '__main__':
    # Load an example image and define a bounding box around the object of interest
    args = parse()


    if args.input_img_path == '':
        input_path = f'data/imgs/{args.input_name}.jpg'
    else:
        input_path = args.input_img_path

    if args.use_file_rect:
        rect = tuple(map(int, open(f"data/bboxes/{args.input_name}.txt", "r").read().split(' ')))
    else:
        rect = tuple(map(int,args.rect.split(',')))


    img = cv2.imread(input_path)

    # Run the GrabCut algorithm on the image and bounding box
    start = time.time()
    mask, bgGMM, fgGMM = grabcut(img, rect)
    end = time.time()
    print(end- start)
    mask[mask == GC_PR_BGD]= GC_BGD
    mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)[1]

    # Print metrics only if requested (valid only for course files)
    if args.eval:
        gt_mask = cv2.imread(f'data/seg_GT/{args.input_name}.bmp', cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.threshold(gt_mask, 0, 1, cv2.THRESH_BINARY)[1]
        acc, jac = cal_metric(mask, gt_mask)
        print(f'Accuracy={acc}, Jaccard={jac}')

    # Apply the final mask to the input image and display the results
    img_cut = img * (mask[:, :, np.newaxis])
    cv2.imshow('Original Image', img)
    cv2.imshow('GrabCut Mask', 255 * mask)
    cv2.imshow('GrabCut Result', img_cut)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
