import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
from scipy.sparse.linalg import spsolve
import argparse

def laplac(y, x, mask):
    matrix_A = scipy.sparse.lil_matrix((x, x))
    values=[-1, -1, 4]
    diagonals=[-1, 1, 0]
    for val, diag in zip(values, diagonals):
        matrix_A.setdiag(val, diag)
    matrix_D = scipy.sparse.block_diag([matrix_A] * y).tolil()
    matrix_D.setdiag(-1, x)
    matrix_D.setdiag(-1, -x)
    laplac_mat=matrix_D
    for m in range(1, y - 1):
        for n in range(1, x - 1):
            if mask[m, n] == 0:
                k = n + (m * x)
                for i in [1, -1, x, -x]:
                    matrix_D[k, k + i] = 0
                matrix_D[k, k] = 1

    return matrix_D.tocsc(),laplac_mat.tocsc()

def calc_b(mask,source, target, laplac_mat):#alpha=1
        b = laplac_mat.dot(source)
        b[mask == 0] = target[mask == 0]
        return b

def calc_x(D, b, y_max, x_max):
    x = spsolve(D, b).reshape(y_max, x_max)
    x[x > 255] = 255
    x[x < 0] = 0
    return x.astype('uint8')


def poisson_blend(im_src, im_tgt, im_mask, center):
    target = im_tgt.copy()
    h_pad = int(abs(im_tgt.shape[0] - im_mask.shape[0]) / 2)
    w_pad = int(abs(im_tgt.shape[1] - im_mask.shape[1]) / 2)
    source = np.zeros(target.shape)
    mask = np.zeros(target.shape[:-1])
    source[h_pad:h_pad + im_mask.shape[0], w_pad:w_pad + im_mask.shape[1]] = im_src
    mask[h_pad:h_pad + im_mask.shape[0], w_pad:w_pad + im_mask.shape[1]] = im_mask
    y_max, x_max = source.shape[:-1]
    mask = mask[0:y_max, 0:x_max]
    mask[mask != 0] = 1
    matrix_D,laplac_mat = laplac(y_max, x_max, mask)
    flatted_mask = mask.flatten()
    for colour in range(3):
        flatted_src = source[0:y_max, 0:x_max, colour].flatten()
        flatted_tgt = target[0:y_max, 0:x_max, colour].flatten()
        b=calc_b(flatted_mask,flatted_src, flatted_tgt, laplac_mat)
        x=calc_x(matrix_D, b, y_max, x_max)
        target[0:y_max, 0:x_max, colour] = x
    im_blend = target
    return im_blend


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='./data/imgs/book.jpg', help='image file path')
    parser.add_argument('--mask_path', type=str, default='./data/seg_GT/book.bmp', help='mask file path')
    parser.add_argument('--tgt_path', type=str, default='./data/bg/table.jpg', help='mask file path')
    return parser.parse_args()

if __name__ == "__main__":
    # Load the source and target images
    args = parse()

    im_tgt = cv2.imread(args.tgt_path, cv2.IMREAD_COLOR)
    im_src = cv2.imread(args.src_path, cv2.IMREAD_COLOR)
    if args.mask_path == '':
        im_mask = np.full(im_src.shape, 255, dtype=np.uint8)
    else:
        im_mask = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)
        im_mask = cv2.threshold(im_mask, 0, 255, cv2.THRESH_BINARY)[1]

    center = (int(im_tgt.shape[1] / 2), int(im_tgt.shape[0] / 2))

    im_clone = poisson_blend(im_src, im_tgt, im_mask, center)

    cv2.imshow('Cloned image', im_clone)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
