import argparse 
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from common import *


def get_basis_vectors():
    
    x1s = np.linspace(0, 7, 8)
    x2s = np.linspace(0, 7, 8)

    basis_vectors = list()

    img = np.zeros((64, 64))

    for kx1 in range(0, 8):
        for kx2 in range(0, 8):
            y1s = np.cos( (2*x1s + 1) * kx1 * np.pi / 16 ) * .5
            y2s = np.cos( (2*x2s + 1) * kx2 * np.pi / 16 ) * .5

            ys = np.zeros((8, 8))
            for i in range(8):
                for j in range(8):
                    ys[i, j] = np.abs(y1s[i] + y2s[j])



            img[kx1*8:kx1*8+8, kx2*8:kx2*8+8] = ys
            basis_vectors.append(ys.flatten())

    # plt.imshow(img, cmap='gray')
    # plt.xticks(np.arange(-0.5, 63.5, 8), ['' for i in range(8)])
    # plt.yticks(np.arange(-0.5, 63.5, 8), ['' for i in range(8)])
    # plt.grid()
    # plt.show()
    # plt.savefig('figures/dct_basis_vectors.png')

    basis_vectors = np.array(basis_vectors)

    return basis_vectors

def row_col_to_index(row, col):
    return row * 8 + col

def get_diagonal_idx_pattern():
    idx_pattern = list()

    for sum in range(0, 15):
        for col in range(0, sum+1):
            row = sum - col
            if sum < 7 or (sum >= 7 and row < 8 and col < 8):
                # print(row, col)
                idx_pattern.append(row_col_to_index(col, row))

    # print(idx_pattern)
    return idx_pattern

def convert_img_to_flattened_blocks(img):
    blocks = list()
    for i in range(0, img.shape[0], 8):
        for j in range(0, img.shape[1], 8):
            block = img[i:i+8, j:j+8]
            if block.shape[0] == 8 and block.shape[1] == 8:
                blocks.append(block.flatten())

    return blocks

def reconstruct_img_with_top_n_coefs(weights, n, original_img_shape):
    basis_vectors = get_basis_vectors()
    
    std_of_coefs = list(np.std(np.array(weights), axis=0))
    idxes = [i for i in range(len(std_of_coefs))]

    #sort std_of_coefs
    for i in range(len(std_of_coefs)):
        for j in range(i+1, len(std_of_coefs)):
            if std_of_coefs[i] < std_of_coefs[j]:
                std_of_coefs[i], std_of_coefs[j] = std_of_coefs[j], std_of_coefs[i]
                idxes[i], idxes[j] = idxes[j], idxes[i]

    for i in range(len(std_of_coefs)-1):
        if std_of_coefs[i] < std_of_coefs[i+1]:
            raise Exception("std_of_coefs not sorted correctly")
        
    keep_idxes = idxes[:n]
    remove_idxes = idxes[n:]

    X = basis_vectors
    predicted_blocks = list()
    for B in weights:
        B[remove_idxes] = 0
        y = np.matmul(X, B)
        predicted_blocks.append(y.reshape(8, 8))

    predicted_img = np.zeros((original_img_shape[0] - 1 // 8, original_img_shape[1] - 1 // 8))
    for i in range(0, original_img_shape[0], 8):
        for j in range(0, original_img_shape[1], 8):
            predicted_img[i:i+8, j:j+8] = predicted_blocks.pop(0)

    return predicted_img

def part_a(img_path):
    original_img = cv2.imread(img_path, 0) / 255.0

    basis_vectors = get_basis_vectors()
    blocks = convert_img_to_flattened_blocks(original_img)

    X = basis_vectors
    X_transpose = np.transpose(X)
    t1 = np.linalg.inv(np.matmul(X_transpose, X))
    t1 = np.matmul(t1, X_transpose)

    weights = list()
    for block in blocks:
        y = block
        B = np.matmul(t1, y)
        weights.append(B)

    std_of_coefs = np.std(np.array(weights), axis=0)
    std_of_coefs_zigzag = list()
    for x in get_diagonal_idx_pattern():
        std_of_coefs_zigzag.append(std_of_coefs[x])

    predicted_img_64 = reconstruct_img_with_top_n_coefs(weights, 64, original_img.shape)
    predicted_img_32 = reconstruct_img_with_top_n_coefs(weights, 32, original_img.shape)
    predicted_img_16 = reconstruct_img_with_top_n_coefs(weights, 16, original_img.shape)
    predicted_img_8 = reconstruct_img_with_top_n_coefs(weights, 8, original_img.shape)
    predicted_img_4 = reconstruct_img_with_top_n_coefs(weights, 4, original_img.shape)
    predicted_img_2 = reconstruct_img_with_top_n_coefs(weights, 2, original_img.shape)

    fig, axarr = plt.subplots(4, 2, figsize=(9, 16))
    fig.suptitle("Reconstructing using DCT and changing number of coefficients")
    
    for ax in axarr.flatten():
        ax.axis('off')

    ax = axarr[0, 0]
    ax.imshow(original_img, cmap='gray')
    ax.title.set_text('Original Image')

    ax = axarr[0, 1]
    ax.plot(std_of_coefs_zigzag)
    ax.title.set_text('Std. Deviation of Coefficients')
    ax.set_xlabel('Coef. Index in Zig Zag Order')
    ax.set_ylabel('Std. Deviation')
    ax.axis('on')

    ax = axarr[1, 0]
    ax.imshow(predicted_img_64, cmap='gray')
    ax.title.set_text('Reconstructed using 64 Coefficients')

    ax = axarr[1, 1]
    ax.imshow(predicted_img_32, cmap='gray')
    ax.title.set_text('Reconstructed using 32 Coefficients')

    ax = axarr[2, 0]
    ax.imshow(predicted_img_16, cmap='gray')
    ax.title.set_text('Reconstructed using 16 Coefficients')

    ax = axarr[2, 1]
    ax.imshow(predicted_img_8, cmap='gray')
    ax.title.set_text('Reconstructed using 8 Coefficients')

    ax = axarr[3, 0]
    ax.imshow(predicted_img_4, cmap='gray')
    ax.title.set_text('Reconstructed using 4 Coefficients')

    ax = axarr[3, 1]
    ax.imshow(predicted_img_2, cmap='gray')
    ax.title.set_text('Reconstructed using 2 Coefficients')

    plt.subplots_adjust(left=0, bottom=0, right=.99, top=.93, wspace=0, hspace=.279)

    plt.savefig('figures/part2a-%s.png' % os.path.basename(img_path))
    plt.show()

def part_b(img_path):
    original_img = cv2.imread(img_path, 0) / 255.0

    basis_vectors = get_basis_vectors()
    blocks = convert_img_to_flattened_blocks(original_img)

    X = basis_vectors
    X_transpose = np.transpose(X)
    t1 = np.linalg.inv(np.matmul(X_transpose, X))
    t1 = np.matmul(t1, X_transpose)

    weights = list()
    for block in blocks:
        y = block
        B = np.matmul(t1, y)
        weights.append(B)

    mses = list()
    coefs_used = list()
    for i in tqdm(range(63, 0, -1)):
        predicted_img = reconstruct_img_with_top_n_coefs(weights, i, original_img.shape)
        mse = get_psnr(original_img, predicted_img)
        mses.append(mse)
        coefs_used.append(i)

    fig, axarr = plt.subplots(1, 2, figsize=(9, 4))

    for ax in axarr.flatten():
        ax.axis('off')

    ax = axarr[0]
    ax.imshow(original_img, cmap='gray')
    ax.title.set_text('Original Image')

    ax = axarr[1]
    ax.plot(coefs_used, mses)
    ax.title.set_text('PSNR vs Number of Coefficients Used')
    ax.set_xlabel('Number of DCT Coefficients Used')
    ax.set_ylabel('PSNR (dB)')
    ax.axis('on')

    plt.savefig('figures/part2b-%s.png' % os.path.basename(img_path))
    

def main():
    parser = argparse.ArgumentParser(
                    prog = 'ECE634 Project 2, Part 2')
    parser.add_argument('part', help="(a|b)", type=str)
    parser.add_argument('img', help="Image to quantize", type=str, nargs=1)
    args = parser.parse_args()
    get_diagonal_idx_pattern()

    if args.part == 'a':
        part_a(args.img[0])
    elif args.part == 'b':
        part_b(args.img[0])
    else:
        print("Invalid part, must be a or b")

if __name__ == "__main__":
    main()