import argparse
import threading
import os

from common import *

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import random

random.seed(1)

def convert_img_to_flattened_blocks(img):
    training_data = list()
    for i in range(0, img.shape[0], 4):
        for j in range(0, img.shape[1], 4):
            block = img[i:i+4, j:j+4]
            if block.shape[0] == 4 and block.shape[1] == 4:
                training_data.append(block.flatten())

    return training_data

def distance_squared(point1, point2):
    dist = (point2 - point1) ** 2
    return np.sum(dist)

def assign_points_to_centroids_multithreaded(points, centroids, num_threads = 16):
    ''' Returns list of points of len(points) where each element is the index of the centroid that the point is closest to
        Points: List of np arrays
        Centroids: List of np arrays
    '''
    def assign_point_rng_to_centroid(point_idxes, points, centroids, rtn_list):
        rtn_list.append(list())
        rtn_list.append(list())
        
        for point_idx in point_idxes:
            closest_idx = -1
            closest_dist = -1
            for curr_centroid in range(len(centroids)):
                dist = distance_squared(points[point_idx], centroids[curr_centroid])
                if closest_idx == -1 or dist < closest_dist:
                    closest_idx = curr_centroid
                    closest_dist = dist

            rtn_list[0].append(point_idx)
            rtn_list[1].append(closest_idx)

    threads = list()
    delta = int(len(points) / num_threads) + 1
    for i in range(num_threads):
        rtn_list = list()
        start_idx = i * delta
        end_idx = min( (i + 1) * delta, len(points) )
        rng = range(start_idx, end_idx)
        t = threading.Thread(target=assign_point_rng_to_centroid, args=(rng, points, centroids, rtn_list))
        threads.append((t, rtn_list))

    #for t, _ in threads:
        t.start()

    point_idxes = list()
    closest_centroid_idxes = list()
    for t, rtn_list in threads:
        t.join()
        tmp_point_idx = rtn_list[0]
        tmp_closest_centroid_idx = rtn_list[1]
        [point_idxes.append(point_idx) for point_idx in tmp_point_idx]
        [closest_centroid_idxes.append(closest_centroid_idx) for closest_centroid_idx in tmp_closest_centroid_idx]


    # # selection sort point_idx
    # for i in range(len(point_idxes)-1):
    #     min_idx = i
    #     for j in range(i+1, len(point_idxes)):
    #         if point_idxes[j] < point_idxes[min_idx]:
    #             min_idx = j

    #     point_idxes[i], point_idxes[min_idx] = point_idxes[min_idx], point_idxes[i]
    #     closest_centroid_idxes[i], closest_centroid_idxes[min_idx] = closest_centroid_idxes[min_idx], closest_centroid_idxes[i]


    for i in range(len(point_idxes)-1):
        if (point_idxes[i] > point_idxes[i+1]):
            raise Exception("Sorting error")
        if (point_idxes[i] + 1 != point_idxes[i+1]):
            raise Exception("Missing Indexes")

    return closest_centroid_idxes

def assign_points_to_centroids(points, centroids):
    ''' Returns list of points of len(points) where each element is the index of the centroid that the point is closest to
        Points: List of np arrays
        Centroids: List of np arrays
    '''
    closest_centroid_idx = list()
    for i in range(len(points)):
        closest = -1
        closest_dist = -1
        for j in range(len(centroids)):
            dist = distance_squared(points[i], centroids[j])
            if closest == -1 or dist < closest_dist:
                closest = j
                closest_dist = dist
        
        closest_centroid_idx.append(closest)

    return closest_centroid_idx

def get_new_centroids(points, closest_centroid_idx, num_centroids):
    new_centroids = list()
    for i in range(num_centroids):
        curr_sum = np.zeros(points[0].shape)
        count = 0

        curr_group = list()

        for j in range(len(points)):
            if closest_centroid_idx[j] == i:
                curr_sum += points[j].astype(np.float32)
                count += 1

                curr_group.append(points[j])

        new_centroids.append(curr_sum / count)


    return new_centroids

def closest_centroid_idxes_same(old_closest_centroid_idx, new_closest_centroid_idx):
    if len(old_closest_centroid_idx) != len(new_closest_centroid_idx):
        raise Exception("Dimension mismatch")
    
    for i in range(len(old_closest_centroid_idx)):
        if old_closest_centroid_idx[i] != new_closest_centroid_idx[i]:
            return False

    return True

def get_codebook(codebook_size, training_img_fpaths):
    training_data = list()

    for img_fpath in training_img_fpaths:
        print(img_fpath)
        img = cv2.imread(img_fpath, 0)
        training_data = training_data + convert_img_to_flattened_blocks(img)

    if codebook_size >= len(training_data):
        print("Codebook size is bigger than the number of training data points")
        return

    rand_idxes = list()
    starting_centroids = list()
    for _ in range(codebook_size):
        idx = random.randint(0, len(training_data) - 1)
        while idx in rand_idxes:
            idx = random.randint(0, len(training_data) - 1)

        rand_idxes.append(idx)
        starting_centroids.append(training_data[idx])

    old_centroids = starting_centroids
    old_centroid_assignments = assign_points_to_centroids_multithreaded(training_data, old_centroids)
    new_centroids = None

    break_next = False
    for _ in tqdm(range(20)):
        new_centroids = get_new_centroids(training_data, old_centroid_assignments, codebook_size)
        new_centroid_assignments = assign_points_to_centroids_multithreaded(training_data, new_centroids)
        
        if break_next:
            if not (closest_centroid_idxes_same(old_centroid_assignments, new_centroid_assignments)):
                print("Warning: Thought we converged but didn't")
            break
        
        if (closest_centroid_idxes_same(old_centroid_assignments, new_centroid_assignments)):
            print("Converged!")
            break_next = True
        
        old_centroids = new_centroids
        old_centroid_assignments = new_centroid_assignments
    
    return new_centroids

def quantize_img(img_fpath, codebook):
    original_img = cv2.imread(img_fpath, 0)

    blocks = convert_img_to_flattened_blocks(original_img)

    codebook_idx_match = list()
    for i in range(len(blocks)):
        closest_idx = -1
        for j in range(len(codebook)):
            if closest_idx == -1 or distance_squared(blocks[i], codebook[j]) < distance_squared(blocks[i], codebook[closest_idx]):
                closest_idx = j
        codebook_idx_match.append(closest_idx)

    codebook_idx_match = np.array(codebook_idx_match).reshape(original_img.shape[0] // 4, original_img.shape[1] // 4)

    codebook_reshaped = list()
    for i in range(len(codebook)):
        codebook_entry = codebook[i].reshape(4, 4)
        codebook_reshaped.append(codebook_entry)

    img = np.zeros(original_img.shape)

    for i in range(codebook_idx_match.shape[0]):
        for j in range(codebook_idx_match.shape[1]):
            img[i*4:(i+1)*4, j*4:(j+1)*4] = codebook_reshaped[codebook_idx_match[i, j]]

    
    # plt.imshow(img, cmap='gray')
    # plt.show()

    return img, get_psnr(img, original_img, max=255)

def main():
    parser = argparse.ArgumentParser(
                    prog = 'ECE634 Project 2, Part 1')
    parser.add_argument('part', help="(a|b) Part of project to execute code for", type=str)
    parser.add_argument('codebook_size', help="(128|256) Codebook size", type=int)
    parser.add_argument('training_imgs', help="Paths to training images", type=str, nargs='+')
    parser.add_argument('quantize_img', help="Path to image to quantize", type=str, nargs=1)

    args = parser.parse_args()

    codebook = get_codebook(args.codebook_size, args.training_imgs)
    img, psnr = quantize_img(args.quantize_img[0], codebook)

    if args.part == 'a':
        fig, axarr = plt.subplots(1, 2, figsize=(9, 5))
        fig.suptitle("Quantized Image with Codebook Size %d" % args.codebook_size)
        for ax in axarr.flatten():
            ax.axis('off')

        ax = axarr[0]
        ax.title.set_text("Original Image (Used for Training)")
        ax.imshow(cv2.imread(args.training_imgs[0], 0), cmap='gray')
        ax = axarr[1]
        ax.title.set_text("Quantized Image (PSNR: %.2f)" % psnr)
        ax.imshow(img, cmap='gray')

        plt.savefig('figures/part1a-%d-%s.png' % (args.codebook_size, os.path.basename(args.quantize_img[0])))
        plt.show()
    elif args.part == 'b':
        fig, axarr = plt.subplots(1, 2, figsize=(9, 5))
        fig.suptitle("Quantized Image with Codebook Size %d" % args.codebook_size)
        for ax in axarr.flatten():
            ax.axis('off')

        ax = axarr[0]
        ax.title.set_text("Original Image")
        ax.imshow(cv2.imread(args.quantize_img[0], 0), cmap='gray')
        ax = axarr[1]
        ax.title.set_text("Quantized Image (PSNR: %.2f)" % psnr)
        ax.imshow(img, cmap='gray')

        plt.savefig('figures/part1b-%d-%s.png' % (args.codebook_size, os.path.basename(args.quantize_img[0])))
        # plt.show()
    else:
        raise Exception("Invalid part")
    
    

if __name__ == "__main__":
    main()
