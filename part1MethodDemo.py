import argparse
import os

from part1 import closest_centroid_idxes_same, get_new_centroids, assign_points_to_centroids, assign_points_to_centroids_multithreaded

import numpy as np
import matplotlib.pyplot as plt

color_idx = {
    0: 'red',
    1: 'blue',
    2: 'green',
    3: 'orange',
    4: 'cyan',
    5: 'magenta'
}

def writeStepToFile(centroids, centroid_assignments, data, num_centroids, dirname, img_idx):
    fig, ax = plt.subplots()
    for i in range(num_centroids):
        xs = list()
        ys = list()
        for j in range(len(centroid_assignments)):
            if (centroid_assignments[j] == i):
                xs.append(data[j][0])
                ys.append(data[j][1])
        ax.scatter(xs, ys, color=color_idx[i], marker='.')
        ax.plot(centroids[i][0], centroids[i][1], color=color_idx[i], marker='x', markersize=20)
    fname = os.path.join(dirname, "iter:%d.png" % img_idx)
    fig.savefig(fname)

def main():
    parser = argparse.ArgumentParser(
                    prog = 'ECE634 Project 2, Part 1')
    parser.add_argument('num_centroids', help="Number of centroids", type=int)
    parser.add_argument('data_per_centroid', help="Number of data points for each centroid", type=int)
    parser.add_argument('-s', "--seed", help="Random seed", default=1, required=False, type=int)
    args = parser.parse_args()

    num_centroids = args.num_centroids
    data_per_centroid = args.data_per_centroid
    np.random.seed(args.seed)

    all_data = list()
    for i in range(num_centroids):
        mean = np.random.randint(0, 100, size=2)
        cov = np.random.randint(1, 10, size=(2, 2))
        cov[0, 1] = cov[1, 0]
        # print(mean)
        # print(cov)
        for n in range(data_per_centroid):
            point = np.random.multivariate_normal(mean, cov)
            all_data.append(point)

    
    dirname = "figures/method-%d-%d-%d/" % (num_centroids, data_per_centroid, args.seed) 
    try:
        os.mkdir(dirname)
    except FileExistsError:
        print("Directory already exists, press enter to continue")
        input()

    plt.scatter([x[0] for x in all_data], [x[1] for x in all_data])
    plt.savefig(os.path.join(dirname, 'original.png'))

    old_centroids = [all_data[i] for i in np.random.randint(0, len(all_data), size=num_centroids)]
    old_centroid_assignments = assign_points_to_centroids_multithreaded(all_data, old_centroids)

    writeStepToFile(old_centroids, old_centroid_assignments, all_data, num_centroids, dirname, 0)

    img_idx = 1
    break_next = False
    for _ in range(10):
        new_centroids = get_new_centroids(all_data, old_centroid_assignments, num_centroids)
        writeStepToFile(new_centroids, old_centroid_assignments, all_data, num_centroids, dirname, img_idx)
        img_idx += 1

        new_centroid_assignments = assign_points_to_centroids_multithreaded(all_data, new_centroids)
        writeStepToFile(new_centroids, new_centroid_assignments, all_data, num_centroids, dirname, img_idx)
        img_idx += 1

        if break_next:
            if not (closest_centroid_idxes_same(new_centroid_assignments, old_centroid_assignments)):
                print("Warning: Thought we converged but didn't")
            break

        if closest_centroid_idxes_same(new_centroid_assignments, old_centroid_assignments):
            print("Converged!")
            break_next = True

        old_centroids = new_centroids
        old_centroid_assignments = new_centroid_assignments

        

if __name__ == "__main__":
    main()