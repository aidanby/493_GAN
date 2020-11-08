# Raw FID scores over epochs
import tensorflow as tf
import numpy as np
import scipy as sp
import multiprocessing
import os


class Process:
    def __init__(self):
        print("Evaluating images")

    def fid(self, info1, info2):
        (mu1, cov1) = info1  # p_x
        (mu2, cov2) = info2  # p_g
        covSqrt = sp.linalg.sqrtm(np.matmul(cov1, cov2))
        if np.iscomplexobj(covSqrt):
            covSqrt = covSqrt.real
        fidScore = np.linalg.norm(mu1 - mu2) + np.trace(cov1 + cov2
                                                        - 2 * covSqrt)
        return fidScore

    def __call__(self, info):
        (string1, img2, info1) = info
        mu2 = img2.mean(axis=0)
        cov2 = np.cov(np.transpose(img2))
        score = self.fid(info1, (mu2, cov2))
        # print("For alpha = " + string1 + " the FID value is " + str(score))
        return score


def main():
    version = int(ver)
    subversion = int(subver)
    trial_num = int(trial_n)
    (trainIm, trainL), (_, _) = tf.keras.datasets.mnist.load_data()
    trainIm = trainIm.reshape(trainIm.shape[0], 28, 28, 1).astype('float32')
    trainIm = trainIm[np.random.choice(50000, 10000, replace=False), :, :, :]
    trainIm = trainIm.reshape(10000, 28 * 28).astype('float32')
    trainIm = trainIm / 255.0
    print(trainIm.shape)
    mu1 = trainIm.mean(axis=0)
    trainIm = np.transpose(trainIm)
    cov1 = np.cov(trainIm)
    info1 = (mu1, cov1)
    proc = Process()
    pool = multiprocessing.Pool(processes=16)
    while trial_num < trial_num + 1:
        print(trial_num)
        pFiles = []
        for epoch in range(250):
            p = np.load('data/annealing/v' + str(version) + '-' + str(subversion) + '/trial' + str(trial_num)
                        + '/predictions' + str(epoch) + '.npy')
            p = p.reshape(p.shape[1], 28, 28, 1).astype('float32')
            p = p[np.random.choice(50000, 10000, replace=False), :, :, :]
            p = p.reshape(10000, 28 * 28).astype('float32')
            p = (p * 127.5 + 127.5) / 255.0
            if np.isnan(p).any():
                break
            pFiles.append(('sim_ann_epoch' + str(epoch), p, info1))
        score_list = pool.map(proc, pFiles)
        np.save('data/annealing/v' + str(version) + '-' + str(subversion) + '/trial' + str(trial_num) + '/scores.npy', score_list)
        print(score_list)
        # If you are running low on space, uncomment the below code to automatically delete all
        # predictions.npy files except for the one that has the lowest FID score. 
        #for epoch in range(250):
        #    if epoch != np.nanargmin(score_list):
        #        os.remove('data/annealing/v' + str(version) + '-' + str(subversion) + '/trial' + str(trial_num)
        #                  + '/predictions' + str(epoch) + '.npy')
        trial_num = trial_num + 1


if __name__ == "__main__":
    ver, subver, trial_n = input("Version, subversion, trial_num: ").split()
    main()
