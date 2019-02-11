#!/usr/bin/python3

"""
On this file we handle the first phase of the training of
    SVM models. The training is done in separate threads
    in a way to increase the speed of the results
"""

import numpy as np
from sklearn import svm, linear_model

from constants import *

import threading

import pickle

# Shared variables to keep track of the best model parameters
best_model = best_c = best_kernel = best_gamma = best_coef0 = None
best_score = float("-inf")

class Worker(threading.Thread):
    """
    Thread to handle the training of a specific hyper parameters
    """
    def __init__(self, mutex, wait, xTrain, yTrain, xValidation, yValidation, C, kernel, gamma="auto", coef0=0):
        super(Worker, self).__init__()
        self.mutex = mutex
        self.wait = wait
        self.xTrain = xTrain
        self.yTrain = yTrain
        self.xValidation = xValidation
        self.yValidation = yValidation
        self.C = C
        self.kernel = kernel
        self.gamma=gamma
        self.coef0 = coef0

    def run(self):
        print("launched {}_{}_{}_{}".format(self.kernel, self.C, self.gamma, self.coef0))
        clf = svm.SVC(C=self.C, gamma=self.gamma, kernel=self.kernel, coef0=self.coef0)
        clf.fit(self.xTrain, self.yTrain)

        score = clf.score(self.xValidation, self.yValidation)

        self.mutex.acquire()
        with open("results.txt", "a+") as f:
            f.write("{}_{}_{}_{} : {}\n".format(self.kernel, self.C, self.gamma, self.coef0, score))

        # If a better model was found save it
        if score > best_score:
            best_model = clf
            best_c = self.c
            best_kernel = self.kernel
            best_gamma = self.gamma
            best_coef0 = self.coef0
            best_score = score
        self.mutex.release()

        print("end of {}_{}_{}_{}".format(self.kernel, self.C, self.gamma, self.coef0))

        self.wait.release()

def main():
    with open("../../" + PROCESSED_DATA_PATH, "rb") as f:
        with np.load(f) as npz:
            xTrain      = npz[npz.files[0]]
            yTrain      = npz[npz.files[2]]
            xValidation = npz[npz.files[3]]
            yValidation = npz[npz.files[5]]
            xTest       = npz[npz.files[6]]
            yTest       = npz[npz.files[8]]

    write_mutex = threading.Semaphore()

    threads = []

    #Combination of the several hyper parameters
    for C in [0.001, 1, 30]:
        threads.append(
            Worker(write_mutex, thread_wait, xTrain, yTrain, xValidation, yValidation, C, "linear"),
                )
        for gamma in ["auto", "scale"]:
            Worker(write_mutex, thread_wait, xTrain, yTrain, xValidation, yValidation, C, "rbf", gamma),
            for coef0 in [-25, 0, 25]:
                thread.append(
                    Worker(write_mutex, thread_wait, xTrain, yTrain, xValidation, yValidation, C, "sigmoid", gamma, coef0),
                    )

    thread_wait = threading.Semaphore(4) #Only launch 4 threads at the time (Quad-core processors)
    for thread in threads:
        thread_wait.acquire()
        thread.start()

    for thread in threads:
        thread.join()

    # save parameters and model itself of the best model found
    with open("best_model_first_phase.bin", "wb+") as f:
        pickle.dump(best_model, f)
    with open("best_parameters_first_phase", "w+") as f:
        f.write(str(best_score) + "\n")
        f.write(str(best_c) + "\n")
        f.write(best_kernel + "\n")
        f.write(best_gamma + "\n")
        f.write(str(best_coef0) + "\n")

    return 0

if __name__ == "__main__":
    main()
