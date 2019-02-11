#!/usr/bin/python3

"""
On this file we do the second phase of the training of
    the SVM models where we do a more deep seach of the
    hyperparameter for a specific kernel.
"""

import numpy as np
from sklearn import svm, linear_model

from constants import *

import threading

import pickle

best_model = best_c = best_kernel = best_gamma = best_coef0 = None
best_score = float("-inf")

# Load data from the last phase
with open("best_parameters_first_phase", "r") as f:
    best_score = float(f.readline().strip())
with open("best_model_first_phase.bin", "rb") as f:
    best_model = pickle.load(f)

class Worker(threading.Thread):
    """
    Thread to handle the train for specific parameters
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

    for C in [0.01, 0.01, 10, 20]:
        for gamma in ["auto", "scale"]:
            Worker(write_mutex, thread_wait, xTrain, yTrain, xValidation, yValidation, C, "rbf", gamma),

    thread_wait = threading.Semaphore(4) #Only launch 4 threads at the time
    for thread in threads:
        thread_wait.acquire()
        thread.start()

    for thread in threads:
        thread.join()

    # Train the final model with train and validation data
    best_model.fit(np.concatenate((xTrain, xValidation)), np.concatenate((yTrain, yValidation)))
    score = best_model.score(xTest, yTest)

    with open("results.txt", "a+") as f:
        f.write("FINAL RBF " + best_gamma.upper() + " " + str(best_c) + "\n")
        f.write(str(score) + "\n")

    with open("../../predict/svm/model.bin", "wb+") as f:
        pickle.dump(best_model, f)

    return 0

if __name__ == "__main__":
    main()
