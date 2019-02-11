#!/usr/bin/python3

"""
Creates a cli with the user that predicts the language
    of the word inserted.
The programs executes on a while True until the user inserts
    an empty string
"""

import numpy as np
from sklearn import svm, linear_model

from constants import *

import pickle

def main():
    # Load the best model obtained
    with open("model.bin", "rb") as f:
        model = pickle.load(f)

    # Load the number of classes used on data
    with open("../../" + NUMBER_OF_CLASSES_PATH, "r") as f:
        NUMBER_OF_CLASSES = int(f.read())

    # Load the table of labels. Associates a language to an index
    label_table = [None] * NUMBER_OF_CLASSES
    with open("../../" + LABELS_PATH, "r") as f:
        for line in f:
            label, index = line.strip().split(":")
            index = int(index)
            label_table[index] = label

    while True:
        word = input("$ ").lower()

        if word == "":
            break
        elif len(word) > MAX_LENGTH_WORD:
            print("Max number of letter is " + str(MAX_LENGTH_WORD))

        wordArr = wordToMask(word)

        predictVect = np.zeros((1, BITS_NEEDED * MAX_LENGTH_WORD))
        for i, bit in enumerate(wordArr):
            predictVect[0, i] = bit

        prediction = model.predict(predictVect)

        print(label_table[int(prediction[0])])

    return 0

if __name__ == "__main__":
    main()
