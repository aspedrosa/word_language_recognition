#!/usr/bin/python3

"""
Creates a cli with the user that predicts the language
    of the word inserted.
The programs executes on a while True until the user inserts
    an empty string
"""

from keras.models import Sequential
from keras.layers import Dense
import numpy as np

from constants import *

# Load the number of classes for the problem
with open("../../" + NUMBER_OF_CLASSES_PATH, "r") as f:
    NUMBER_OF_CLASSES = int(f.read())

# Load the best hyper parameters
with open("hyper_parameters", "r") as f:
    optimizer, structure = f.read().split(":")
    structure = [int(dim) for dim in structure.split(",")]

# Build the DNN loading the weights from the best model found
network = Sequential()
network.add(Dense(structure[0], input_dim=BITS_NEEDED * MAX_LENGTH_WORD, activation="sigmoid"))
for dim in structure[1:]:
    network.add(Dense(dim, activation="sigmoid"))
network.add(Dense(NUMBER_OF_CLASSES, activation="softmax"))
network.load_weights('weights.hdf5')
network.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# Load the table of labels (language:index)
label_table = [None] * NUMBER_OF_CLASSES
with open("../../" + LABELS_PATH, "r") as f:
    for line in f:
        line = line.strip()
        label, index = line.split(":")
        index = int(index)
        label_table[index] = label

while True:
    word = input("$ ").lower()

    if word == "":
        break
    elif len(word) > MAX_LENGTH_WORD:
        print("Max number of letters is " + str(MAX_LENGTH_WORD))
        continue

    wordArr = wordToMask(word)

    predictVect = np.zeros((1, BITS_NEEDED * MAX_LENGTH_WORD))
    for i, bit in enumerate(wordArr):
        predictVect[0, i] = bit

    prediction = network.predict(predictVect)

    score_board = []
    for i, score in enumerate(prediction[0]):
        score_board.append((label_table[i], score))

    for label, score in sorted(score_board, key=lambda elem : elem[1], reverse=True):
        print("{:10} : {}".format(label, float(score*100)))
