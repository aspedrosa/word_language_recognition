#!/usr/bin/python3

"""
On this file we train several model of DNN's having
    different optimizers and differente structures with different
    number of layers and number of units per layer. This
    train is done with the train data
At the end of this we retrain the best model found with
    both train and validation data
"""

from constants import *

from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import optimizers

import numpy as np

import os

def main():
    with open("../" + PROCESSED_DATA_PATH, "rb") as f:
        with np.load(f) as npz:
            xTrain      = npz[npz.files[0]]
            yTrain      = npz[npz.files[1]]
            xValidation = npz[npz.files[3]]
            yValidation = npz[npz.files[4]]
            xTest       = npz[npz.files[6]]
            yTest       = npz[npz.files[7]]

    with open("../" + NUMBER_OF_CLASSES_PATH, "r") as f:
        NUMBER_OF_CLASSES = int(f.read())

    ############################################vv Hyper parameters
    structures = [
            [200, 150, 100, 100],
            [200, 100, 50],
            [200, 100],
            [100, 50,  25,  25],
            [100, 50,  25],
            [100, 50]
            ]

    optimizers = [
            "sgd",
            "rmsprop",
            "adagrad",
            "adadelta",
            "adam",
            "adamax",
            "nadam"
            ]
    ###########################################^^ Hyper parameters

    if not os.path.exists("models"):
        os.mkdir("models")

    max_score = float("-inf")
    best_structure = []
    best_optimizer = ""

    results_file = open("results.txt", "w+")

    ##########################################vv Cross validation
    print("started cross validation")
    count = 1
    for optimizer in optimizers:

        if not os.path.exists("models/" + optimizer):
            os.mkdir("models/" + optimizer)

        results_file.write(optimizer.upper() + "\n")

        for structure in structures:
            print(count, "of", len(optimizer) + len(structures))

            network = Sequential()
            network.add(Dense(structure[0], input_dim=BITS_NEEDED * MAX_LENGTH_WORD, activation="sigmoid"))
            for dim in structure[1:]:
                network.add(Dense(dim, activation="sigmoid"))
            network.add(Dense(NUMBER_OF_CLASSES, activation="softmax"))
            network.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

            structureStr = "_".join([str(dim) for dim in structure])

            checkpoint = ModelCheckpoint(
                    './models/' + optimizer + "/" + structureStr + ".hdf5",
                    monitor='val_acc',
                    verbose=1,
                    save_best_only=True,
                    mode='max')
            tboard = TensorBoard(
                    log_dir='./logs/' + optimizer + "/" + structureStr,
                    write_graph=True,
                    write_images=True)
            callbacks_list = [checkpoint, tboard]

            history = network.fit(
                    xTrain,
                    yTrain,
                    epochs=EPOCHS,
                    validation_data=(xValidation, yValidation),
                    verbose=0,
                    callbacks=callbacks_list)

            validation_score = max(history.history["val_acc"])
            if validation_score > max_validation_score:
                best_structure = structure
                best_optimizer = optimizer
                max_validation_score = validation_score

            results_file.write("{:15} - {}\n".format(structureStr, validation_score))

            count += 1
    ################################################################################^^ Cross validation

    ################################################################################vv Training of the final model
    network = Sequential()
    network.add(Dense(best_structure[0], input_dim=BITS_NEEDED * MAX_LENGTH_WORD, activation="sigmoid"))
    for dim in best_structure[1:]:
        network.add(Dense(dim, activation="sigmoid"))
    network.add(Dense(NUMBER_OF_CLASSES, activation="softmax"))
    network.load_weights("models/" + best_optimizer + "/" "_".join([str(dim) for dim in best_structure]) + ".hdf5")
    network.compile(loss="binary_crossentropy", optimizer=best_optimizer, metrics=["accuracy"])

    checkpoint = ModelCheckpoint(
            './models/final.hdf5',
            monitor='val_acc',
            verbose=1,
            save_best_only=True,
            mode='max')
    tboard = TensorBoard(
            log_dir='./logs/final',
            write_graph=True,
            write_images=True)
    callbacks_list = [checkpoint, tboard]

    history = network.fit(
            np.concatenate((xTrain, xValidation)),
            np.concatenate((yTrain, yValidation)),
            epochs=EPOCHS,
            validation_data=(xTest, yTest),
            verbose=0,
            callbacks=callbacks_list)
    #################################################################################^^ Training of the final model

    results_file.write("FINAL\n")
    results_file.write(max(history.history["val_acc"]))
    results_file.close()

    with open("../../predict/dnn/hyper_parameters", "w+") as f:
        f.write(best_optimizer + ":" + str(best_structure)[1:-1])

    os.system("cp models/final.hdf5 ../../predict/dnn/weights.hdf5")

    return 0

if __name__ == "__main__":
    main()
