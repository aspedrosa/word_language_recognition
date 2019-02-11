#!/usr/bin/python3

"""
On this file a preprocessing to the list of words is done
-from all the words on a list of a specific language only the words
    with less than MAX_LENGTH_WORD (defined on contants.py) are ignored;
-non ascii caracter are transformed into ascii
-all words are put to lower case
-it is retreived from each language the same amount of words
-each set of words extracted are split into three sets, train 60%, validation
    20% and test 20%.
-all this small sets are merged into three big ones for train, validation and test
"""

import os
from math import ceil
from unidecode import unidecode

import numpy as np

from constants import *

def retrieveWords(higherNumber, words, X, Y_dnn, Y_svm, label_dnn, label_svm):
    """
    Extracts from the list 'words' the number of words equal to higherNumber.
    Inserts each word on the X matrix transforming it using the wordToMask function.
    Also fills the vectors Y with the labels received
    """
    for wordNumber in np.linspace(len(words), 1, higherNumber):
        wordNumber = int(wordNumber) - 1
        word = words[wordNumber]
        X.append(wordToMask(word))
        Y_dnn.append(label_dnn)
        Y_svm.append(label_svm)
        del words[wordNumber]

def main():
    words = dict()
    non_processed_data_path = "../" + NON_PROCESSED_DATA_PATH
    for entry in os.scandir(non_processed_data_path): #go over each list of words
        words[entry.name] = []
        with open(non_processed_data_path + "/" + entry.name, "rb") as f: #open the file
            for line in f:
                l = line.decode("iso-8859-1") #decode the content into the iso-8859-1 encoding
                                              #notice that we are reading as a binary file ("rb" argument)
                word = l.strip() #take out the new line
                word = unidecode(word) #transform non ascii characters
                if MIN_LENGTH_WORD <= len(word) <= MAX_LENGTH_WORD \
                and "." not in word \
                and "/" not in word \
                and " " not in word \
                and "'" not in word \
                and "-" not in word:
                        #ignore words out of the range of
                        #[MIN_LENGTH_WORD-MAX_LENGTH_WORD]
                        # and also
                        #filter those words with strange characters (. /, ...)
                    words[entry.name].append(word.lower())

    number_of_classes = len(words.keys())
    with open("../" + NUMBER_OF_CLASSES_PATH, "w+") as f:
        f.write(str(number_of_classes))

    minLines = min([len(l) for l in words.values()]) #calculate the minimum of words per language

    #number of examples per set
    trainN = ceil(minLines * 0.6)
    validationN = ceil(minLines * 0.2)
    testN = minLines - trainN - validationN

    #A separation between dnn and svm output is done because
    #  they have different outputs.
    #DNN outputs a column and the SVM outputs scalar
    xTrain = []
    yTrain_dnn = []
    yTrain_svm = []

    xValidation = []
    yValidation_dnn = []
    yValidation_svm = []

    xTest = []
    yTest_dnn = []
    yTest_svm = []

    labels = open("../" + LABELS_PATH, "w+")

    for i, (dictName, words) in enumerate(words.items()):
        dictName = dictName[:-4]
        label = [0 for _ in range(i)] + [1] + [0 for _ in range(number_of_classes - i - 1)]
        labels.write(dictName + ":" + str(i) + "\n")
        print(dictName + ":" + str(i))

        words.sort() #sort the words of a language alphabetically

        beginLines = len(words)

        retrieveWords(trainN, words, xTrain, yTrain_dnn, yTrain_svm, label, i)

        retrieveWords(validationN, words, xValidation, yValidation_dnn, yValidation_svm, label, i)

        if beginLines == minLines: #used by the language that has the minimum of words.
                                   #if this wasn't here the test data of that language
                                   #  would be smaller than the others
            for word in words:
                xTest.append(wordToMask(word))
                yTest_dnn.append(label)
                yTest_svm.append(label)
            continue

        retrieveWords(testN, words, xTest, yTest_dnn, yTest_svm, label, i)

    labels.close()

    words = None

    #Transform the built arrays to numpy arrays
    xTrain          = np.array(xTrain)
    yTrain_dnn      = np.array(yTrain_dnn)
    yTrain_svm      = np.array(yTrain_svm)

    xValidation     = np.array(xValidation)
    yValidation_dnn = np.array(yValidation_dnn)
    yValidation_svm = np.array(yValidation_svm)

    xTest           = np.array(xTest)
    yTest_dnn       = np.array(yTest_dnn)
    yTest_svm       = np.array(yTest_svm)

    #Save the numpy arrrays into one file
    np.savez("../" + PROCESSED_DATA_PATH , xTrain, yTrain_dnn, yTrain_svm,
                                           xValidation, yValidation_dnn, yValidation_svm,
                                           xTest, yTest_dnn, yTest_svm)

    return 0

if __name__ == "__main__":
    main()
