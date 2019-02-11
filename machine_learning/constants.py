#!/usr/bin/python3

"""
On this module it is stored the constants to where things are stored
    and some constants to that are assumed during the creation/train of the models
"""

# Locations
NON_PROCESSED_DATA_PATH = "data/non_processed_data" #where the lists of words are stored
PROCESSED_DATA_PATH     = "data/processed_data.npz" #where the dataset after preprocessing are stored
NUMBER_OF_CLASSES_PATH  = "data/number_of_classes"  #where the file with the number of calles is stored
                                                    #(created after preprocessing)
LABELS_PATH             = "data/labels"             #where the label association file is tored

# Contants
MIN_LENGTH_WORD = 4
MAX_LENGTH_WORD = 12
NUMBER_OF_CHARS = (ord("z") - ord("a") + 1) #26
from math import ceil, log2
BITS_NEEDED = ceil(log2(NUMBER_OF_CHARS))
EPOCHS = 400 # DNN epochs for training

def wordToMask(word):
    """
    Function to transform word into an array (sequence of 0's and 1's)
    """
    mask = []
    inserted = 0
    for char in word:
        inserted += 1
        i = ord(char) - 97 + 1 #get the index of the letter
                               #on the ascii alphabet 'a' as the value of 97
                               #  so this will give me the index of the letter 
                               #  on the alphabet

        mask += [int(b) for b in format(i, "0" + str(BITS_NEEDED) + "b")]
        #concatenate the array of this letter to the array of the word

    return mask + ([0] * BITS_NEEDED) * (MAX_LENGTH_WORD - inserted)
                  #add padding to the end according to the number of missing letters
