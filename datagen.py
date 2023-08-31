import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from dataaug import SmilesEnumerator  # Assuming you have this library for data augmentation

smiles_dict = {'#': 1, '(': 2, ')': 3, '+': 4, '-': 5, '/': 6, '1': 7, '2': 8, '3': 9, '4': 10, '5': 11, '6': 12, '7': 13, '8': 14, '=': 15, 'C': 16, 'F': 17, 'H': 18, 'I': 19, 'N': 20, 'O': 21, 'P': 22, 'S': 23, '[': 24, '\\': 25, ']': 26, '_': 27, 'c': 28, 'Cl': 29, 'Br': 30, 'n': 31, 'o': 32, 's': 33, '@': 34, '.': 35, 'a': 36, 'B': 37, 'e': 38, 'i': 39, '9': 40, '10': 41, '11': 42}

#smiles_dict = {1: 'c', 2: 'C', 3: '(', 4: ')', 5: '1', 6: 'O', 7: '2', 8: 'N',  9: '=', 10: '3', 11: '[', 12: ']', 13: 'n', 14: '@', 15: '4', 16: 'H', 17: '-', 18: '/', 19: 'l', 20: '.', 21: '+', 22: 'F', 23: '5', 24: 'o',  25: 'B', 26: 'r', 27: 'S', 28: '\\', 29: '#', 30: 's', 31: '6', 32: 'I',  33: 'P', 34: 'e', 35: '7', 36: '8', 37: 'i', 38: 'a'}

def smiles_to_seq(smiles, seq_length, char_dict=smiles_dict):
    """ Tokenize characters in smiles to integers
    """
    smiles_len = len(smiles)
    seq = []
    keys = char_dict.keys()
    i = 0
    while i < smiles_len:
        # Skip all spaces
        if smiles[i:i + 1] == ' ':
            i = i + 1
        # For 'Cl', 'Br', etc.
        elif smiles[i:i + 2] in keys:
            seq.append(char_dict[smiles[i:i + 2]])
            i = i + 2
        elif smiles[i:i + 1] in keys:
            seq.append(char_dict[smiles[i:i + 1]])
            i = i + 1
        else:
            print(smiles)
            print(smiles[i:i + 1], i)
            raise ValueError('character not found in dict')
    for i in range(seq_length - len(seq)):
        # Padding with '_'
        seq.append(0)
    return seq

from dataaug import SmilesEnumerator
import numpy as np
import keras
import random

class DataGenerator(keras.utils.Sequence):
    def __init__(self, X, y, seq_length, batch_size=128, data_augmentation=True, shuffle=True):
        self.X = X
        self.y = y
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.data_augmentation = data_augmentation
        self.shuffle = shuffle
        self.sme = SmilesEnumerator()  # Instantiate SmilesEnumerator
        self.indexes = np.arange(len(self.X))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_X = [self.X[i] for i in batch_indexes]
        batch_y = [self.y[i] for i in batch_indexes]

        if self.data_augmentation:
            augmented_X = [self.augment_smiles(smiles) for smiles in batch_X]
            padded_X = self.pad_sequences(augmented_X)
        else:
            padded_X = self.pad_sequences(batch_X)
            
        #tokenize padded X
        padded_X = [smiles_to_seq(x, seq_length=self.seq_length, char_dict=smiles_dict) for x in padded_X]

        return np.array(padded_X), np.array(batch_y)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def augment_smiles(self, smiles):
        # Implement data augmentation using SmilesEnumerator
        augmented_smiles = self.sme.randomize_smiles(smiles)
        return augmented_smiles

    def pad_sequences(self, sequences):
        # Pad sequences to a uniform length
        padded_sequences = []
        for seq in sequences:
            if len(seq) < self.seq_length:
                padded_seq = seq + ' ' * (self.seq_length - len(seq))
            else:
                padded_seq = seq[:self.seq_length]
            padded_sequences.append(padded_seq)
        return padded_sequences

