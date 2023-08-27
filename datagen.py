import numpy as np
import tensorflow.keras as keras


smiles_dict = {'#': 1, '(': 2, ')': 3, '+': 4, '-': 5, '/': 6, '1': 7, '2': 8, '3': 9, '4': 10, '5': 11, '6': 12, '7': 13, '8': 14, '=': 15, 'C': 16, 'F': 17, 'H': 18, 'I': 19, 'N': 20, 'O': 21, 'P': 22, 'S': 23, '[': 24, '\\': 25, ']': 26, '_': 27, 'c': 28, 'Cl': 29, 'Br': 30, 'n': 31, 'o': 32, 's': 33, '@': 34, '.': 35, 'a': 36, 'B': 37, 'e': 38, 'i': 39, '9': 40, '10': 41, '11': 42}


# Pasa 
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

#class DataGenerator(keras.utils.Sequence):
#    def __init__(self, X, y, seq_length, batch_size=128, data_augmentation=True, shuffle=True):
#        # Agregar aca todas las propiedades necesarias para resolver el problema
#        # No olvidar la aumentación de datos
#        self.on_epoch_end()
#
#
#    def __len__(self):
#        'Denotes the number of batches per epoch'
#        return int(np.ceil(len(self.X) / self.batch_size))
#
#    def __getitem__(self, index):
#        'Generate one batch of data'
#        # Generate indexes of the batch
#        indexes = # Implementar
#
#        # Generate data
#        if self.data_augmentation:
#            # Implementar
#        else:
#            # Implementar
#        y = # Implementar
#        return X, y
#
#    def on_epoch_end(self):
#        'Updates indexes after each epoch'
#        self.indexes = np.arange(len(self.X))
#        if self.shuffle == True:
#            # Implementar

import numpy as np
import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np
import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

class DataGenerator(keras.utils.Sequence):
    def __init__(self, X, y, seq_length, batch_size=128, data_augmentation=True, shuffle=True):
        super(DataGenerator, self).__init__()  # Initialize the superclass
        self.X = X
        self.y = y
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.data_augmentation = data_augmentation
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # Generate data
        batch_X = self.X[indexes]
        if self.data_augmentation:
            batch_X = self.apply_data_augmentation(batch_X)
        else:
            batch_X = pad_sequences(batch_X, maxlen=self.seq_length, padding='post')

        batch_y = self.y[indexes]

        return batch_X, batch_y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.X))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def apply_data_augmentation(self, batch_X):
        # Placeholder for data augmentation logic
        augmented_data = []
        for x in batch_X:
            augmented_data.append(x)  # Implement actual augmentation here
        return np.array(augmented_data)

