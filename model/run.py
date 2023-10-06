
from os import listdir, makedirs, remove, environ
from os.path import isfile, join, exists, splitext, abspath, basename
import glob
import numpy as np
from pathlib import Path
import csv
import shutil
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation, Rescaling
from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.python.framework.config import set_visible_devices, list_physical_devices
from tensorflow.keras import layers
from tensorflow import device
from tensorflow.distribute import MirroredStrategy, HierarchicalCopyAllReduce
import pickle
from importlib import import_module
from time import sleep
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from os import listdir, getcwd, makedirs, chdir, curdir
from os.path import join, basename, isdir, exists
from gc import collect


# change the path of these folders accordingly for your working environment
folder_train = r"D:\calisma\projeler\minyatür\_calisma_\dataset\egitim_seti"
test_path = r"D:\calisma\projeler\minyatür\_calisma_\dataset\test_seti"

modul = import_module('cnn')


# combinations of some hyper-parameters
params = {'lrate': {1e-4, 1e-3 }, 'max_pooling': (True, False),
     'batch_normalization': ( False, ),
     'optimizer': ('Adamax', 'Nadam', 'Adadelta', 'Adagrad', 'Adam', 'RMSPROP', 'SGD'),
     'activation': ('relu', 'selu', 'prelu', 'tanh', 'sigmoid'),
     }


lrates = (1e-3, 1e-4)
max_poolings = (True, True)
batch_normalizations = (True, False )
# optimizers =('SGD', 'Adam', 'RMSprop', )
optimizers =( 'Adam', 'RMSprop', )
activations= ('relu', 'elu', 'selu', 'exponential', 'tanh', 'sigmoid')
batch_size = 16

name=modul.__name__


model = modul.My_Model(name='temp', training_path=folder_train, test_path=test_path)
model.batch_size= batch_size

train_ds = image_dataset_from_directory(
    model.training_path,
    validation_split=model.val_split,
    seed=model.seed,
    subset="training",
    image_size=model.input_shape[0:2],
    label_mode='categorical',
    batch_size=model.batch_size,
)

val_ds = image_dataset_from_directory(
    model.training_path,
    validation_split=model.val_split,
    seed=model.seed,
    subset="validation",
    image_size=model.input_shape[0:2],
    label_mode='categorical',
    batch_size=model.batch_size,
)

test_ds = image_dataset_from_directory(
    test_path,
    validation_split=None,
    seed=model.seed,
    # subset="test",
    image_size=model.input_shape[0:2],
    label_mode='categorical',
    batch_size=model.test_batch_size,
)

df_train = pd.DataFrame()
df_test = pd.DataFrame()
first = True

for optimizer in optimizers:
    name1 = name + '_' + optimizer
    
    for lrate in lrates:
        name2 = name1 + '_lr_' + str(lrate)

        for activation in activations:
            name3 = name2 + '_' + activation

            for max_pooling in max_poolings:
                name4 = name3 + '_mp_' + str(max_pooling)

                for batch_normalization in batch_normalizations:
                    name5 = name4 + '__bn_' + str(batch_normalization)

                    model = modul.My_Model(name=name5, training_path=folder_train, test_path=test_path)
                    model.epochs = 1000
                    model.lrate = lrate
                    model.optimizer = globals()[optimizer](learning_rate=model.lrate)
                    model.activation = activation
                    model.max_pooling = max_pooling
                    model.normalize_batch = batch_normalization
                    
                    model.verbose=2
                    # model.set_folders()
                    
                    m, t = model.__train__(train_ds=train_ds, val_ds=val_ds)
                    training_times = np.repeat(t, len(m.history['loss']))

                    sleep(1)
                    collect()
                    model = modul.My_Model(name=name5, training_path=folder_train, test_path=test_path)
                    sleep(3)
                    test = model.__test__(test_ds=test_ds, )
                    
                    cols = list(model.history.history.keys())
                    cols.append('training_time')

                    scores = list(model.history.history.values()) 
                    scores.append(training_times)
                    mux = pd.MultiIndex.from_product([[model.name], cols], names=['model', 'score'])
                    train = pd.DataFrame(np.transpose(scores), columns=mux)

                    if first:
                        df_train = train
                        df_test = test
                        first = False
                    else:
                        df_train = pd.concat([df_train, train], axis=1)    
                        df_test = pd.concat([df_test, test], axis=0)
                        
                    
                    df_train.to_excel(join(getcwd(), r'output\training.xlsx'))
                    df_test.to_excel(join(getcwd(), r'output\test.xlsx'))

a = 5


