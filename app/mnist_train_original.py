
from sklearn.model_selection import cross_validate,train_test_split

import pandas as pd

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D as Conv2D, MaxPooling2D,Reshape
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
import tensorflow as tf

def build_model():
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(784, activation='relu'))
    model.add(Dense(52, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    return model

if __name__ == '__main__':
    main()