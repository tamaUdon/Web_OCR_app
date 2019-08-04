
import mnist_train_original
import mnist_train_num
import numpy as np
import tensorflow as tf
from keras import backend as K

### CALLED from MainPredict.py

class ImgPredict_al:
    
    def __init__(self):
        K.clear_session()
        self.alphabet = mnist_train_original.build_model()
        self.alphabet.load_weights('font_draw_al.hdf5', by_name=True)
        self.graph_al = tf.get_default_graph()     
        
    def predict_al(self, texts): 
        al_list = None
        with self.graph_al.as_default():
            al_list = self.alphabet.predict(np.array(texts))
        return al_list