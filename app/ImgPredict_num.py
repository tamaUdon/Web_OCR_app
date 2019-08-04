import mnist_train_original
import mnist_train_num
import numpy as np
import tensorflow as tf
from keras import backend as K

### CALL from MainPredict.py

class ImgPredict_num:
    
    def __init__(self):
        
        K.clear_session()
 
        self.numeral = mnist_train_num.build_model()
        self.numeral.load_weights('font_draw_num.hdf5', by_name=True)
        self.graph_num = tf.get_default_graph()
                               
    def predict_num(self, texts):
        num_list = None
        with self.graph_num.as_default():
            num_list = self.numeral.predict(np.array(texts))
        return num_list