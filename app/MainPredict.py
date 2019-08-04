import ImgProcessing as IProcess
import ImgPredict_al as Al_Predict
import ImgPredict_num as Num_Predict
import numpy as np
import cv2
import requests
import json

class MainPredict:
    
    def __init__(self):
        self.alphabet_dic = {i:chr(c) for i,c in enumerate(range(ord('A'),ord('Z')+1))}
        self.alphabet_dic.update({i+26:chr(c) for i,c in enumerate(range(ord('a'),ord('z')+1))}
    
    def predict(self, path):
        alphabet = None
        numeral = None
        nlist = None
        result = []

        im_process = IProcess.ImgProcessing()
        im = cv2.imread(path)
        
        # resize, image processing
        resized_img = im_process.get_resized_img(im)
        rects, im_W = im_process.get_detected_texts_coordinates(resized_img)
        binarized_im = im_process.get_binarized_img(resized_img)
        
        if not rects:
            return ' READ ERROR... ' 
        unpacked_rects, word_pics = im_process.get_extract_line_im(rects, binarized_im, im_W)
        char_img = im_process.get_charImg(word_pics)
                
        al_list = self._predict_al(char_img)  
        num_list = self._predict_num(char_img) 
        
        result_text = []
        res = []
        json_txt = []
        result_text.extend(self._relative_probability(al_list, num_list))
        
        for s in result_text:
            res.append(str(s))
        result_text = "".join(res)
        result_text = result_text.replace(',', ' ')
        print(result_text), print(type(result_text))
        return result_text 
        
        
    def _predict_al(self, line):
        al_list = []
        im_predict_al = Al_Predict.ImgPredict_al()
        for txt in line:
            al_list.extend(im_predict_al.predict_al(txt)) 
        return al_list
        
        
    def _predict_num(self, line):
        num_list = []
        im_predict_num = Num_Predict.ImgPredict_num()
        for txt in line:
            num_list.extend(im_predict_num.predict_num(txt))
        return num_list
    
    
    def _relative_probability(self, al_list, num_list):
        result = []
        res_str = []
        alpha = self._get_gen_probability(al_list)
        numb = self._get_gen_probability(num_list)

        for al_i, al_pro in alpha: 
            num_i, num_pro = next(numb) 
            print("max"),print(al_pro),print(num_pro),print(num_i)

            if al_pro < 0.5 and num_pro < 0.5:
                result.append(" ")
            elif al_pro > num_pro or al_pro > 0.7:
                res = self.alphabet_dic.get(al_i)
                result.append(res)
                print(res)
            else:
                result.append(num_i)
                print(num_i)
        return result
                              
    def _get_gen_probability(self, list):
        for i, l in enumerate(list):
            print("l"), print(l)
            yield np.argmax(l), np.max(l)
            
        