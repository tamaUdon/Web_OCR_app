
import matplotlib.pyplot as plt
import io
import tempfile
import requests
import sys
import tkinter
import time
import numpy as np
import cv2
from imutils.object_detection import non_max_suppression
from sklearn.mixture import BayesianGaussianMixture
import math
import pandas as pd


class ImgProcessing:
    
    def get_resized_img(self, image):
        (H, W) = image.shape[:2]       
        (newW, newH) = (512, 512)
        rW = W / float(newW)
        rH = H / float(newH)
        
        # Image expand
        if rW or rH > 1:
            resized_img = cv2.resize(image, (newW, newH), interpolation = cv2.INTER_CUBIC)
        # Image reduction
        elif rW or rH < 1:
            resized_img = cv2.resize(image, (newW, newH), interpolation = cv2.cv2.INTER_AREA)
        else:
            resized_img = image
        cv2.imshow("resize", resized_img)
        return resized_img
    
    def get_binarized_img(self,resized_im):
        gray = cv2.cvtColor(resized_im, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 1, 21, 11) # NEED TO TUNE
        cv2.imwrite("alphabet-th.png", thresh)
        
        return thresh


    # EAST detector 
    # Reference from <https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/>
    def get_detected_texts_coordinates(self, resize):
        (H, W) = resize.shape[:2]
        orig = resize.copy()

        layerNames = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"]

        print("[INFO] loading EAST text detector...")
        net = cv2.dnn.readNet("frozen_east_text_detection.pb")

        blob = cv2.dnn.blobFromImage(resize, 1.0, (W, H),
            (123.68, 116.78, 103.94), swapRB=True, crop=False)
        start = time.time()
        net.setInput(blob)
        (scores, geometry) = net.forward(layerNames)
        end = time.time()

        print("[INFO] text detection took {:.6f} seconds".format(end - start))

        (numRows, numCols) = scores.shape[2:4]
        rects = []
        confidences = []
        coordinates = []
        
        for y in range(0, numRows):
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]
            min_confidence = 0.5 
            
            for x in range(0, numCols):
                if scoresData[x] < min_confidence: 
                    continue

                (offsetX, offsetY) = (x * 4.0, y * 4.0)
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)

                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]

                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x])) + 5
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x])) + 5
                startX = int(endX - w) - 10
                startY = int(endY - h) - 10
                
                rects.append((startX, startY, endX, endY)) 
                confidences.append(scoresData[x])
                
        boxes = non_max_suppression(np.array(rects), probs=confidences)
        
        # rectangle
        index_boxes = []
        for (startX, startY, endX, endY) in boxes:
            cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 1)
            index = endY * W + startX
            index_boxes.append([index, startX, startY, endX, endY])
            
        # sort
        index_boxes = sorted(index_boxes, key=lambda x:x[0])
        print("index"), print(index_boxes)

        cv2.imwrite("text.png", orig)
        return index_boxes, W
    
    
    # Extract words
    def get_extract_line_im(self, rects, binarized_im, im_w):
        trim_img = None
        line = []
        unpacked_rects = []
        word_pics = []

        for i, r in enumerate(rects):
            r = list(r)
            index, x, y, endX, endY = r
            h = endY - y
            w = endX - x
            
            lower_left_y = y + h
            right_corner_x = x + w
            y2 = round(lower_left_y / 10) * 10
            index = y2 * im_w  + x
            unpacked_rects.append([index, x,y,w,h, lower_left_y, right_corner_x])

            trim_img = binarized_im[y:y+h, x:x+w] # get ROI
            cv2.imwrite("save/dir/path"+ str(i)+"-trim.png", trim_img)

            ww = (round((w if w > h else h) * 1.2)).astype(np.int64)
            print(type(ww)), print(type(y))
            spc = np.zeros((ww, ww))
            wy = (ww-h)//2
            wx = (ww-w)//2
            spc[wy:wy+h, wx:wx+w] = trim_img # get normalized image 

            cv2.imwrite("save/dir/path"+ str(i)+"-textz.png", spc)
            word_pics.append(spc)
        
        print("unpacked_rects"), print(unpacked_rects)
        return unpacked_rects, word_pics
    
    
    # get each character
    def get_charImg(self, word_pics):
            char_imgs = []
            breakline = np.zeros((28, 28))
            breakline = breakline.reshape(-1, 28, 28, 1)
            breakline = breakline.astype("float32") / 255
            idx = 0
            for i, spc in enumerate(word_pics):
                
                # findcontours
                uint_im = np.uint8(spc)
                char_rects = cv2.findContours(uint_im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                
                # sort
                im_w = spc.shape[1]
                rcts = self._get_unpacked_rects(char_rects, im_w)
                line = sorted(rcts, key=lambda x:x[1])
                
                # integrate
                char_rects = self._get_integrated_rectangle(line)
                
                char_im = self._extract_chars_im(char_rects, spc)
                char_imgs.extend(char_im)
                
                # insert breakline
                char_imgs.append(breakline)
            
            return char_imgs
    
    
    def _get_unpacked_rects(self, char_rects, im_w):
        rct = []
        for rec in char_rects:
            x,y,w,h = cv2.boundingRect(rec)
            lower_left_y = y + h
            right_corner = x + w
            y2 = round(lower_left_y / 10) * 10
            index = y2 * im_w + x
            
            rct.append([index, x,y,w,h,lower_left_y, right_corner])
        print("rct"), print(rct)
        return rct
            
        
    # integrate split region (ex. 'i', 'j') 
    def _get_integrated_rectangle(self, corners):
        x_region = []
        del_pt = []
        pt_list = self._tuple_to_list(corners)
        
        for i,r in enumerate(pt_list):
            md = None
            index, x, y, w, h, lower_left_y, right_corner = r
            x_region.append(x+w/2)
            
            print("x_region"), print(x_region)
            if i > 0:
                md = x_region[i-1] 
                print("md"), print(md)
                if x <= md <= x+w:
                    del_pt.append(i)
                    
        if del_pt:
            print(del_pt)
            for dl in reversed(del_pt):   
                pt_list[dl][1] = max(pt_list[dl-1][1], pt_list[dl][1]) # x
                pt_list[dl][3] = max(pt_list[dl-1][3], pt_list[dl][3]) # w
                pt_list[dl][6] = pt_list[dl][1] + pt_list[dl][3] # x+w right_corner
                pt_list[dl][2] = min(pt_list[dl-1][2], pt_list[dl][2]) # y
                pt_list[dl][5] = max(pt_list[dl-1][5], pt_list[dl][5]) # lower_left_y
                pt_list[dl][4] = pt_list[dl][5] - pt_list[dl][2] # h
                del pt_list[dl-1]
            tuple(pt_list)
        return pt_list
    
    # tuple to list
    def _tuple_to_list(self, corners):
        l = []
        for i in corners:
            t_list = list(i)
            l.append(t_list)
        print("to_list"), print(l)
        return l

    # get each character image
    def _extract_chars_im(self, corners, word_pic):
        
        char_im = []
        corners = self._get_integrated_rectangle(corners)
        
        for i, cnt in enumerate(corners):
            index,x,y,w,h,lower_left_y,right_corner = cnt
            
            trim_img = word_pic[y:y+h, x:x+w] 
            ww = round((w if w > h else h) * 1.2)#.astype(np.int64)
            
            zeros = np.zeros((ww, ww))
            wy = (ww-h)//2
            wx = (ww-w)//2
            zeros[wy:wy+h, wx:wx+w] = trim_img
            
            # リサイズ、正規化
            trim_img = cv2.resize(zeros,(28, 28))
            cv2.imwrite("save/dir/path" + str(index)+"-normanilzed.png", trim_img)
            trim_img = trim_img.reshape(-1, 28, 28, 1)   
            trim_img = trim_img.astype("float32") / 255
            char_im.append(trim_img)
            
        return char_im
    

#### Use when you need to rotate image ####

#    def get_clusterized_coordinates(self,im):
#        
#        kp_array = np.empty((0,2), int)
#        
#        akaze = cv2.AKAZE_create() # AKAZE
#        keypoints = akaze.detect(self.thresh)
#        
#        for marker in keypoints:
#            akaze_im = cv2.drawMarker(im, 
#                                      tuple(int(i) for i in marker.pt),
#                                      color=(0, 255, 0))
#            cv2.imwrite("akaze.png", akaze_im)
#            kp_array = np.concatenate([kp_array, np.array([marker.pt])], 0)
#            
#        #VBGMM
#        bgmm = BayesianGaussianMixture(n_components=5, 
#                                       weight_concentration_prior_type='dirichlet_process')
#        bgmm = bgmm.fit(kp_array)
#        cluster = bgmm.predict(kp_array)
#        cluster_count = len(bgmm.weights_)
#        labeled_Pt_x = [[] for i in range(cluster_count)]
#        labeled_Pt_y = [[] for i in range(cluster_count)]
#        
#        marker_pt = []
#        cluster = cluster.tolist()
#        
#        for i in range(cluster_count):
#            for marker in keypoints:
#                marker_pt.append([int(n) for n in marker.pt if n != None])
#            labeled_Pt_x[i] = [marker_pt[n][0] for n,label in enumerate(cluster) if i == label]
#            labeled_Pt_y[i] = [marker_pt[n][1] for n,label in enumerate(cluster) if i == label]
#        
#        labeled_Pt_x = [l for l in labeled_Pt_x if l != None]
#        labeled_Pt_y = [l for l in labeled_Pt_y if l != None]
#        print(labeled_Pt_x)
#        print(labeled_Pt_y)
#        
#        return labeled_Pt_x, labeled_Pt_y
#
#
#    # rotate
#    def get_rotatedImg_rects(self,labeled_Pt_x,labeled_Pt_y,im):
#        
#        gradients = []
#        
#        for i in range(len(labeled_Pt_x)):
#            kp_x = np.array(labeled_Pt_x[i])
#            kp_y = np.array(labeled_Pt_y[i])
#            
#            ones = np.array([kp_x,np.ones(len(kp_x))])
#            ones = ones.T
#            a,b = np.linalg.lstsq(ones,kp_y)[0]
#            gradients.append(a)
#        #max_gradient = gradients[np.argmax(np.abs(gradients))]
#        max_gradient = gradients[int(np.mean(gradients))]
#        
#        rows,cols = self.thresh.shape
#        d = math.degrees(max_gradient)
#        print(d)
#        M = cv2.getRotationMatrix2D((cols/2,rows/2),d,1)
#        self.thresh = cv2.warpAffine(self.thresh,M,(cols,rows))
#        cv2.imwrite("rotate.png", self.thresh)
#
#        kernel = np.ones((6,6),np.uint8)
#        line_text_img = cv2.dilate(self.thresh, kernel, iterations=1)
#        cv2.imwrite("dilate.png", line_text_img)
#        
#        # 一行ずつ外接矩形を再認識
#        #rects = self._find_contours(line_text_img, im)
#        #rects = sorted(rects, key=lambda x:x[0])
#        
#        return rects

    
#### draw histgram
#    def get_hist_pt(self, hist_img, width, height, axis):
#        histgram = []
#        
#        if axis == 'x': 
#            for i in range(width):
#                px = hist_img[0:height, i]
#                #print("px"), print(px)
#                hist_n = [1 for x in px if (x == [255, 255, 255]).all()] #[1, 1, 1,...]
#                histgram.append(len(hist_n))
#                
#            plt.bar(range(width), histgram, width=1.0)
#            plt.ylim(0, height)
#            plt.show()
#            
#        elif axis == 'y':
#            for i in range(height):
#                px = hist_img[i, 0:width]
#                #print("px"), print(px)
#                hist_n = [1 for x in px if (x == [255, 255, 255]).all()] #[1, 1, 1,...]
#                histgram.append(len(hist_n))
#            plt.bar(range(height), histgram, width=1.0)
#            plt.ylim(0, width)
#            plt.show()
#        
#        x_pt = [i for i,x in enumerate(histgram) if x != 0]
#        y_pt = [i for i,y in enumerate(histgram) if y != 0]
        
#        pt_abs = [a for i,a in enumerate(x_pt) if i > 0 and abs(a - x_pt[i-1]) > 1]
#        print("abs")
#        print(pt_abs)
#        
#        return pt_abs



   