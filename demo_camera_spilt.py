#!/usr/bin/env python
# -*- coding: utf-8 -*-
from rknn.api import RKNN
#import _init_paths
import cv2
import numpy as np
import math
import threading
from time import sleep
imreadLock = threading.Lock()
#from python_wrapper import *
#import os
import sys
PNET_PYRAMID= np.array([[270,207],[192,147],[136,104],[97,74],[69,53],[49,37],[35,27],[25,19],[18,14]])
ODD_PYRAMID = np.array([[270,207],[136,104],[69,53],[35,27],[18,14]])
ODD_SCALES = np.array([0.6,0.302,0.153,0.0778,0.04])
EVEN_PYRAMID = np.array([[192,147],[97,74],[49,37],[25,19]])
EVEN_SCALES = np.array([0.4267,0.2156,0.1089,0.0556])
#PNET_PYRAMID_ARR= np.array([[[1, 2, 130, 99],[1, 4, 130, 99]],[[1, 2, 91, 69],[1, 4, 91, 69]],[[1, 2, 63, 47],[1, 4, 63, 47]],[[1, 2, 44, 32],[1, 4, 44, 32]],[[1, 2, 30, 22],[1, 4, 30, 22]],[[1, 2, 20, 14],[1, 4, 20, 14]],[[1, 2, 13, 9],[1, 4, 13, 9]],[[1, 2, 8, 5],[1, 4, 8, 5]],[[1, 2, 4, 2],[1, 4, 4, 2]]])
PNET_ODD_ARR = np.array([[1, 2, 198, 99],[1, 4, 198, 99]])
PNET_EVEN_ARR = np.array([[1, 2, 140, 69],[1, 4, 140, 69]])
boundingbox_list = []
IMAGE = np.zeros((344,450,3))
IMAGE_list = []
IMAGE_list.append(IMAGE)
IMAGE_list.append(IMAGE)
proflag = 1

#----------------------------------------------多线程类

class MtcnnThread (threading.Thread):   #继承父类threading.Thread
    def __init__(self):
        threading.Thread.__init__(self)
    def run(self):                   #把要执行的代码写到run函数里面 线程在创建后会直接运行run函数
        minsize = 20
        threshold = [0.8, 0.9, 0.95]
        factor = 0.709
        pnet_rknn_list=init_pnet()
        rnet_rknn = RKNN()
        onet_rknn = RKNN()
        rnet_rknn.load_rknn('./RNet.rknn')
        onet_rknn.load_rknn('./ONet.rknn')
        ret = rnet_rknn.init_runtime()
        if ret != 0:
        #print('Init rnet runtime environment failed')
            exit(ret)
        ret = onet_rknn.init_runtime()
        if ret != 0:
        #print('Init onet runtime environment failed')
            exit(ret)
        sys.stdout = open('/dev/stdout', 'w')
        sys.stderr = open('/dev/stderr', 'w')
        global proflag
        global IMAGE_list
        global boundingbox_list
        nonfacecount = 0
        #wrongimg = 1
        while(proflag ==1):
            imreadLock.acquire()
            img0 = IMAGE_list[0].copy()
            img = IMAGE_list[1].copy()
            imreadLock.release()
            #tic()
            score_cmp = compare_image(img0,img)
            #print("score_cmp",score_cmp)
            #toc()
            if score_cmp < 0.98:
            #imreadLock.release()
            #print("detect face start")
            #cv2.imwrite("aa.jpg",img)
                tic()
                boundingboxes, points = detect_face(img, minsize, pnet_rknn_list, rnet_rknn, onet_rknn, threshold, False,factor)
            #print("boundingboxes shape",boundingboxes.shape)
                print("total cost")
                toc()
                if boundingboxes.shape[0] != 0:
                    if len(boundingbox_list) != 0:
                        boundingbox_list.clear()
                    boundingbox_list.append(boundingboxes)
                else:
                #path = str(wrongimg)+".jpg"
                #cv2.imwrite(path,img) 
                #wrongimg += 1
                    nonfacecount += 1
                    if nonfacecount >= 3:
                        boundingbox_list.clear()
                        nonfacecount = 0
        for i in range(2):
            pnet_rknn_list[i].release()
        rnet_rknn.release()
        onet_rknn.release()
    '''
    def return_res(self):
        try:
            return self.boundingboxes
        except Exception:
            return None
    '''
'''
class MtcnnThread(threading.Thread):
    def __init__(self, target, args):
        super().__init__()
        self.target = target
        self.args = args

    def run(self):
        self.target(*self.args)
'''
class ShowImgThread (threading.Thread):   #继承父类threading.Thread
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):                   #把要执行的代码写到run函数里面 线程在创建后会直接运行run函数 \
        global proflag
        global IMAGE_list
        global boundingbox_list
        proflag = 1
        capture = cv2.VideoCapture(0)
        while (True):
        #tic()
            ret, img = capture.read()
            if ret == True:
            #img = cv2.imread('./test4.jpg')
                img = cv2.resize(img,(450,344))
                #print("img shape",img.shape)
                image = img.copy()
                del IMAGE_list[0]
                IMAGE_list.append(image)
                #print(IMAGE)
                #IMAGE = cv2.cvtColor(IMAGE, cv2.COLOR_BGR2RGB)
                if len(boundingbox_list) != 0:
                    imreadLock.acquire()
                    img = drawBoxes(img, boundingbox_list[0])
                    imreadLock.release()
                cv2.imshow('img', img)
                c = cv2.waitKey(1) & 0xff
                if c==27:
                    break
            #toc()
            #print("imshow--------------------")
	        #if boundingboxes.shape[0] > 0:
            #    error.append[imgpath]
            #print(error)
        proflag = 0
        cv2.destroyAllWindows()                
        capture.release()  

def compare_image(image1, image2):  #图片对比函数
    
    grayA = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    grayB = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
    diff = np.sum((grayA-grayB)**2)
    score = 1-diff/450/344/255
    return score




def bbreg(boundingbox, reg):
    reg = reg.T 
    
    # calibrate bouding boxes
    if reg.shape[1] == 1:
        #print("reshape of reg")
        pass # reshape of reg
    w = boundingbox[:,2] - boundingbox[:,0] + 1
    h = boundingbox[:,3] - boundingbox[:,1] + 1

    bb0 = boundingbox[:,0] + reg[:,0]*w
    bb1 = boundingbox[:,1] + reg[:,1]*h
    bb2 = boundingbox[:,2] + reg[:,2]*w
    bb3 = boundingbox[:,3] + reg[:,3]*h
    
    boundingbox[:,0:4] = np.array([bb0, bb1, bb2, bb3]).T
    #print("bb", boundingbox)
    return boundingbox


def pad(boxesA, w, h):
    boxes = boxesA.copy() # shit, value parameter!!!
    #print('#################')
    #print('boxes', boxes)
    #print('w,h', w, h)
    
    tmph = boxes[:,3] - boxes[:,1] + 1
    tmpw = boxes[:,2] - boxes[:,0] + 1
    numbox = boxes.shape[0]

    #print('tmph', tmph)
    #print('tmpw', tmpw)

    dx = np.ones(numbox)
    dy = np.ones(numbox)
    edx = tmpw 
    edy = tmph

    x = boxes[:,0:1][:,0]
    y = boxes[:,1:2][:,0]
    ex = boxes[:,2:3][:,0]
    ey = boxes[:,3:4][:,0]
   
   
    tmp = np.where(ex > w)[0]
    if tmp.shape[0] != 0:
        edx[tmp] = -ex[tmp] + w-1 + tmpw[tmp]
        ex[tmp] = w-1

    tmp = np.where(ey > h)[0]
    if tmp.shape[0] != 0:
        edy[tmp] = -ey[tmp] + h-1 + tmph[tmp]
        ey[tmp] = h-1

    tmp = np.where(x < 1)[0]
    if tmp.shape[0] != 0:
        dx[tmp] = 2 - x[tmp]
        x[tmp] = np.ones_like(x[tmp])

    tmp = np.where(y < 1)[0]
    if tmp.shape[0] != 0:
        dy[tmp] = 2 - y[tmp]
        y[tmp] = np.ones_like(y[tmp])
    
    # for python index from 0, while matlab from 1
    dy = np.maximum(0, dy-1)
    dx = np.maximum(0, dx-1)
    y = np.maximum(0, y-1)
    x = np.maximum(0, x-1)
    edy = np.maximum(0, edy-1)
    edx = np.maximum(0, edx-1)
    ey = np.maximum(0, ey-1)
    ex = np.maximum(0, ex-1)
    
    #print("dy"  ,dy )
    #print("dx"  ,dx )
    #print("y "  ,y )
    #print("x "  ,x )
    #print("edy" ,edy)
    #print("edx" ,edx)
    #print("ey"  ,ey )
    #print("ex"  ,ex )


    #print('boxes', boxes)
    return [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]



def rerec(bboxA):
    # convert bboxA to square
    w = bboxA[:,2] - bboxA[:,0]
    h = bboxA[:,3] - bboxA[:,1]
    l = np.maximum(w,h).T
    
    #print('bboxA', bboxA)
    #print('w', w)
    #print('h', h)
    #print('l', l)
    bboxA[:,0] = bboxA[:,0] + w*0.5 - l*0.5
    bboxA[:,1] = bboxA[:,1] + h*0.5 - l*0.5 
    bboxA[:,2:4] = bboxA[:,0:2] + np.repeat([l], 2, axis = 0).T 
    return bboxA


def nms(boxes, threshold, type):
    """nms
    :boxes: [:,0:5]
    :threshold: 0.5 like
    :type: 'Min' or others
    :returns: TODO
    """
    if boxes.shape[0] == 0:
        return np.array([])
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    s = boxes[:,4]
    area = np.multiply(x2-x1+1, y2-y1+1)
    I = np.array(s.argsort()) # read s using I
    
    pick = [];
    while len(I) > 0:
        xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]])
        yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])
        xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])
        yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if type == 'Min':
            o = inter / np.minimum(area[I[-1]], area[I[0:-1]])
        else:
            o = inter / (area[I[-1]] + area[I[0:-1]] - inter)
        pick.append(I[-1])
        I = I[np.where( o <= threshold)[0]]
    return pick


def generateBoundingBox(map_prob, reg, scale, pyramid, threshold):
    stride = 2
    cellsize = 12
    map_prob = map_prob.T
    #print("map_prob shape",map_prob.shape)
    dx1 = reg[0,:,:].T
    dy1 = reg[1,:,:].T
    dx2 = reg[2,:,:].T
    dy2 = reg[3,:,:].T
    (x, y) = np.where(map_prob >= threshold)
    #print("x:",x)
    #print("y:",y)
    yy = y.copy()
    xx = x.copy()
        
    lenface = len(yy)
    #print("lenface:",lenface)
    lenpy = len(pyramid)
    w0 = pyramid[0][0]
    #print("w0:",w0)
    #print("(w0-6)/2+1:",(w0-6)/2+1)
    pyramid_reverse = []
    htmp = 0
    for i in range(1,lenpy):  
        htmp += pyramid[i][1]
        pyramid_reverse.append(htmp)
    pyramid_reverse.reverse()
    #print("pyramid_reverse:",pyramid_reverse)
    label = np.zeros(lenface)+lenpy-1
      
    for i in range(lenface):
        if yy[i] > math.ceil((w0-6)/2)+1:
            yy[i] -= math.ceil((w0)/2)+1
            yy[i] = max(yy[i],0)
            if xx[i] < math.ceil((pyramid_reverse[lenpy-2]-6)/2)+1:
                label[i] = 1
            else:  
                for j in range(1,lenpy-1):
                    if xx[i] > math.ceil((pyramid_reverse[j]-6)/2)+1:
                        xx[i] -= math.ceil((pyramid_reverse[j])/2)+1
                        xx[i] = max(xx[i],0)
                        label[i] -= j-1    
                        break
        else:
            label[i] = 0
    
    scales_py = np.zeros(lenface)
    for i in range(0,lenface):
        scales_py[i] = scale[int(label[i])]
    #print(scales_py)
    #print("......................................")
    #print(scales_py)
    '''
    if y.shape[0] == 1: # only one point exceed threshold
        y = y.T
        x = x.T
        score = map_prob[x,y].T
        dx1 = dx1.T
        dy1 = dy1.T
        dx2 = dx2.T
        dy2 = dy2.T
        # a little stange, when there is only one bb created by PNet
        
        #print("1: x,y", x,y)
        a = (x*map_prob.shape[1]) + (y+1)
        x = a/map_prob.shape[0]
        y = a%map_prob.shape[0] - 1
        #print("2: x,y", x,y)
    else:
        score = map_prob[x,y]
    '''
    #print("dx1.shape", dx1.shape)
    #print('map_prob.shape', map_prob.shape)
   

    score = map_prob[x,y]
    reg = np.array([dx1[x,y], dy1[x,y], dx2[x,y], dy2[x,y]])

    if reg.shape[0] == 0:
        pass 
    
 
    boundingbox = np.array([yy, xx]).T
    scales_py = np.vstack((scales_py,scales_py))
    scales_py = scales_py.T
    #print(boundingbox.shape)
    bb1 = np.fix((stride * (boundingbox) + 1) / scales_py).T # matlab index from 1, so with "boundingbox-1"
    bb2 = np.fix((stride * (boundingbox) + cellsize - 1 + 1) / scales_py).T # while python don't have to
    score = np.array([score])

    boundingbox_out = np.concatenate((bb1, bb2, score, reg), axis=0)

    #print('(x,y)',x,y)
    #print('score', score)
    #print('reg', reg)
    #print("generateBoundingBox over")
    return boundingbox_out.T



def drawBoxes(im, boxes):
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    for i in range(x1.shape[0]):
        cv2.rectangle(im, (int(x1[i]), int(y1[i])), (int(x2[i]), int(y2[i])), (0,255,0), 1)
    return im

from time import time
_tstart_stack = []
def tic():
    _tstart_stack.append(time())
def toc(fmt="Elapsed: %s s"):
    print(fmt % (time()-_tstart_stack.pop()))

#输入原始图片，根据缩放金字塔系数生成奇偶拼接图
def img_joint(img):
    img_odd_pyramid = []
    img_even_pyramid = []
    for i in range(9):
        imgtmp = cv2.resize(img,(PNET_PYRAMID[i][0],PNET_PYRAMID[i][1]))
        #print("imgtmp shape:",imgtmp.shape)
        imgtmp = imgtmp.transpose(2,0,1)
        #print("imgtmp shape:",imgtmp.shape)
       
        if i%2 == 0:
            img_odd_pyramid.append(imgtmp)
        else:   
            img_even_pyramid.append(imgtmp)
    # odd图片拼接
    w0 = ODD_PYRAMID[0][0]
    h0 = ODD_PYRAMID[0][1]
    w1 = ODD_PYRAMID[1][0]
    image_odd = np.zeros((3, h0, w0+w1))
    image_odd[:,0:h0,0:w0] = img_odd_pyramid[0]
    htmp = 0    
    for i in range(1,len(img_odd_pyramid)):
        image_odd[:,htmp:htmp+img_odd_pyramid[i].shape[1],w0:w0+img_odd_pyramid[i].shape[2]] = img_odd_pyramid[i]
        htmp += img_odd_pyramid[i].shape[1]
    # even图片拼接
    w0 = EVEN_PYRAMID[0][0]
    h0 = EVEN_PYRAMID[0][1]
    w1 = EVEN_PYRAMID[1][0]
    image_even = np.zeros((3, h0, w0+w1))
    image_even[:,0:h0,0:w0] = img_even_pyramid[0]
    htmp = 0    
    for i in range(1,len(img_even_pyramid)):
        image_even[:,htmp:htmp+img_even_pyramid[i].shape[1],w0:w0+img_even_pyramid[i].shape[2]] = img_even_pyramid[i]
        htmp += img_even_pyramid[i].shape[1]
    img_odd_even = [image_odd,image_even]
    return img_odd_even

def delete_abn(boxes, width,height):
    """delete abnormal window

    Arguments:
        boxes: a float numpy array of shape [n, 5],
            where each row is (xmin, ymin, xmax, ymax, score).
        width: a int number.
        height:a int number.
    Returns:
        list with indices of the selected boxes
    """

    # if there are no boxes, return the empty list
    if len(boxes) == 0:
        return []
    
    # grab the coordinates of the bounding boxes
    x1, y1, x2, y2, score = [boxes[:, i] for i in range(5)]

    area = (x2 - x1 + 1.0)*(y2 - y1 + 1.0)

    # compute intersections
    # of the box with the image
        
    # warning:在图片中y轴朝下
    # left top corner of intersection boxes
    ix1 = np.maximum(0, x1)
    iy1 = np.maximum(0, y1)

    # right bottom corner of intersection boxes
    ix2 = np.minimum(width, x2)
    iy2 = np.minimum(height, y2)
 
    # width and height of intersection boxes
    w = np.maximum(0.0, ix2 - ix1 + 1.0)
    h = np.maximum(0.0, iy2 - iy1 + 1.0)## 修改，原代码为h = np.maximum(0.0, iy2 - iy1 + 1.0)


    # intersections' areas
    inter = w * h
    overlap = inter/area
    
    # list of picked indices
    pick = []
    pick = np.where(overlap > 0.4)[0]

    return pick

def detect_face(img, minsize, PNet_list, RNet, ONet, threshold, fastresize, factor):
    points = []
    h = img.shape[0]
    w = img.shape[1]
    img_odd_even = img_joint(img)
    #print("img_joint start")
    #print("img_odd_even[0] shape:",img_odd_even[0].shape)
    #print("img_odd_even[1] shape:",img_odd_even[1].shape)
    # first stage
    #'''
    #print("img_odd_even[0]:",img_odd_even[0].shape)
    #img_odd_even[0] = np.swapaxes(img_odd_even[0], 1, 2)
    #img_odd_even[0] = np.swapaxes(img_odd_even[0], 0, 2)
    img_odd_even[0] = img_odd_even[0].transpose(1,2,0)
    img_odd_even[0] = np.array(img_odd_even[0], dtype = np.uint8)
    #cv2.imshow('img_odd_even[0]',img_odd_even[0])
    #ch = cv2.waitKey(50000) & 0xFF

    #img_odd_even[1] = np.swapaxes(img_odd_even[1], 1, 2)
    #img_odd_even[1] = np.swapaxes(img_odd_even[1], 0, 2)
    #print(img_odd_even[1].shape)
    img_odd_even[1] = img_odd_even[1].transpose(1,2,0)
    img_odd_even[1] = np.array(img_odd_even[1], dtype = np.uint8)
    #cv2.imshow('img_odd_even[1]',img_odd_even[1])
    #ch = cv2.waitKey(50000) & 0xFF
    #'''
    img_odd_even[0] = np.swapaxes(img_odd_even[0], 0, 2)
    img_odd_even[1] = np.swapaxes(img_odd_even[1], 0, 2)
    #img_joint is ok
    
    #tic()
    output0= PNet_list[0].inference(inputs=[img_odd_even[0]],data_format='nchw')
    #print("pnet0 inference over")
    #toc()
    odd_prob0 = output0[1]
    odd_conv4_2_0 = output0[0]
    odd_prob0=odd_prob0.reshape(PNET_ODD_ARR[0][1], PNET_ODD_ARR[0][2], PNET_ODD_ARR[0][3])
    odd_conv4_2_0=odd_conv4_2_0.reshape(PNET_ODD_ARR[1][1], PNET_ODD_ARR[1][2], PNET_ODD_ARR[1][3])
    odd_boxes = generateBoundingBox(odd_prob0[1,:,:], odd_conv4_2_0, ODD_SCALES,ODD_PYRAMID,threshold[0])
    #print("odd_boxes:",odd_boxes.shape) 

    #tic()
    output1= PNet_list[1].inference(inputs=[img_odd_even[1]],data_format='nchw')
    #print("pnet inference1 over")
    #toc()
    even_prob1 = output1[1]
    #print("even_prob1 shape:",even_prob1.shape)
    even_conv4_2_1 = output1[0]
    even_prob1= even_prob1.reshape(PNET_EVEN_ARR[0][1], PNET_EVEN_ARR[0][2], PNET_EVEN_ARR[0][3])
    #print(even_prob1[1,:,:])
    even_conv4_2_1=even_conv4_2_1.reshape(PNET_EVEN_ARR[1][1], PNET_EVEN_ARR[1][2], PNET_EVEN_ARR[1][3])
    even_boxes = generateBoundingBox(even_prob1[1,:,:], even_conv4_2_1, EVEN_SCALES,EVEN_PYRAMID,threshold[0])
    #print("even_boxes:",even_boxes.shape)
    total_boxes = np.concatenate((odd_boxes, even_boxes), axis=0)
    if total_boxes.shape[0] != 0:
        pick = delete_abn(total_boxes, 450, 344)
    else:
        pick = []
    if len(pick) > 0 :
        total_boxes = total_boxes[pick, :]

    #print(total_boxes.shape)
    #print("pnet over")

    numbox = total_boxes.shape[0]
    #print("pnet numbox:",numbox)
    if numbox > 0:
        # nms
        pick = nms(total_boxes, 0.4, 'Union')
        total_boxes = total_boxes[pick, :]
        
        # revise and convert to square
        regh = total_boxes[:,3] - total_boxes[:,1]
        regw = total_boxes[:,2] - total_boxes[:,0]
        t1 = total_boxes[:,0] + total_boxes[:,5]*regw
        t2 = total_boxes[:,1] + total_boxes[:,6]*regh
        t3 = total_boxes[:,2] + total_boxes[:,7]*regw
        t4 = total_boxes[:,3] + total_boxes[:,8]*regh
        t5 = total_boxes[:,4]
        total_boxes = np.array([t1,t2,t3,t4,t5]).T


        total_boxes = rerec(total_boxes) # convert box to square
        #print("[4]:",total_boxes.shape[0])
        
        total_boxes[:,0:4] = np.fix(total_boxes[:,0:4])
        #print("[4.5]:",total_boxes.shape[0])
        #print(total_boxes)
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(total_boxes, w, h)
        '''
        print(dy)
        print(edy)
        print(dx)
        print(edx)
        print(y)
        print(ey)
        print(x)
        print(ex)
        '''
        #print(tmpw)
        #print(tmph)
        
    numbox = total_boxes.shape[0]
    #print("window number after pnet+nms:",numbox)
    #####
    # 1 #
    #####
    if numbox > 0:
        # second stage
        #tempimg = np.load('tempimg.npy')
        
        # construct input for RNet
        out_conv5_2=np.zeros((0,4))
        tempimg = np.zeros((numbox, 24, 24, 3)) # (24, 24, 3, numbox)
        tmptarget = np.zeros((24, 24,3))
        score=[]
        #out_prob=0
        '''
        out_conv5_2=np.zeros((0,4))
        img_test = cv2.imread('rnet-24_24_7.jpg')
        print("load img_test successfully")
        cv2.imshow('img_test',img_test)
        ch = cv2.waitKey(10000) & 0xFF
        img_test = np.swapaxes(img_test, 0, 2)
        img_test = np.array([img_test], dtype = np.uint8)
        out=RNet.inference(inputs=[img_test],data_format='nchw')
        print("pent is fine")
        
        tmp00 = np.zeros((24,24,3))
        tmp00[0:24, 0:24,:] = img[24:48, 24:48,:]  
        tmptarget00 = np.zeros((24, 24,3))   
        tmptarget00 = cv2.resize(tmp00, (24, 24))
        tmptarget00 = np.array(tmptarget00, dtype = np.uint8)
        print("tmptarget00 shape:",tmptarget00.shape)
        cv2.imshow('tmptarget00',tmptarget00)
        ch = cv2.waitKey(10000) & 0xFF
        print("try successfully") 
        cv2.imshow('img',img)#######################################
        ch = cv2.waitKey(10000) & 0xFF
        '''
        for k in range(numbox):
            tmp = np.zeros((int(tmph[k]) +1, int(tmpw[k]) + 1,3))
            tmp[int(dy[k]):int(edy[k])+1, int(dx[k]):int(edx[k])+1,:] = img[int(y[k]):int(ey[k])+1, int(x[k]):int(ex[k])+1,:]     
            tmptarget = cv2.resize(tmp, (24, 24))
            tmptarget = np.array(tmptarget, dtype = np.uint8) 
            tmptarget = np.swapaxes(tmptarget, 0, 2)
            #print("tmptarget shape:",tmptarget.shape)
            #print("tmptarget type",type(tmptarget))
            #cv2.imshow('tmptarget',tmptarget)
            #ch = cv2.waitKey(10000) & 0xFF

            #tmptarget = np.array([tmptarget], dtype = np.uint8) 
            #print("tmptarget type",type(tmptarget))
            #print("tmptarget shape:",tmptarget.shape)
            #print(tmptarget)
            #print("before pnet.inference")
            out=RNet.inference(inputs=[tmptarget],data_format='nchw')
            out_prob1=out[1]  
            score=np.append(score,out_prob1[:,1])
            out_conv5_2=np.concatenate((out_conv5_2,out[0]),axis=0)
    
        pass_t = np.where(score>threshold[1])[0] 
        score =  np.array([score[pass_t]]).T
        total_boxes = np.concatenate( (total_boxes[pass_t, 0:4], score), axis = 1)


        mv = out_conv5_2[pass_t, :].T
        if total_boxes.shape[0] > 0:
            pick = nms(total_boxes, 0.5, 'Union')
            #print('pick', pick)
            if len(pick) > 0 :
                total_boxes = total_boxes[pick, :]
                #print("[6]:",total_boxes.shape[0])
                total_boxes = bbreg(total_boxes, mv[:, pick])
                #print("[7]:",total_boxes.shape[0])
                total_boxes = rerec(total_boxes)
                #print("[8]:",total_boxes.shape[0])
        #print("rnet over")
        #####
        # 2 #
        #####

        numbox = total_boxes.shape[0]
        #print("rnet numbox is %d",numbox)
        if numbox > 0:
            # third stage
            
            total_boxes = np.fix(total_boxes)
            [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(total_boxes, w, h)

            tempimg = np.zeros((numbox, 48, 48, 3))
            tmptarget = np.zeros((3, 48, 48)) 
            out_conv6_2=np.zeros((0,4))
            out_conv6_3=np.zeros((0,10))
            score = []
            for k in range(numbox):
                tmp = np.zeros((int(tmph[k]), int(tmpw[k]),3))
                tmp[int(dy[k]):int(edy[k])+1, int(dx[k]):int(edx[k])+1] = img[int(y[k]):int(ey[k])+1, int(x[k]):int(ex[k])+1]
                tempimg[k,:,:,:] = cv2.resize(tmp, (48, 48))
                tmptarget = np.swapaxes(tempimg[k], 0, 2)
                tmptarget = np.array([tmptarget], dtype = np.uint8)
                #tic()
                out=ONet.inference(inputs=[tmptarget],data_format='nchw')
                #toc()
                out_conv6_3=np.concatenate((out_conv6_3,out[1]),axis=0)
                out_conv6_2=np.concatenate((out_conv6_2,out[0]),axis=0)
                score=np.append(score,out[2][:,1])
            #tempimg = (tempimg-127.5)*0.0078125 # [0,255] -> [-1,1]
                
            # ONet
            points = out_conv6_3
            pass_t = np.where(score>threshold[2])[0]#csqerr
            points = points[pass_t, :]
            score = np.array([score[pass_t]]).T
            #print("total_boxes shape ",total_boxes.shape)
            total_boxes = np.concatenate( (total_boxes[pass_t, 0:4], score), axis=1)
            #print("[9]:",total_boxes.shape[0])
            
            mv = out_conv6_2[pass_t, :].T
            w = total_boxes[:,3] - total_boxes[:,1] + 1
            h = total_boxes[:,2] - total_boxes[:,0] + 1

            points[:, 0:5] = np.tile(w, (5,1)).T * points[:, 0:5] + np.tile(total_boxes[:,0], (5,1)).T - 1 
            points[:, 5:10] = np.tile(h, (5,1)).T * points[:, 5:10] + np.tile(total_boxes[:,1], (5,1)).T -1
            #print("onet total_boxes.shape[0] ",total_boxes.shape[0])
            if total_boxes.shape[0] > 0:
                total_boxes = bbreg(total_boxes, mv[:,:])
                #print("[10]:",total_boxes.shape[0])
                pick = nms(total_boxes, 0.7, 'Min')
                
                #print(pick)
                if len(pick) > 0 :
                    total_boxes = total_boxes[pick, :]
                    #print("[11]:",total_boxes.shape[0])
                    points = points[pick, :]

    #####
    # 3 #
    #####
    #print("3:",total_boxes.shape)
    #print("face_dect over")
    return total_boxes, points


def init_pnet():
    list = []
    rknn_odd_name = "PNet_%d_%d.rknn" %(406,207);
    pnet_odd_rknn = RKNN() #verbose=True,verbose_file='./mobilenet_build.log'
    pnet_odd_rknn.load_rknn(rknn_odd_name)
    ret = pnet_odd_rknn.init_runtime()
    if ret != 0:
        #print('Init pnet runtime environment failed')
        exit(ret)
    list.append(pnet_odd_rknn)
    
    rknn_even_name = "PNet_%d_%d.rknn" %(289,147);
    pnet_even_rknn = RKNN() #verbose=True,verbose_file='./mobilenet_build.log'
    pnet_even_rknn.load_rknn(rknn_even_name)
    ret = pnet_even_rknn.init_runtime()
    if ret != 0:
        #print('Init pnet runtime environment failed')
        exit(ret)
    list.append(pnet_even_rknn) 
    
    return list

def main():
 
    imgthread = ShowImgThread()
    winthread = MtcnnThread()
    imgthread.start()
    print("imgthread start")
    winthread.start()
    print("winthread start")
    imgthread.join()
    #print(IMAGE)
    winthread.join()
    #cv2.destroyAllWindows()                

if __name__ == "__main__":
    main()
