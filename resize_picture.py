import numpy as np
import cv2
if __name__ == '__main__':
    num=1
    w=18
    h=14
    while num <37:
        name_in = "pnet_270_207_%d.jpg"%(num);
        name_out ="./pnet_img/pnet_%d_%d_%d.jpg"%(w,h,num);
        img = cv2.imread(name_in)
        img = cv2.resize(img, (w,h)) # default is bilinear
        img = cv2.imwrite(name_out,img) # default is bilinear
        num += 1

