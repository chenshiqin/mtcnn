import numpy as np
import cv2
from rknn.api import RKNN

if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN()
    
    # pre-process config
    print('--> config model')
    rknn.config(channel_mean_value='127.5 127.5 127.5 128', reorder_channel='2 1 0',quantized_dtype='dynamic_fixed_point-8')
    print('done')

    # Load tensorflow model
    print('--> Loading model')
    ret = rknn.load_caffe(model='./RNet.prototxt', proto='caffe', blobs='./RNet.caffemodel')
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=True, dataset='./dataset_rnet.txt')
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export RKNN model')
    ret = rknn.export_rknn('./RNet.rknn')
    if ret != 0:
        print('Export model failed!')
        exit(ret)
    print('done')

    img = cv2.imread('./rnet-24_24_1.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (24,24)) # default is bilinear
    img = np.swapaxes(img, 0, 2)
    #img = cv2.resize(img, (48,48)) # default is bilinear

    tempimg = np.ones((1, 3, 48, 48))

    # init runtime environment
    #print('--> Init runtime environment')
    ret = rknn.init_runtime()
    #ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[tempimg],data_format='nchw')
    print(outputs)
    print('done')

    # perf
    #print('--> Begin evaluate model performance')
    #perf_results = rknn.eval_perf(inputs=[img])
    #print('done')

    rknn.release()


