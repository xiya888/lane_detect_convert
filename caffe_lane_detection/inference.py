import os
#import torch
import time
from PIL import Image
import numpy as np
import scipy.special
import cv2
#import torch.onnx
#from model.model import parsingNet
#import torchvision.transforms as transforms
# torch.backends.cudnn.deterministic = False
import sys
sys.path.append('/home/sany/project/convert_model/caffe/python')
import caffe

caffe.set_mode_gpu()
net = caffe.Net('./ep049_34_modify_selftrain.prototxt', './ep049_34_modify_selftrain.caffemodel', caffe.TEST)
net.blobs['input.1'].reshape(1, 3, 288, 800)

image_mean = np.array([0.485, 0.456, 0.406])
image_std = np.array([0.229, 0.224, 0.225])


#img_transforms = transforms.Compose([
#    transforms.Resize((288, 800)),
#    transforms.ToTensor(),
#    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#])

#img_w, img_h = 1640, 590
img_w, img_h = 1640, 590
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
vout = cv2.VideoWriter('21.avi', fourcc , 15.0, (img_w, img_h))

#for file in os.listdir('./05250559_0320.MP4/'):
for file in os.listdir('./21/'):
    if not file.endswith("jpg"):
        continue
    #imPath= './05250559_0320.MP4/' + file
    imPath= './21/' + file
    img = cv2.imread(imPath)
    #img = img[130:,0:1280,:]
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(800,288))
    img = img / 255.
    img = (img - image_mean) / image_std
    tmp_batch = np.zeros([1, 3, 288, 800], dtype=np.float32)
    tmp_batch[0, :, :, :] = img.transpose(2, 0, 1)
    net.blobs['input.1'].data[...] = tmp_batch
    res = net.forward()

    col_sample = np.linspace(0, 800 - 1, 200)
    col_sample_w = col_sample[1] - col_sample[0]

    out_j = res['352'][0]
    out_j = out_j[:, ::-1, :]
    prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
    idx = np.arange(200) + 1
    idx = idx.reshape(-1, 1, 1)
    loc = np.sum(prob * idx, axis=0)
    out_j = np.argmax(out_j, axis=0)
    loc[out_j == 200] = 0
    out_j = loc

    vis = cv2.imread(imPath)
    for i in range(out_j.shape[1]):
        if np.sum(out_j[:, i] != 0) > 2:
            for k in range(out_j.shape[0]):
                if out_j[k, i] > 0:
                    ppp = (int(out_j[k, i] * col_sample_w * 1640 / 800) - 1, int(590 - k * 20) - 1)
                    cv2.circle(vis, ppp, 5, (0, 255, 0), -1)

    #cv2.imshow("vis",vis)
    #cv2.imwrite(file,vis)
    vout.write(vis)
vout.release()
    #cv2.waitKey(5000)
