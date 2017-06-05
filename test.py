# -*- coding: utf8 -*-
import os
import numpy as np
import cntk as C
from cntk.learners import sgd, learning_rate_schedule, UnitType
from cntk.device import gpu, try_set_default_device
from PIL import Image
from PIL import ImageOps
from cntk.io import transforms
import cntk
from cntk import cntk_py
from cntk.layers import Convolution, MaxPooling, Dense
from cntk.initializer import glorot_uniform
from cntk.ops import relu, sigmoid, input_variable
#https://github.com/usuyama/cntk_unet/blob/master/cntk_unet.py
np.set_printoptions(threshold=np.inf)
AnsPath = "/home/ys/Share/7_DL_model_set/ver20170413/15Z32/DL_Ans_half"
BasePath = "/home/ys/Share/7_DL_model_set/ver20170413/15Z32/trn_half"
list = os.listdir(AnsPath)

testimg = Image.open(AnsPath + "/" + list[0])
testimg = testimg.resize((572, 572))
testimg = ImageOps.grayscale(testimg)

test_image = np.array(testimg)  # .transpose(2,0,1)
test_image = np.array([test_image])
shape = test_image.shape
data_size = test_image.shape[0]

x = C.input_variable(shape)
y = C.input_variable(shape)

print(test_image.shape)
z0 =C.splice(test_image,test_image, axis=0)
z1 =C.splice(test_image,test_image,-3)
z2 =C.splice(test_image,test_image,2)

z = np.array([test_image,test_image])
print(z0.shape)
print(z1.shape)
print(z2.shape)

print(z.shape)
x= np.array([[[0, 0,0], [1,1,1],[2,2,2]],[3,3,3], [4,4,4],[5,5,5]])
print (x.shape)
"""
xr = C.reshape(x, (x.shape[0], x.shape[1], 1, x.shape[2], 1))
xx = C.splice(xr, xr, axis=-1) # axis=-1 refers to the last axis
xy = C.splice(xx, xx, axis=-3) # axis=-3 refers to the middle axis

print (xr)
xr
print (xx)sss
r = C.reshape(xy, (x.shape[0], x.shape[1] * 2, x.shape[2] * 2))

print(r.shape)ssss
"""
t = x.crop(crop_type ='center', crop_size=(2,2))
#t = cntk.io.transforms.crop(crop_type ='center', crop_size=(2,2))(x)
print(x.shape)
