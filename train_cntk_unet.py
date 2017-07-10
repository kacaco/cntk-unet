# -*- coding: utf8 -*-
import os
import numpy as np
import cntk as C
from cntk.learners import sgd, learning_rate_schedule, UnitType
from cntk.device import gpu, try_set_default_device
from PIL import Image
from PIL import ImageOps
import cntk_unet512 as cntk_unet
import time
try_set_default_device(gpu(0))
print ("-------------------------")

def Img2CntkImg(path, resizeX, resizeY):
    img = Image.open(path)
    img = img.resize((256, 512))
    img = ImageOps.grayscale(img)


    training_img = np.array(img)
    training_img = np.array([training_img])
    training_img = training_img.astype(np.float32)

    del img
    return training_img

def train(AnsPath, TrnPath, list, use_existing=False):

    # Get minibatches of training data and perform model training
    minibatch_size = 10
    num_epochs = 1000
    test = np.zeros((1, 512, 256))
    shape = test.shape
    print("test image shape:"+str(test.shape))


    x = C.input_variable(shape)
    y = C.input_variable(shape)

    z = cntk_unet.create_model(x)
    dice_coef = cntk_unet.dice_coefficient(z, y)

    checkpoint_file = "/home/ys/PycharmProjects/cntk-unet/cntk-unet.dnn"
    if use_existing:
        z.load_model(checkpoint_file)

    # Prepare model and trainer
    lr = learning_rate_schedule(0.00001, UnitType.sample)
    momentum = C.learners.momentum_as_time_constant_schedule(0)
    trainer = C.Trainer(z, (-dice_coef, -dice_coef), C.learners.adam(z.parameters, lr=lr, momentum=momentum))

    file_num = len(list)

    training_errors = []
    test_errors = []




    sw = time.time()
    for e in range(0, num_epochs):

        num =0
        for i in range(0, file_num):
            #print(i%minibatch_size)
            if i%minibatch_size ==0:
                training_y = np.array([Img2CntkImg(AnsPath + "/" + list[i], 256, 512)])
                training_x = np.array([Img2CntkImg(TrnPath + "/" + list[i], 256, 512)])
                #print("[1]"+str(training_y.shape))
            elif i%minibatch_size >0 and i%minibatch_size<minibatch_size-1:
                training_y = np.append(training_y,np.array([Img2CntkImg(AnsPath+"/"+list[i], 256, 512)]),axis=0)
                training_x = np.append(training_x,np.array([Img2CntkImg(TrnPath+"/"+list[i], 256, 512)]),axis=0)
                #print("[2]"+str(training_y.shape))
            elif i%minibatch_size==minibatch_size-1:
                training_y = np.append(training_y,np.array([Img2CntkImg(AnsPath+"/"+list[i], 256, 512)]),axis=0)
                training_x = np.append(training_x,np.array([Img2CntkImg(TrnPath+"/"+list[i], 256, 512)]),axis=0)
                trainer.train_minibatch({x: training_x, y: training_y})
                #print("[3]"+str(training_y.shape))
                #print("###################")
                if i == num_epochs-1:
                    # Measure training error
                    training_errors.append(trainer.test_minibatch({x: training_x, y: training_y}))
                    print("Epoch:" + str(e) + "  Error:" + str(np.mean(training_errors)) + "  time:" + str(time.time() - sw))

        #print("epoch:" + str(e) + "  error:"+str(np.mean(training_errors))+"  time:" + str(time.time()-sw))
        training_errors = []

        if e%50==0:
            trainer.save_checkpoint("/home/ys/PycharmProjects/cntk-unet/batch15_cntk-unet512_Zraw_"+str(e)+".dnn")
            print ("epoch:"+str(e))
            print ("time passed:"+str(time.time()-sw))

    return trainer

if __name__ == '__main__':
    AnsPath  = "/home/ys/Share/7_DL_model_set/ver20170413/Z_White"
    BasePath = "/home/ys/Share/7_DL_model_set/ver20170413/Z_raw"
    filelist = os.listdir(AnsPath)
    trainer, training_errors, test_errors =  train(AnsPath, BasePath, filelist, False)
    print (trainer)
    print (training_errors)
    print (test_errors)

