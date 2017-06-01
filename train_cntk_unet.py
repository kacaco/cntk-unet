# -*- coding: utf8 -*-
import os
import numpy as np
import cntk as C
from cntk.learners import sgd, learning_rate_schedule, UnitType
from cntk.device import gpu, try_set_default_device
from PIL import Image
from PIL import ImageOps
import cntk_unet

try_set_default_device(gpu(0))
print ("-------------------------")
def train(AnsPath, BasePath, list, use_existing=False):

    test_image = np.array(Image.open(AnsPath+"\\"+list[0]))
    shape = test_image.shape
    data_size = test_image.shape[0]

    x = C.input_variable(shape)
    y = C.input_variable(shape)

    z = cntk_unet.create_model(x)
    dice_coef = cntk_unet.dice_coefficient(z, y)

    checkpoint_file = "cntk-unet.dnn"
    if use_existing:
        z.load_model(checkpoint_file)

    # Prepare model and trainer
    lr = learning_rate_schedule(0.00001, UnitType.sample)
    momentum = C.learners.momentum_as_time_constant_schedule(0)
    trainer = C.Trainer(z, (-dice_coef, -dice_coef), C.learners.adam(z.parameters, lr=lr, momentum=momentum))

    # Get minibatches of training data and perform model training
    minibatch_size = 2
    num_epochs = 1
    num_mb_per_epoch = int(data_size / minibatch_size)


    for e in range(0, num_epochs):
        for i in range(0, num_mb_per_epoch):

            img_y =Image.open(AnsPath+"\\"+list[i * minibatch_size:(i + 1) * minibatch_size])
            img_x =Image.open(BasePath+"\\"+list[i * minibatch_size:(i + 1) * minibatch_size])
            
            img_y= ImageOps.grayscale(img_y)
            img_x= ImageOps.grayscale(img_x)

            training_y = np.array(img_y)
            training_x = np.array(img_x)
            trainer.train_minibatch({x: training_x, y: training_y})

        trainer.save_checkpoint(checkpoint_file)

    return trainer

if __name__ == '__main__':
    AnsPath  = "E:\\ver20170413\\15B24\\Red_half"
    BasePath = "E:\\ver20170413\\15B24\\trn_half"
    filelist = os.listdir(AnsPath)
    train(AnsPath, BasePath, filelist, False)
