# -*- coding: utf8 -*-
import os
import numpy as np
import cntk as C
from cntk.learners import sgd, learning_rate_schedule, UnitType
from cntk.device import gpu, try_set_default_device
from cntk.layers import Dense, Sequential
from PIL import Image
from PIL import ImageOps
import cntk_unet_deconv as cntk_unet
import time
import sys
import argparse
import random
import re
import cntk.losses
from cntk import cross_entropy_with_softmax, classification_error, reduce_mean
from cntk.cntk_py import binary_cross_entropy

# from docopt import docopt

try_set_default_device(gpu(0))
print("-------------------------")


def Img2CntkImg(path, resizeX, resizeY, is_ans=True):
    img = Image.open(path)
    # img = img.resize((256, 512))#width, height
    img = img.resize((resizeX, resizeY))  # width, height

    img = ImageOps.grayscale(img)
    if is_ans:
        training_img = np.array(img)
        training_img = training_img // 255
    else:
        training_img = np.array(img)
        training_img = training_img / 255
    training_img = np.array([training_img])
    training_img = training_img.astype(np.float32)

    del img
    return training_img


def train(TrnPath, SavePath, savename, imgX, imgY, ans_target, trn_target, bs, epc, imglist, use_existing=False):
    # Get minibatches of training data and perform model training
    minibatch_size = bs
    num_epochs = epc
    # test = np.zeros((1, 512, 256))
    test = np.zeros((1, imgY, imgX))
    shape = test.shape
    print("test image shape:" + str(test.shape))

    x = C.input_variable(shape)
    y = C.input_variable(shape)

    z = cntk_unet.create_model(x)
    #dice_coef = cntk_unet.dice_coefficient(z, y)
    # ll = C.Logistic(z, y)
    # ce = C.CrossEntropy(z, y)
    '''
    checkpoint_file = "/home/ys/PycharmProjects/cntk-unet/cntk-unet.dnn"
    if use_existing:
        z.load_model(checkpoint_file)
    '''

    # Prepare model and trainer
    lr = learning_rate_schedule(0.00005, UnitType.sample)
    momentum = C.learners.momentum_as_time_constant_schedule(0)

    # loss and metric
    ce = binary_cross_entropy(z, y)
    pe = binary_cross_entropy(z, y)
    trainer = C.Trainer(z, (ce, pe), C.learners.adam(z.parameters, lr=lr, momentum=momentum))

    file_num = len(imglist)

    training_errors = []
    test_errors = []

    sw = time.time()
    for e in range(0, num_epochs):
        pattern = "[A-Z]"
        num = 0
        random.shuffle(imglist)
        for i in range(0, file_num):

            # http://qiita.com/wanwanland/items/ce272419dde2f95cdabc
            match = re.search(pattern, imglist[i])
            # print("matchgroup:"+match.group())
            if match.group() == "A":
                ansfile = TrnPath + r"/ans/Amode/" + ans_target + r"/" + imglist[i]
                trnfile = TrnPath + r"/raw/Amode/" + trn_target + r"/" + imglist[i]
            if match.group() == "B":
                ansfile = TrnPath + r"/ans/Bmode/" + ans_target + r"/" + imglist[i]
                trnfile = TrnPath + r"/raw/Bmode/" + trn_target + r"/" + imglist[i]
            if match.group() == "Z":
                ansfile = TrnPath + r"/ans/Zmode/" + ans_target + r"/" + imglist[i]
                trnfile = TrnPath + r"/raw/Zmode/" + trn_target + r"/" + imglist[i]
            # print(ansfile)
            # print(trnfile)

            if i % minibatch_size == 0:
                training_y = np.array([Img2CntkImg(ansfile, imgX, imgY)])
                training_x = np.array([Img2CntkImg(trnfile, imgX, imgY)])

            elif i % minibatch_size > 0 and i % minibatch_size < minibatch_size - 1:
                training_y = np.append(training_y, np.array([Img2CntkImg(ansfile, imgX, imgY, True)]), axis=0)
                training_x = np.append(training_x, np.array([Img2CntkImg(trnfile, imgX, imgY, False)]), axis=0)

            elif i % minibatch_size == minibatch_size - 1:
                training_y = np.append(training_y, np.array([Img2CntkImg(ansfile, imgX, imgY, True)]), axis=0)
                training_x = np.append(training_x, np.array([Img2CntkImg(trnfile, imgX, imgY, False)]), axis=0)
                trainer.train_minibatch({x: training_x, y: training_y})
                # print("###################")
                if i == num_epochs - 1:
                    # Measure training error
                    training_errors.append(trainer.test_minibatch({x: training_x, y: training_y}))
                    print("Epoch:" + str(e) + "  Error:" + str(np.mean(training_errors)) + "  time:" + str(
                        time.time() - sw))

        # print("epoch:" + str(e) + "  error:"+str(np.mean(training_errors))+"  time:" + str(time.time()-sw))


        if e % 50 == 0:
            trainer.save_checkpoint(SavePath + r"/" + savename + r"_" + str(e) + ".dnn")
            # print("epoch:" + str(e))
            # print("time passed:" + str(time.time() - sw))
            training_errors.append(trainer.test_minibatch({x: training_x, y: training_y}))
            print("Epoch:" + str(e) + "  Error:" + str(np.mean(training_errors)) + "  time:" + str(time.time() - sw))
            training_errors = []

    return trainer


if __name__ == '__main__':
    # http://ja.pymotw.com/2/argparse/
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir', action='store', dest='dir', required=True, help="Read directly")
    parser.add_argument('-sdir', action='store', dest='sdir', required=True, help='where you want to save model')
    parser.add_argument('-savename', action='store', dest='savename', required=True, help='Store a model saving name')

    parser.add_argument('-x', action='store', dest='x', type=int, required=True, help='Store image width number')
    parser.add_argument('-y', action='store', dest='y', type=int, required=True, help='Store image height number')

    parser.add_argument('-anstgt', action='store', dest='anstgt', required=True,
                        help='Deep learning answer target. Ex.RB, red, White, etc...')
    parser.add_argument('-trntgt', action='store', dest='trntgt', default="raw",
                        help='Deep learning raw target. Ex.dns, raw, etc...')

    parser.add_argument('-bs', action='store', dest='bs', type=int, required=True, help='Batch Size')
    parser.add_argument('-epc', action='store', dest='epc', type=int, required=True, help='Total Epoch')

    results = parser.parse_args()
    # filelist = os.listdir(AnsPath)
    # print(len(filelist))

    # print (results.ans_dir+'/ans/Amode/'+results.tgt)
    filelist = os.listdir(results.dir + '/ans/Amode/' + results.anstgt)
    filelist.extend(os.listdir(results.dir + '/ans/Bmode/' + results.anstgt))
    filelist.extend(os.listdir(results.dir + '/ans/Zmode/' + results.anstgt))
    random.shuffle(filelist)
    # print(results.dir+'/raw/Amode/'+results.trntgt+"/8A24_160914150823_25.bmp")
    # print (filelist)
    # print(len(filelist))

    print('ans directory     =', results.dir)
    print('save directory    =', results.sdir)
    print('save model name   =', results.savename)
    print('training target   =', results.trntgt)
    print('answer target     =', results.anstgt)

    print('image width       =', results.x)
    print('image height      =', results.y)
    print('Batch size        =', results.bs)
    print('Total Epochs       =', results.epc)

    #make directory
    if not os.path.exists(results.sdir):
        os.makedirs(results.sdir)
    # start training
    # train(TrnPath, SavePath, savename, imgX, imgY, ans_target, trn_target, bs, epc, imglist, use_existing=False):
    train(results.dir, results.sdir, results.savename, results.x, results.y, results.anstgt, results.trntgt, results.bs,
          results.epc, filelist, False)
    print("finish all training")

    '''
    print (trainer)
    print (training_errors)
    print (test_errors)
    '''