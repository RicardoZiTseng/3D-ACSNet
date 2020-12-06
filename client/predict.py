import argparse
import os
import time

import keras.backend as K
import nibabel as nib
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from models.model import DenseUNet
from util.metrics import Dice as dice
from util.predict_funcs import predict_segmentation
from util.prep import image_norm, vote
from util.timer import Clock
from util.misc import maybe_mkdir_p

def save(t1_path, save_path, segmentation):
    t1_affine = nib.load(t1_path).get_affine()
    segmentation = np.array(segmentation, dtype=np.uint8)
    seg = nib.AnalyzeImage(segmentation, t1_affine)
    nib.save(seg, save_path)


def analyze_score(dsc_score):
    csf = []
    gm = []
    wm = []
    for i in range(len(dsc_score)):
        csf.append(dsc_score[i][0])
        gm.append(dsc_score[i][1])
        wm.append(dsc_score[i][2])
    print('%s   | %2.4f | %2.4f | %2.4f | %2.4f |' % ('Avg Dice', np.mean(csf), np.mean(
        gm), np.mean(wm), np.mean([np.mean(csf), np.mean(gm), np.mean(wm)])))


def cal_acc(label, pred):
    dsc = []
    print('------------------------------------------')
    for i in range(1, 4):
        dsc_i = dice(pred, label, i)
        dsc_i = round(dsc_i*100, 2)
        dsc.append(dsc_i)
    print('Data     | CSF     | GM      | WM      | Avg     |')
    print('Dice     | %2.4f | %2.4f | %2.4f | %2.4f |' %
          (dsc[0], dsc[1], dsc[2], np.mean(dsc)))
    return dsc


def predict(path_dict, model, cube_size, strides):
    t1_data = nib.load(path_dict['t1w']).get_data()
    t2_data = nib.load(path_dict['t2w']).get_data()

    t1_data_norm = image_norm(t1_data)
    t2_data_norm = image_norm(t2_data)

    subject_data = {'t1w': t1_data_norm, 't2w': t2_data_norm}

    segmentation = predict_segmentation(
        subject_data, 30, cube_size, strides, model, 1)
    
    segmentation = np.expand_dims(np.argmax(segmentation, axis=-1), axis=-1)

    if 'label' in path_dict.keys():
        label_data = nib.load(path_dict['label']).get_data()
        cal_acc(label_data, segmentation)

    return segmentation


def main(params):
    gpu_id = params.gpu_id
    save_folder = params.save_folder
    data_path = params.data_path
    label_path = params.label_path
    cube_size = [params.cube_size] * 3
    strides = [params.crop_size] * 3
    clock = Clock()
    maybe_mkdir_p(os.path.abspath(save_folder))
    print("Overlapping step size is {}.".format(params.crop_size))

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    if gpu_id is not None:
        gpu = '/gpu:' + str(gpu_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        set_session(tf.Session(config=config))
    else:
        gpu = '/cpu:0'

    with tf.device(gpu):
        normal_model = DenseUNet(deploy=params.deploy).build_model()

        dice_score = []

        for i in params.subjects:
            t1_path = os.path.join(data_path, 'subject-' + str(i) + '-T1.img')
            t2_path = os.path.join(data_path, 'subject-' + str(i) + '-T2.img')
            save_seg_path = os.path.join(save_folder, 'subject-' + str(i) + '-label.img')
            input_dict = {'t1w': t1_path, 't2w': t2_path}
            segmentation_sets = []
            if params.predict_mode == 'evaluation':
                input_dict['label'] = os.path.join(
                    label_path, 'subject-' + str(i) + '-label.img')
            clock.tic()
            for j in range(len(params.model_files)):
                print("subj: {}, model: `{}`".format(i, params.model_files[j]))
                normal_model.load_weights(params.model_files[j])
                seg = predict(input_dict, normal_model, cube_size, strides)
                segmentation_sets.append(seg)
            final_seg = vote(segmentation_sets)
            save(t1_path, save_seg_path, final_seg)
            print("Subject {}\'s segmentation result is stored in `{}`.".format(i, save_seg_path))

            clock.toc()
            if params.predict_mode == 'evaluation':
                if len(params.model_files) > 1:
                    print("The ensemble result is :")
                    label_data = nib.load(input_dict['label']).get_data()
                    dsc = cal_acc(label_data, final_seg)
                    dice_score.append(dsc)
        print("The average segmentation time per subject is about {:.3f} seconds (about {:.3f} minutes).".format(clock.average_time, clock.average_time/60))

        if params.predict_mode == 'evaluation' and len(params.model_files) > 1:
            analyze_score(dice_score)
