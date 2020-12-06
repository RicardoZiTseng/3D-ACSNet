import os
from functools import partial

import keras
import numpy as np
import tensorflow as tf
from keras.backend import clear_session, set_session
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from models.model import DenseUNet, cross_entropy
from util.callbacks import ModelCheckpointParallel
from util.data import DataGen
from util.misc import save_params, maybe_mkdir_p
from util.schedule import cosine_annealing

def main(params):
    # ======================================
    #       Set Environment Variable      #
    # ======================================
    work_path = os.path.abspath('./workdir')
    save_path = os.path.join(work_path, 'save', params.name)
    maybe_mkdir_p(save_path)
    save_params(params, os.path.join(save_path, 'train.json'))
    save_file_name = os.path.join(save_path, '{epoch:02d}.h5')

    # ======================================
    #       Close Useless Information     #
    # ======================================
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    clear_session()

    # ==============================================================
    #         Set GPU Environment And Initialize Networks         #
    # ==============================================================
    gpu_nums = len(params.gpu.split(','))
    os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    set_session(tf.Session(config=config))

    model = DenseUNet().build_model()

    if params.load_model_file is not None:
        print('Loadding ', params.load_model_file)
        model.load_weights(params.load_model_file)

    if gpu_nums > 1:
        print('MultiGpus: {}'.format(gpu_nums))
        save_schedule = ModelCheckpointParallel(filepath=save_file_name, period=params.save_period)
        mg_model = multi_gpu_model(model, gpus=gpu_nums)
    else:
        print('Single device.')
        save_schedule = keras.callbacks.ModelCheckpoint(filepath=save_file_name, period=params.save_period)
        mg_model = model

    # ===================================================================
    #        Set Training Callbacks and Initialize Data Generator      #
    # ===================================================================
    train_gen = DataGen(params.train_dir, params.train_ids,
                        params.batch_size, params.cube_size).make_gen()

    lr_schedule_fn = partial(cosine_annealing, lr_init=params.lr_init, lr_min=params.lr_min, cycle=params.cycle)

    lr_schedule = keras.callbacks.LearningRateScheduler(lr_schedule_fn, verbose=1)

    call_backs = [lr_schedule, save_schedule]
    
    mg_model.compile(optimizer=Adam(lr=params.lr_init), loss=cross_entropy)
    mg_model.fit_generator(train_gen,
                           steps_per_epoch=params.train_nums_per_epoch//params.batch_size,
                           epochs=params.epochs, callbacks=call_backs)

