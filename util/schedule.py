import numpy as np

def step_decay(epoch, decay_epoch):
    lrs = [0.0003, 0.0001, 0.00001, 0.000001]
    return lrs[epoch // decay_epoch]

def ramp_up_down(epoch, up_epoch=50, total_epoch=400):
    alpha = 0.0003
    epoch += 1
    if epoch <= up_epoch:
        return alpha * np.exp(-5 * np.square(1 - epoch/up_epoch))
    else:
        return 0.5 * alpha * (np.cos(np.pi * (epoch - up_epoch) / total_epoch) + 1)

def cosine_annealing(epoch, lr_init, lr_min, cycle):
    _epoch = epoch + 1
    return ((lr_init-lr_min)/2)*(np.cos(np.pi*(np.mod(_epoch-1,cycle)/(cycle)))+1)+lr_min

