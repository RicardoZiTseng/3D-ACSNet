import numpy as np
import os

def make_onehot_label(label, num_classes):
    onehot_encoding = []
    for c in range(num_classes):
        onehot_encoding.append(label == c)
    onehot_encoding = np.concatenate(onehot_encoding, axis=-1)
    onehot_encoding = np.array(onehot_encoding, dtype=np.int16)
    return onehot_encoding

def image_norm(img):
    mask = (img > 0)
    mean = np.mean(img[mask])
    std = np.std(img[mask])
    return (img - mean) / std

def cut_edge(data, keep_margin):
    '''
    function that cuts zero edge
    # Ricardo: calculating the margin number of non-zero area.
    '''
    D, H, W, _ = data.shape
    print(D, H, W)
    D_s, D_e = 0, D - 1
    H_s, H_e = 0, H - 1
    W_s, W_e = 0, W - 1

    while D_s < D:
        if data[D_s].sum() != 0:
            break
        D_s += 1
    while D_e > D_s:
        if data[D_e].sum() != 0:
            break
        D_e -= 1
    while H_s < H:
        if data[:, H_s].sum() != 0:
            break
        H_s += 1
    while H_e > H_s:
        if data[:, H_e].sum() != 0:
            break
        H_e -= 1
    while W_s < W:
        if data[:, :, W_s].sum() != 0:
            break
        W_s += 1
    while W_e > W_s:
        if data[:, :, W_e].sum() != 0:
            break
        W_e -= 1

    if keep_margin != 0:
        D_s = max(0, D_s - keep_margin)
        D_e = min(D - 1, D_e + keep_margin)
        H_s = max(0, H_s - keep_margin)
        H_e = min(H - 1, H_e + keep_margin)
        W_s = max(0, W_s - keep_margin)
        W_e = min(W - 1, W_e + keep_margin)

    return int(D_s), int(D_e), int(H_s), int(H_e), int(W_s), int(W_e)

def vote(segmentation_sets):
    onehot_segmentation_sets = [make_onehot_label(seg, 4) for seg in segmentation_sets]
    vote_seg = np.argmax(sum(onehot_segmentation_sets), axis=-1)
    vote_seg = np.array(np.expand_dims(vote_seg, axis=-1), dtype=np.uint8)
    return vote_seg
