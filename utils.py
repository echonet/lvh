import torch
import numpy as np
from scipy.signal import find_peaks
import cv2
from pathlib import Path
import argparse

weights_path = Path(__file__).parent / 'weights'
model_paths = {
    'plax': weights_path / 'lv_measurement_model.pt',
    'amyloid': weights_path / 'amyloid.pt',
    'as': weights_path / 'as_model.pt'
}


class BoolAction(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):
        b = values.lower()[0] in ['t', 'y', '1']
        setattr(namespace, self.dest, b)
        print(parser)


def get_clip_dims(paths):
    dims = []
    for p in paths:
        cap = cv2.VideoCapture(str(p))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        dims.append((frame_count, w, h))
    return np.array(dims).T

def read_clip(path, res=None, max_len=None):
    cap = cv2.VideoCapture(str(path))
    frames = []
    i = 0
    while True:
        if max_len is not None and i >= max_len:
            break
        i += 1
        ret, frame = cap.read()
        if not ret:
            break
        if res is not None:
            frame = crop_and_scale(frame, res)
        frames.append(frame)
    cap.release()
    return np.array(frames)

def get_systole_diastole(lvid, kernel=[1, 2, 3, 2, 1], distance=25):
    
    kernel = np.array(kernel)
    kernel = kernel / kernel.sum()
    
    lvid_filt = np.convolve(lvid, kernel, mode='same')
    diastole_i, _ = find_peaks(lvid_filt, distance=distance)
    systole_i, _ = find_peaks(-lvid_filt, distance=distance)

    t = np.arange(len(lvid))
    if len(systole_i) != 0 and len(diastole_i) != 0:
        start_minmax = np.concatenate([diastole_i, systole_i]).min()
        end_minmax = np.concatenate([diastole_i, systole_i]).max()
        diastole_i = np.delete(diastole_i, np.where((diastole_i == start_minmax) | (diastole_i == end_minmax)))
        systole_i = np.delete(systole_i, np.where((systole_i == start_minmax) | (systole_i == end_minmax)))
    
    return systole_i, diastole_i

def get_lens_np(pts):
    return np.sum((pts[..., 1:, :] - pts[..., :-1, :]) ** 2, axis=-1) ** 0.5

def get_points_np(preds, threshold=0.3):
    preds = np.copy(preds)
    preds[preds < threshold] = 0
    Y, X = np.mgrid[:preds.shape[-3], :preds.shape[-2]]
    np.seterr(divide='ignore', invalid='ignore')
    x_pts = np.sum(X[None, ..., None] * preds, axis=(-3, -2)) / np.sum(preds, axis=(-3, -2))
    y_pts = np.sum(Y[None, ..., None] * preds, axis=(-3, -2)) / np.sum(preds, axis=(-3, -2))
    return np.moveaxis(np.array([x_pts, y_pts]), 0, -1)

def get_angles_np(pts):
    a_m = np.arctan2(*np.moveaxis(pts[..., 1:, :] - pts[..., :-1, :], -1, 0))
    a = (a_m[..., 1:] - a_m[..., :-1]) * 180 / np.pi
    a[a > 180] -= 360
    a[a < -180] += 360
    return a

def get_pred_measurements(preds, scale=1):
    pred_pts = get_points_np(preds)
    pred_lens = get_lens_np(pred_pts) * scale
    sys_i, dia_i = get_systole_diastole(pred_lens[:, 1])
    angles = get_angles_np(pred_pts)
    return pred_pts, pred_lens, sys_i, dia_i, angles

def overlay_preds(a, background=None, c=np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1], [0, 1, 0]])):
    # Input shapes [112, 112, 3], [112, 112, 4]
    if background is None:
        background = np.zeros((a.shape[0], a.shape[1], 3))
    np.seterr(divide='ignore', invalid='ignore')
    color = (a ** 2).dot(c) / np.sum(a, axis=-1)[..., None]
    alpha = (1 - np.prod(1 - a, axis=-1))[..., None]
    alpha = np.nan_to_num(alpha)
    color = np.nan_to_num(color)
    return alpha * color + (1 - alpha) * background

def crop_and_scale(img, res=(640, 480)):
    in_res = (img.shape[1], img.shape[0])
    r_in = in_res[0] / in_res[1]
    r_out = res[0] / res[1]

    if r_in > r_out:
        padding = int(round((in_res[0] - r_out * in_res[1]) / 2))
        img = img[:, padding:-padding]
    if r_in < r_out:
        padding = int(round((in_res[1] - in_res[0] / r_out) / 2))
        img = img[padding:-padding]
    
    img = cv2.resize(img, res)

    return img
