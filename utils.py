from typing import Iterable, Tuple
import torch
import numpy as np
from scipy.signal import find_peaks
import cv2 ### 4.5.3.56
from pathlib import Path
import argparse
from typing import Union
import torchvision
import tqdm
import pandas as pd 
import pydicom
import shutil
import os

# relative paths to weights for various models
weights_path = Path(__file__).parent / 'weights'
model_paths = {
    'plax': weights_path / 'hypertrophy_model.pt',
    'amyloid': weights_path / 'amyloid.pt',
    'as': weights_path / 'as_model.pt'
}


class BoolAction(argparse.Action):

    """Class used by argparse to parse binary arguements.
    Yes, Y, y, True, T, t are all accepted as True. Any other
    arguement is evaluated as False.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        b = values.lower()[0] in ['t', 'y', '1']
        setattr(namespace, self.dest, b)


def get_clip_dims(paths: Iterable[Union[Path, str]]) -> Tuple[np.ndarray, list]:
    """Gets the dimensions of all the videos in a list of paths.

    Args:
        paths (Iterable[Union[Path, str]]): List of paths to iterrate through

    Returns:
        dims (np.ndarray): array of clip dims (frames, width, height). shape=(n, 3)
        filenames (list): list of filenames. len=n
    """
    
    dims = []
    fnames = []
    for p in paths:
        if isinstance(p, str):
            p = Path(p)
        if '.avi' not in p.name:
            continue
        cap = cv2.VideoCapture(str(p))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        dims.append((frame_count, w, h))
        fnames.append(p.name)
    return np.array(dims).T, fnames

def read_clip(path, res=(640,480), max_len=None) -> np.ndarray:
    """Reads a clip and returns it as a numpy array

    Args:
        path ([Path, str]): Path to video to read
        res (Tuple[int], optional): Resolution of video to return. If None, 
            original resolution will be returned otherwise the video will be 
            cropped and downsampled. Defaults to None.
        max_len (int, optional): Max length of video to read. Only the first n 
            frames of longer videos will be returned. Defaults to None.

    Returns:
        np.ndarray: Numpy array of video. shape=(n, h, w, 3)
    """
    cap = cv2.VideoCapture(str(path))#,cv2.CAP_FFMPEG)
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

def get_systole_diastole(lvid: np.ndarray, kernel=[1, 2, 3, 2, 1], distance: int=25) -> Tuple[np.ndarray]:
    """Finds heart phase from a representative signal. Signal must be maximum at end diastole and
    minimum at end systole.

    Args:
        lvid (np.ndarray): Signal representing heart phase. shape=(n,)
        kernel (list, optional): Smoothing kernel used before finding peaks. Defaults to [1, 2, 3, 2, 1].
        distance (int, optional): Minimum distance between peaks in find_peaks(). Defaults to 25.

    Returns:
        systole_i (np.ndarray): Indices of end systole. shape=(n_sys,)
        diastole_i (np.ndarray): Indices of end diastole. shape=(n_dia,)
    """

    # Smooth input
    kernel = np.array(kernel)
    kernel = kernel / kernel.sum()
    lvid_filt = np.convolve(lvid, kernel, mode='same')

    # Find peaks
    diastole_i, _ = find_peaks(lvid_filt, distance=distance)
    systole_i, _ = find_peaks(-lvid_filt, distance=distance)

    # Ignore first/last index if possible
    if len(systole_i) != 0 and len(diastole_i) != 0:
        start_minmax = np.concatenate([diastole_i, systole_i]).min()
        end_minmax = np.concatenate([diastole_i, systole_i]).max()
        diastole_i = np.delete(diastole_i, np.where((diastole_i == start_minmax) | (diastole_i == end_minmax)))
        systole_i = np.delete(systole_i, np.where((systole_i == start_minmax) | (systole_i == end_minmax)))
    
    return systole_i, diastole_i

def get_lens_np(pts: np.ndarray) -> np.ndarray:
    """Used to get the euclidean distance between consecutive points.

    Args:
        pts (np.ndarray): Input points. shape=(..., n, 2)

    Returns:
        np.ndarray: Distances. shape=(..., n-1)
    """
    return np.sum((pts[..., 1:, :] - pts[..., :-1, :]) ** 2, axis=-1) ** 0.5

def get_points_np(preds: np.ndarray, threshold: float=0.3) -> np.ndarray:
    """Gets the centroid of heatmaps.

    Args:
        preds (np.ndarray): Input heatmaps. shape=(n, h, w, c)
        threshold (float, optional): Value below which input pixels are ignored. Defaults to 0.3.

    Returns:
        np.ndarray: Centroid locations. shape=(n, c, 2)
    """

    preds = np.copy(preds)
    preds[preds < threshold] = 0
    Y, X = np.mgrid[:preds.shape[-3], :preds.shape[-2]]
    np.seterr(divide='ignore', invalid='ignore')
    x_pts = np.sum(X[None, ..., None] * preds, axis=(-3, -2)) / np.sum(preds, axis=(-3, -2))
    y_pts = np.sum(Y[None, ..., None] * preds, axis=(-3, -2)) / np.sum(preds, axis=(-3, -2))
    return np.moveaxis(np.array([x_pts, y_pts]), 0, -1)

def get_angles_np(pts: np.ndarray) -> np.ndarray:
    """Returns the angles between corresponding segments of a polyline.

    Args:
        pts (np.ndarray): Input polyline. shape=(..., n, 2)

    Returns:
        np.ndarray: Angles in degrees. Constrained to [-180, 180]. shape=(..., n-1)
    """

    a_m = np.arctan2(*np.moveaxis(pts[..., 1:, :] - pts[..., :-1, :], -1, 0))
    a = (a_m[..., 1:] - a_m[..., :-1]) * 180 / np.pi
    a[a > 180] -= 360
    a[a < -180] += 360
    return a

def get_pred_measurements(preds: np.ndarray, scale: float=1) -> Tuple[np.ndarray]:
    """Given PLAX heatmap predictions, generate values of interest.

    Args:
        preds (np.ndarray): PLAX model heatmap predictions. shape=(n, h, w, 4)
        scale (int, optional): Image scale [cm/px]. Defaults to 1.

    Returns:
        pred_pts (np.ndarray): Centroids of heatmaps. shape=(n, 4, 2)
        pred_lens (np.ndarray): Measurement lengths. shape=(n, 3)
        sys_i (np.ndarray): Indices of end systole. shape=(n_sys,)
        dia_i (np.ndarray): Indices of end diastole. shape=(n_dia,)
        angles (np.ndarray): Angles between measurements in degrees. shape=(n, 2)
    """

    pred_pts = get_points_np(preds)
    pred_lens = get_lens_np(pred_pts) * scale
    sys_i, dia_i = get_systole_diastole(pred_lens[:, 1])
    angles = get_angles_np(pred_pts)
    return pred_pts, pred_lens, sys_i, dia_i, angles

def overlay_preds(
            a: np.ndarray, 
            background=None, 
            c=np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1], [0, 1, 0]])
        ) -> np.ndarray:
    """Used to visualize PLAX model predictions over echo frames

    Args:
        a (np.ndarray): Predicted heatmaps. shape=(h, w, 4)
        background (np.ndarray, optional): Echo frame to overlay on top of. shape=(h, w, 3) Defaults to None.
        c (np.ndarray, optional): RGB colors corresponding to each channel of the predictions. shape=(4, 3)
            Defaults to np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1], [0, 1, 0]]).

    Returns:
        np.ndarray: RGB image visualization of heatmaps. shape=(h, w, 3)
    """

    if background is None:
        background = np.zeros((a.shape[0], a.shape[1], 3))
    np.seterr(divide='ignore', invalid='ignore')
    color = (a ** 2).dot(c) / np.sum(a, axis=-1)[..., None]
    alpha = (1 - np.prod(1 - a, axis=-1))[..., None]
    alpha = np.nan_to_num(alpha)
    color = np.nan_to_num(color)
    return alpha * color + (1 - alpha) * background

def crop_and_scale(img: np.ndarray,res)->np.ndarray: #res=(640, 480)) -> np.ndarray:
    """Scales and crops an numpy array image to specified resolution.
    Image is first cropped to correct aspect ratio and then scaled using
    bicubic interpolation.

    Args:
        img (np.ndarray): Image to be resized. shape=(h, w, 3)
        res (tuple, optional): Resolution to be scaled to. Defaults to (640, 480).

    Returns:
        np.ndarray: Scaled image. shape=(res[1], res[0], 3)
    """

    in_res = (img.shape[1], img.shape[0])
    r_in = in_res[0] / in_res[1]
    r_out = res[0] / res[1]

<<<<<<< HEAD
    # if r_in > r_out:
    #     padding = int(round((in_res[0] - r_out * in_res[1]) / 2))
    #     img = img[:, padding:-padding]
    # if r_in < r_out:
    #     padding = int(round((in_res[1] - in_res[0] / r_out) / 2))
    #     img = img[padding:-padding]
=======
    if r_in > r_out:
        padding = int(round((in_res[0] - r_out * in_res[1]) / 2))
        img = img[:, padding:-padding]
    if r_in < r_out:
        padding = int(round((in_res[1] - in_res[0] / r_out) / 2))
        img = img[padding:-padding]
>>>>>>> 5044dc3 (added changes to fix inference loop)
    
    img = cv2.resize(img, res)

    return img
<<<<<<< HEAD

'''
DICOM PROCESSING + VIEW CLASSIFICATION 
'''
VIEW_MEAN = torch.tensor([29.110628, 28.076836, 29.096405], dtype=torch.float32).reshape(3,1,1,1)
VIEW_STD = torch.tensor([47.989223, 46.456997, 47.20083], dtype=torch.float32).reshape(3,1,1,1)
# ALL_VIEWS = ['A2C','A3C','A4C','A5C','Apical_Doppler','Doppler_PLAX','Doppler_PSAX','PLAX','PSAX','SSN','Subcostal']
ALL_VIEWS = [
    "A2C",
    "A2C_LV",
    "A3C",
    "A3C_LV",
    "A4C",
    "A4C_LA",
    "A4C_LV",
    "A4C_MV",
    "A4C_RV",
    "A5C",
    "PLAX",
    "PLAX_AV_MV",
    "PLAX_Zoom_out",
    "PLAX_Proximal_Ascending_Aorta",
    "PLAX_RV_inflow",
    "PLAX_RV_outflow",
    "PLAX_zoomed_AV",
    "PLAX_zoomed_MV",
    "PSAX_(level_great_vessels)",
    "PSAX_(level_great_vessels)_focus_on_PV_and_PA",
    "PSAX_(level_great_vessels)_focus_on_TV",
    "PSAX_(level_great_vessels)_zoomed_AV",
    "PSAX_(level_of_MV)",
    "PSAX_(level_of_apex)",
    "PSAX_(level_of_papillary_muscles)",
    "SSN_aortic_arch",
    "Subcostal_4C",
    "Subcostal_Abdominal_Aorta",
    "Subcostal_IVC",
    "DOPPLER_PSAX_level_great_vessels_TV",
    "DOPPLER_PSAX_level_great_vessels_PA",
    "DOPPLER_PSAX_level_great_vessels_AV",
    "DOPPLER_PLAX_AV_zoomed",
    "DOPPLER_PLAX_MV_zoomed",
    "DOPPLER_PLAX_AV_MV",
    "DOPPLER_PLAX_Ascending_Aorta",
    "DOPPLER_PLAX_IVS",
    "DOPPLER_PLAX_RVOT",
    "DOPPLER_PLAX_RVIT",
    "DOPPLER_A4C_MV_TV",
    "DOPPLER_PSAX_MV",
    "DOPPLER_A4C_MV",
    "DOPPLER_A4C_TV",
    "DOPPLER_A4C_Apex",
    "DOPPLER_A4C_IVS",
    "DOPPLER_A4C_IAS",
    "DOPPLER_A4C_IVS_IAS",
    "DOPPLER_A2C",
    "DOPPLER_PSAX_IAS",
    "DOPPLER_PSAX_IVS",
    "DOPPLER_A5C",
    "DOPPLER_A3C",
    "DOPPLER_A3C_MV",
    "DOPPLER_A3C_AV",
    "DOPPLER_SSN_Aortic_Arch",
    "DOPPLER_A4C_Pulvns",
    "DOPPLER_SC_4C_IAS",
    "DOPPLER_SC_4C_IVS",
    "DOPPLER_SC_IVC",
    "DOPPLER_SC_aorta",
    "M_mode_PLAX_Ao_LA",
    "M_mode_PLAX_MV",
    "M_mode_PLAX_LV",
    "M_mode_A4C_RV_TAPSE",
    "M_mode_SC_IVC",
    "Doppler_PLAX_RVIT_CW",
    "Doppler_PSAX_Great_vessel_level_TV_CW",
    "Doppler_PSAX_Great_vessel_level_PA_PW",
    "Doppler_PSAX_Great_vessel_level_PA_CW",
    "Doppler_A4C_MV_PW",
    "Doppler_A4C_MV_CW",
    "Doppler_A4C_IVRT_PW",
    "Doppler_A4C_PV_PW",
    "Doppler_A4C_TV_CW",
    "Doppler_A5C_AV_PW",
    "Doppler_A5C_AV_CW",
    "Doppler_A3C_AV_PW",
    "Doppler_A3C_AV_CW",
    "Doppler_SC_HV/IVC_PW",
    "Doppler_SC_abdominal_AO_PW",
    "Doppler_SSN_descending_AO_PW",
    "Doppler_SSN_descending_AO_CW",
    "TDI_MV_Lateral e",
    "TDI_MV_Medial e",
    "Doppler_A3C_MV_CW",
    "M-mode_A4C_TV_TAPSE",
    "M-mode_PSAX_MV level",
    "M-mode_PSAX_LV_PM level",
    "Doppler_A2C_MV_CW",
    "Doppler_A4C_TV_CW_Mayoview",
    "Doppler_SC_TV_CWQ",
    "TDI_TV_Lateral S",
    "M-mode_PSAX_AV",
    "Doppler_LV_midcavitary_PW",
    "Doppler_A4C_TV_PW",
]

def get_ybr_to_rgb_lut(filepath,save_lut=False):
    global _ybr_to_rgb_lut
    _ybr_to_rgb_lut = None
    # return lut if already exists
    if _ybr_to_rgb_lut is not None:
        return _ybr_to_rgb_lut
    # try loading from file
    lut_path = Path(filepath).parent / 'ybr_to_rgb_lut.npy'
    if lut_path.is_file():
        _ybr_to_rgb_lut = np.load(lut_path)
        return _ybr_to_rgb_lut
    # else generate lut
    a = np.arange(2 ** 8, dtype=np.uint8)
    ybr = np.concatenate(np.broadcast_arrays(a[:, None, None, None], a[None, :, None, None], a[None, None, :, None]), axis=-1)
    _ybr_to_rgb_lut = pydicom.pixel_data_handlers.util.convert_color_space(ybr, 'YBR_FULL', 'RGB')
    if save_lut:
        np.save(lut_path, _ybr_to_rgb_lut)
    return _ybr_to_rgb_lut

def ybr_to_rgb(filepath,pixels: np.array):
    lut = get_ybr_to_rgb_lut(filepath)
    return lut[pixels[..., 0], pixels[..., 1], pixels[..., 2]]

def change_doppler_color(dcm_path):
    ds = pydicom.dcmread(dcm_path)
    pixels = ds.pixel_array
    if ds.PhotometricInterpretation == 'MONOCHROME2':
        input_image = np.stack((pixels,)*3,axis=-1)
    elif ds.PhotometricInterpretation in ['YBR_FULL',"YBR_FULL_422"]:
        input_image = ybr_to_rgb(dcm_path,pixels)
    elif ds.PhotometricInterpretation == "RGB": 
        pass
    else:
        print("Unsupported Photometric Interpretation: ",ds.PhotometricInterpretation)
    return input_image 

def mask_outside_ultrasound(original_pixels: np.array) -> np.array:
    """
    Masks all pixels outside the ultrasound region in a video.

    Args:
    vid (np.ndarray): A numpy array representing the video frames. FxHxWxC

    Returns:
    np.ndarray: A numpy array with pixels outside the ultrasound region masked.
    """
    try:
        testarray=np.copy(original_pixels)
        vid=np.copy(original_pixels)
        ##################### CREATE MASK #####################
        # Sum all the frames
        frame_sum = testarray[0].astype(np.float32)  # Start off the frameSum with the first frame
        frame_sum = cv2.cvtColor(frame_sum, cv2.COLOR_YUV2RGB)
        frame_sum = cv2.cvtColor(frame_sum, cv2.COLOR_RGB2GRAY)
        frame_sum = np.where(frame_sum > 0, 1, 0) # make all non-zero values 1
        frames = testarray.shape[0]
        for i in range(frames): # Go through every frame
            frame = testarray[i, :, :, :].astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_YUV2RGB)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = np.where(frame>0,1,0) # make all non-zero values 1
            frame_sum = np.add(frame_sum,frame)

        # Erode to get rid of the EKG tracing
        kernel = np.ones((3,3), np.uint8)
        frame_sum = cv2.erode(np.uint8(frame_sum), kernel, iterations=10)

        # Make binary
        frame_sum = np.where(frame_sum > 0, 1, 0)

        # Make the difference frame fr difference between 1st and last frame
        # This gets rid of static elements
        frame0 = testarray[0].astype(np.uint8)
        frame0 = cv2.cvtColor(frame0, cv2.COLOR_YUV2RGB)
        frame0 = cv2.cvtColor(frame0, cv2.COLOR_RGB2GRAY)
        frame_last = testarray[testarray.shape[0] - 1].astype(np.uint8)
        frame_last = cv2.cvtColor(frame_last, cv2.COLOR_YUV2RGB)
        frame_last = cv2.cvtColor(frame_last, cv2.COLOR_RGB2GRAY)
        frame_diff = abs(np.subtract(frame0, frame_last))
        frame_diff = np.where(frame_diff > 0, 1, 0)

        # Ensure the upper left hand corner 20x20 box all 0s.
        # There is a weird dot that appears here some frames on Stanford echoes
        frame_diff[0:20, 0:20] = np.zeros([20, 20])

        # Take the overlap of the sum frame and the difference frame
        frame_overlap = np.add(frame_sum,frame_diff)
        frame_overlap = np.where(frame_overlap > 1, 1, 0)

        # Dilate
        kernel = np.ones((3,3), np.uint8)
        frame_overlap = cv2.dilate(np.uint8(frame_overlap), kernel, iterations=10).astype(np.uint8)

        # Fill everything that's outside the mask sector with some other number like 100
        cv2.floodFill(frame_overlap, None, (0,0), 100)
        # make all non-100 values 255. The rest are 0
        frame_overlap = np.where(frame_overlap!=100,255,0).astype(np.uint8)
        contours, hierarchy = cv2.findContours(frame_overlap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # contours[0] has shape (445, 1, 2). 445 coordinates. each coord is 1 row, 2 numbers
        # Find the convex hull
        for i in range(len(contours)):
            hull = cv2.convexHull(contours[i])
            cv2.drawContours(frame_overlap, [hull], -1, (255, 0, 0), 3)
        frame_overlap = np.where(frame_overlap > 0, 1, 0).astype(np.uint8) #make all non-0 values 1
        # Fill everything that's outside hull with some other number like 100
        cv2.floodFill(frame_overlap, None, (0,0), 100)
        # make all non-100 values 255. The rest are 0
        frame_overlap = np.array(np.where(frame_overlap != 100, 255, 0),dtype=bool)
        ################## Create your .avi file and apply mask ##################
        # Store the dimension values

        # Apply the mask to every frame and channel (changing in place)
        for i in range(len(vid)):
            frame = vid[i, :, :, :].astype('uint8')
            frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR)
            frame = cv2.bitwise_and(frame, frame, mask = frame_overlap.astype(np.uint8))
            vid[i,:,:,:]=frame
        return vid
    except Exception as e:
        print("Error masking returned as is.")
        return vid

def make_view_input(dcm_path,n=224):
    pixels = change_doppler_color(dcm_path)
    pixels = mask_outside_ultrasound(pixels)
    frames_to_take = 32
    frame_stride = 2
    vid_tensor = torch.as_tensor(pixels,dtype=torch.float) # FHWC
    if len(vid_tensor.size())<4:
        return
    vid_tensor = vid_tensor.permute(0,-1,1,2) #FCHW
    resizer = torchvision.transforms.Resize((n,n))
    x = resizer(vid_tensor).permute(1,0,2,3)
    x.sub_(VIEW_MEAN).div_(VIEW_STD)
    if x.shape[1] < frames_to_take:
        padding = torch.zeros((3,frames_to_take - x.shape[1],n,n),dtype=torch.float)
        x = torch.cat((x,padding),dim=1)
    return x[:,0:frames_to_take:frame_stride,:,:]

def load_view_classifier(weights_path='/workspace/vic/lvh/weights/epoch=21-step=17842.ckpt'):
    device = torch.device('cuda') if torch.cuda.is_available()==True else 'cpu'
    checkpoint = torch.load(weights_path,map_location=device)
    state_dict = {key[6:]:value for key,value in checkpoint['state_dict'].items()}
    view_classifier = torchvision.models.convnext_base()
    view_classifier.classifier[-1] = torch.nn.Linear(
        view_classifier.classifier[-1].in_features,len(ALL_VIEWS)
    )
    view_classifier.load_state_dict(state_dict)
    view_classifier.to(device)
    view_classifier.eval()
    for param in view_classifier.parameters():
        param.requires_grad = False
    return view_classifier

def get_views(input,view_classifier,filename,batch_size=32):
    device = torch.device('cuda') if torch.cuda.is_available()==True else 'cpu'
    labels = torch.zeros(len(input))
    ds = torch.utils.data.TensorDataset(input,labels)
    batch_size = batch_size
    dl = torch.utils.data.DataLoader(ds,batch_size=batch_size,num_workers=0,shuffle=False)
    pred_views = [""]*len(dl)
    with torch.no_grad():
        for idx,(x,views) in tqdm.tqdm(enumerate(dl)):
            x = x.to(device)
            out = view_classifier(x)
            out = torch.argmax(out,dim=1)
            preds = [ALL_VIEWS[o] for o in out]
            start = idx*batch_size
            end = min((idx+1)*batch_size,len(ds))
            pred_views[start:end] = preds
    predicted_view = {key:val for (key,val) in zip(filename,pred_views)}
    predicted_view = pd.DataFrame.from_dict({
        'filename':list(predicted_view.keys()),
        'view':list(predicted_view.values())
    })
    predicted_view.to_csv('predicted_view.csv',index=None)
    return predicted_view

def get_dicoms(parent,new_dir):
    new_path = Path(new_dir)/'all_dicoms'
    if not new_path.exists():
        os.mkdir(new_path)
    parent = Path(parent)
    for item in parent.iterdir(): 
        if item.is_dir():
            id = item.name
            dicom_dir = Path(item/'dicom')
            for dcm in dicom_dir.iterdir():
                dst_path = new_path/f'{id}_{dcm.name}'
                shutil.copy(src=dcm,dst=dst_path)
=======
>>>>>>> 5044dc3 (added changes to fix inference loop)
