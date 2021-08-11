# docker exec -it -w /workspace/Grant/echo_amyloid grant python -W ignore /workspace/Grant/echo_amyloid/run_plax_inference.py

import argparse
from operator import mod
from typing import Union
from argparse import ArgumentParser
import cv2
import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
from matplotlib import animation
import torch
from pathlib import Path
from tqdm import tqdm
from scipy.signal import find_peaks
import shutil
import sys
from threading import Thread, Lock

from utils import BoolAction, get_clip_dims, read_clip, get_systole_diastole, get_lens_np, get_points_np
from utils import get_angles_np, get_pred_measurements, overlay_preds, model_paths
from models import PlaxModel as Model


plt_thread_lock = Lock()


def save_preds(
            p: Union[str, Path], fn: str, clip: np.ndarray, preds: np.ndarray, 
            csv=True, avi=True, plot=True, npy=False, angle_threshold=30
        ) -> None:
    """Save results for model predictions. All results are saved to a new directory within the 
    output path with the name [fn]. Frame-by-frame analysis to predict ES and ED is used and 
    results are saved to [fn].csv.

    Args:
        p (Union[str, Path]): output path to save to
        fn (str): output filename
        clip (np.ndarray): shape=(n, h, w, 3), input clip used to run inference
        preds (np.ndarray): shape=(n, h, w, 4), model predictions from input clip
        csv (bool, optional): Save frame-by-frame results to .csv. Defaults to True.
        avi (bool, optional): Save prediction animation as .avi. Defaults to True.
        plot (bool, optional): Save measurement vs time plot as .png. Defaults to True.
        npy (bool, optional): Save raw predictions as numpy .npy. Defaults to False.
        angle_threshold (int, optional): Angle between measurement lines above which are 
            considered 'bad'. Defaults to 30.
    """

    # Create folder
    folder_name = fn.replace('.avi', '').replace('.', '_')
    inf_path = p / folder_name
    if not inf_path.exists():
        inf_path.mkdir()
    
    # Save raw predictions as .npy
    if npy:
        np.save(inf_path / (folder_name + '.npy'), preds)
    pred_pts, pred_lens, sys_i, dia_i, angles = get_pred_measurements(preds)

    # Save predicted points to .csv
    if csv:
        phase = np.array([''] * len(pred_pts), dtype=np.object)
        phase[sys_i] = 'ES'
        phase[dia_i] = 'ED'
        df = pd.DataFrame({
            'frame': np.arange(len(pred_pts)),
            'X1': pred_pts[:, 0, 0],
            'Y1': pred_pts[:, 0, 0],
            'X2': pred_pts[:, 1, 0],
            'Y2': pred_pts[:, 1, 0],
            'X2': pred_pts[:, 2, 0],
            'Y3': pred_pts[:, 2, 0],
            'X4': pred_pts[:, 3, 0],
            'Y4': pred_pts[:, 3, 0],
            'LVPW': pred_lens[:, 0],
            'LVID': pred_lens[:, 1],
            'IVS': pred_lens[:, 2],
            'predicted_phase': phase,
            'LVPW_LVID_angle': angles[:, 0],
            'LVID_IVS_angle': angles[:, 1],
            'bad_angle': (abs(angles[:, 0]) > angle_threshold) | (abs(angles[:, 1]) > angle_threshold)
        })
        df.set_index('frame')
        df.to_csv(inf_path / (folder_name + '.csv'))

    # Save an animation of the predictions overlayed on the cropped video as .avi
    if avi:
        with plt_thread_lock:
            # make_animation(inf_path / (folder_name + '.avi'), clip, preds, pred_pts, pred_lens, sys_i, dia_i)
            make_animation_cv2(inf_path / (folder_name + '.avi'), clip, preds, pred_pts)
    
    # Save a plot of measurements vs time for whole clip
    if plot:
        make_plot(inf_path / (folder_name + '.png'), folder_name, pred_lens, sys_i, dia_i)

def make_animation(
            save_path: Union[Path, str], clip: np.ndarray, preds: np.ndarray, 
            pred_pts: np.ndarray, pred_lens: np.ndarray, sys_i, dia_i, 
            figsize=(12, 12), units='PX', fps=50
        ) -> None:
    
    """Save animation of predictions using matplotlib.

    Args:
        save_path (Union[Path, str]): Location to save animation .avi to
        clip (np.ndarray): shape=(n, h, w, 3). Input video used for inference.
        preds (np.ndarray): shape=(n, h, w, 4). Raw model predictions.
        pred_pts (np.ndarray): shape=(n, 4, 2). Predicted points for measurements.
        pred_lens (np.ndarray): shape=(n, 3). Predicted measurement values [LVPW, LVID, IVS]
        sys_i (array-like): indices predicted to be end systole
        dia_i (array-like): indices predicted to be end diastole
        figsize (tuple, optional): plt figure size. Defaults to (12, 12).
        units (str, optional): Units to show on plot. Defaults to 'PX'.
        fps (int, optional): Frame rate so save animation in. Defaults to 50.
    """
    
    # Ensure Path object
    if isinstance(save_path, str):
        save_path = Path(save_path)
    
    # Setup figure layout and static plot
    grid = plt.GridSpec(4, 1)
    fig = plt.figure(0, figsize=figsize)
    ax1 = fig.add_subplot(grid[3:, 0])
    ax2 = fig.add_subplot(grid[:3, 0])
    for l, n in zip(pred_lens.T, ['LVPW', 'LVID', 'IVS']):
        ax1.plot(l, label=n)
    l1, = ax1.plot([0, 0, 0], pred_lens[0], 'ro')
    ax1.legend()
    ax1.set_xlabel('Frame')
    ax1.set_ylabel(f'Measurement [{units}]')
    ax1.vlines(sys_i, pred_lens.min(), pred_lens.max(), linestyles='dashed', colors='b', label='Systole')
    ax1.vlines(dia_i, pred_lens.min(), pred_lens.max(), linestyles='dashed', colors='g', label='Diastole')
    im = ax2.imshow(overlay_preds(preds[0], clip[0] / 255))
    l2, = ax2.plot(*pred_pts[0].T, 'C1o-')
    ax2.set_title(save_path.name)

    # Modifies plot for each frame of animation
    def animate(i):
        im.set_data(overlay_preds(preds[i], clip[i] / 255))
        l1.set_data([i, i , i], pred_lens[i])
        l2.set_data(*pred_pts[i].T)

    # Save animation
    ani = animation.FuncAnimation(fig, animate, frames=len(clip), interval=1000 / fps)
    writer = animation.FFMpegWriter(fps)
    ani.save(save_path, writer)

    del fig

def make_plot(
            save_path: Union[Path, str], title: str, pred_lens: np.ndarray, 
            sys_i, dia_i, figsize=(8, 6)
        ) -> None:
    """Save a plot showing measurement values over time.

    Args:
        save_path (Union[Path, str]): .png path to save plot to.
        title (str): Plot title.
        pred_lens (np.ndarray): shape=(n, 3). Predicted measurements [LVPW, LVID, IVS]
        sys_i (array-like): indices predicted to be end systole
        dia_i (array-like): indices predicted to be end diastole
        figsize (tuple, optional): plt figure size. Defaults to (8, 6).
    """

    plt.figure(1, figsize=figsize)
    plt.clf()
    for l, n in zip(pred_lens.T, ['LVPW', 'LVID', 'IVS']):
        plt.plot(l, label=n)
    plt.plot(sys_i, pred_lens[sys_i], 'r+')
    plt.plot(dia_i, pred_lens[dia_i], 'rx')
    plt.plot([], [], 'rx', label='Diastole')
    plt.plot([], [], 'r+', label='Systole')
    plt.title(title)
    plt.xlabel('Frame')
    plt.ylabel('Measurement [px]')
    plt.legend()
    plt.savefig(save_path)


def make_animation_cv2(
            save_path: Union[Path, str], clip: np.ndarray, preds: np.ndarray, pred_pts: np.ndarray, 
            fps=30.0, line_color=(1, 1, 0), point_color=(1, 0.5, 0), linewidth=2, markersize=4
        ) -> None:
    """Creates an animation with predictions overlayed on top of the clip.

    Args:
        save_path (Union[Path, str]): .avi path to save animation to.
        clip (np.ndarray): shape=(n, h, w, 3). Input video used for inference.
        preds (np.ndarray): shape=(n, h, w, 4). Raw model predictions.
        pred_pts (np.ndarray): shape=(n, 4, 2). Predicted points for measurements.
        fps (float, optional): Animation frame rate. Defaults to 30.0.
        line_color (tuple, optional): Color of measurement lines. Defaults to (1, 1, 0).
        point_color (tuple, optional): Color of measurement endpoints. Defaults to (1, 0.5, 0).
        linewidth (int, optional): Width of measurement lines. Defaults to 2.
        markersize (int, optional): Size of measurement endpoints. Defaults to 4.
    """
    out = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*'MJPG'), fps, (clip.shape[2], clip.shape[1]))
    for frame, pred, line in zip(clip, preds, pred_pts):
        
        # Overlay raw predictions
        img = overlay_preds(pred, frame / 255)
        if not np.isnan(line).any():
            line = line.round().astype(int)
        
            # Draw measurement
            for pt0, pt1 in zip(line[:-1], line[1:]):
                img = cv2.line(img, tuple(pt0), tuple(pt1), line_color, linewidth)
            for pt in line:
                img = cv2.circle(img, tuple(pt), radius=markersize, color=point_color, thickness=-1)
        
        # Write to file
        img = (img * 255).astype(np.uint8)
        out.write(img[:, :, ::-1])
    
    # Close file
    out.release()


class PlaxHypertrophyInferenceEngine:

    def __init__(
                self, model_path: Union[Path, str]=model_paths['plax'],
                device='cuda:0', 
            ) -> None:
        if isinstance(model_path, str):
            model_paths = Path(model_path)
        self.device = device
        self.model = None
        self.figure = None
        self.model_path = model_path
    
    def load_model(self):
        """Loads the model to be used for inference into memory.

        Returns:
            NamedTuple: See torch.Module.load_state_dict() for details
        """
        self.model = Model().eval().to(self.device)
        return self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
    
    def run_on_dir(
                self, in_dir: Union[Path, str], out_dir: Union[Path, str], batch_size=100, 
                h=480, w=640, channels_in=3, channels_out=4, 
                n_threads=16, verbose=True, save_csv=True, save_avi=True, save_npy=False, save_plot=True
            ) -> None:
        """Run inference on PLAX videos in a directory

        Args:
            in_dir (Union[Path, str]): Directory of PLAX videos to run inference on.
            out_dir (Union[Path, str]): Directory to save results to.
            batch_size (int, optional): Batch size in frames to use when running model. Defaults to 100.
            h (int, optional): Video height. Defaults to 480.
            w (int, optional): Video width. Defaults to 640.
            channels_in (int, optional): Input channels, RGB=3. Defaults to 3.
            channels_out (int, optional): Output channels, 4 for LV measurement points. Defaults to 4.
            n_threads (int, optional): Number of threads to use while generating batches and saving inferences. Defaults to 16.
            verbose (bool, optional): Print progress and updates. Defaults to True.
            save_csv (bool, optional): Save CSV for each video with all predicted values for every frame. Defaults to True.
            save_avi (bool, optional): Save input model with inference overlay. Defaults to True.
            save_npy (bool, optional): Save raw predictions in numpy form. Defaults to False.
        """

        # Prepare
        in_dir = Path(in_dir) if isinstance(in_dir, str) else in_dir
        out_dir = Path(out_dir) if isinstance(out_dir, str) else out_dir
        p = lambda s: print(s) if verbose else None
        if not out_dir.exists():
            out_dir.mkdir()

        # Start inference threads. Run inference, save results to out_dir
        p('Running inference')
        threads = []
        for fn, clip, preds in engine._run_on_clips(list(in_dir.iterdir()), verbose=verbose, 
                                h=h, w=w, channels_in=channels_in, channels_out=channels_out,
                                batch_size=batch_size):
            if len(threads) > n_threads:
                t = threads.pop(0)
                t.join()
            t = Thread(target=save_preds, args=(out_dir, fn.name, clip, preds, save_csv, save_avi, save_plot, save_npy))
            t.start()
            threads.append(t)
        
        # Wait for remaining threads to finish
        p('Waiting for threads to finish')
        for t in tqdm(threads) if verbose else threads:
            t.join()
        p('Finished')

    def _run_on_clips(
                self, paths, batch_size=100, h=480, w=640, channels_in=3, channels_out=4, verbose=True
            ) -> None:
        """Internally used to iterate through a list of paths and run inferece. Batches may share frames from
        several clips. Yields clips, predictions, and filenames when inference is finished. Used for performance.
        """
        
        # Prepare
        (n, w_all, h_all), fnames = get_clip_dims(paths)
        frame_map = pd.DataFrame({
            'frame': np.concatenate([np.arange(ni) for ni in n]),
            'path': np.concatenate([np.array([str(p)] * ni, dtype=np.object) for ni, p in zip(n, paths)]),
        })

        # Run
        clips = dict()  # clips currently being processed
        batch = np.zeros((batch_size, h, w, channels_in))
        for si in tqdm(range(0, len(frame_map), batch_size)) if verbose else range(0, len(frame_map), batch_size):
            
            # Get batch files
            batch_map = frame_map.iloc[si:min(si + batch_size, len(frame_map))]
            batch_paths = batch_map['path'].unique()
            l = list(clips.items())
            
            # Check if inference has finished for all current clips
            # and yield results for any finished.
            for k, v in l:
                if k not in batch_paths:
                    clips.pop(k)
                    yield Path(k), v[0], v[1]

            # Generate batch
            for p in batch_paths:
                if p not in clips:
                    c = read_clip(p, res=(w, h))
                    clips[p] = (c, np.zeros((len(c), h, w, channels_out), dtype=np.float))
                batch[:len(batch_map)][batch_map['path'] == p] = clips[p][0][batch_map[batch_map['path'] == p]['frame']]
            
            # Run inference and set results
            preds = self.run_model_np(batch[:len(batch_map)])
            for p in batch_paths:
                clips[p][1][batch_map[batch_map['path'] == p]['frame']] = preds[batch_map['path'] == p]
    

    def run_model_np(self, x: np.ndarray) -> np.ndarray:
        """Run inference on a numpy array video.

        Args:
            x (np.ndarray): shape=(n, h, w, 3)

        Returns:
            (np.ndarray): shape=(n, h, w, 4). Raw model predictions.
        """
        if self.model is None:
            self.load_model()
        input_tensor = torch.from_numpy(np.moveaxis(x, -1, 1) / 255).to(torch.float).to(self.device)
        with torch.no_grad():
            preds_tensor = self.model(input_tensor)
        preds = np.moveaxis(preds_tensor.detach().cpu().numpy(), 1, -1)
        return preds

if __name__ == '__main__':
    
    # CLI Interface for running inference on a directory
    # and saving predictions to an output directory.
    args = {
        'device': ('cuda:0', 'Device to run inference on. Ex: "cuda:0" or "cpu"'),
        'verbose': (True, 'Print progress and statistics while running. y/n'),
        'batch_size': (100, 'Number of frames to run inference on at once.'),
        'model_path': (model_paths['plax'], f'Path to model weights.'),
        'n_threads': (16, 'Number of threads to use while generating animations and results.'),
        'save_csv': (True, 'Save frame-by-frame predictions to .csv'),
        'save_avi': (True, 'Save animation with predictions overlaid on input video to .avi file.'),
        'save_npy': (False, 'Save raw model predictions in numpy format to .npy file.'),
    }
    parser = ArgumentParser()
    parser.add_argument('in_dir', type=str, help='Directory containing .avi\' to run inference on.')
    parser.add_argument('out_dir', type=str, help='Direcotry to output predictions to.')
    for k, (v, h) in args.items():
        h += f' default={v}'
        if isinstance(v, bool):
            parser.add_argument('--' + k.replace('_', '-'), action=BoolAction, default=v, help=h)
        else:
            parser.add_argument('--' + k.replace('_', '-'), type=type(v), default=v, help=h)
    args.update({k.replace('-', '_'): v for k, (v, h) in vars(parser.parse_args()).items()})
    get_args = lambda *l: {k: args[k][0] for k in l}

    # Run inference
    engine = PlaxHypertrophyInferenceEngine(**get_args('device', 'model_path'))
    engine.run_on_dir(**get_args('in_dir', 'out_dir', 'batch_size', 'n_threads', 'verbose', 'save_csv', 'save_avi', 'save_npy'))
