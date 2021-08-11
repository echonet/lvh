from argparse import ArgumentParser
from operator import index, mod
from pandas.core.algorithms import isin
import torch
from models import ClassificationModel as Model
from tqdm import tqdm
from pathlib import Path
from utils import BoolAction, read_clip, model_paths
import numpy as np
import pandas as pd
from typing import Union


class A4cClassificationInferenceEngine:

    def __init__(self, model_path: Union[Path, str]=model_paths['amyloid'], device: str='cuda:0', res=(112, 112)) -> None:
        """Create a A4cClassificationInferenceEngine instance used for running classification inference.

        Args:
            model_path (Union[Path, str], optional): Path to saved model weights. Defaults to model_paths['amyloid'].
            device (str, optional): Device to run model on. Defaults to 'cuda:0'.
            res (tuple, optional): Image resolution. Defaults to (112, 112).
        """
        if isinstance(model_path, str):
            model_path = Path(model_path)
        self.model_path = model_path
        self.device = device
        self.res = res
        self.model = None
    
    def load_model(self):
        """Loads model onto device.

        Returns:
            Results of model.load_state_dict()
        """
        self.model = Model()
        self.model.to(self.device)
        return self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))

    def run_on_dir(self, 
                in_dir: Union[Path, str], 
                out_dir: Union[Path, str], 
                batch_size: int=4, clip_length: int=96, 
                verbose: bool=True, threshold: float=0.5
            ) -> None:
        """Runs inference on all .avi files in a directory. Saves results to a .csv file.

        Args:
            in_dir (Union[Path, str]): Directory containing .avi files to run inference on.
            out_dir (Union[Path, str]): Directory to save .csv file to
            batch_size (int, optional): Batch size for running inference. Defaults to 4.
            clip_length (int, optional): Number of frames used to run inference. The first 
                n frames of the video are used. Any videos shorter than this length are ignored. Defaults to 96.
            verbose (bool, optional): Prints progress and stats while running. Defaults to True.
            threshold (float, optional): Threshold used to consider classification positive. Defaults to 0.5.
        """

        # Prepare directories
        if not isinstance(in_dir, Path):
            in_dir = Path(in_dir)
        if not isinstance(out_dir, Path):
            out_dir = Path(out_dir)
        if not out_dir.exists():
            out_dir.mkdir()
        
        # Load model if not loaded already
        if self.model is None:
            self.load_model()
        
        # Yield batches of videos from directory
        def batch_gen():
            batch = ([], [])
            for p in tqdm(list(in_dir.iterdir())) if verbose else in_dir.iterdir():
                if '.avi' not in p.name:
                    continue
                clip = read_clip(p, res=self.res, max_len=clip_length)
                if len(clip) != clip_length:
                    continue
                batch[0].append(p)
                batch[1].append(clip)
                if len(batch[0]) == batch_size:
                    yield batch[0], np.array(batch[1])
                    batch = ([], [])
            if len(batch[0]) != 0:
                yield batch[0], np.array(batch[1])
        
        # Run inference
        results = {'Filename': [], 'Positive Confidence': []}
        for paths, clips in batch_gen():
            clips = torch.from_numpy(np.moveaxis(np.array(clips), -1, 1)).to(torch.float).to(self.device) / 255.0
            with torch.no_grad():
                preds = torch.sigmoid(self.model(clips)).detach().cpu().numpy()
                results['Filename'].append([p.name for p in paths])
                results['Positive Confidence'].append(preds[:, 0])
        
        # Process results and save to .csv
        results = pd.DataFrame({k: np.concatenate(v) for k, v in results.items()})
        n_pos = (results['Positive Confidence'] > threshold).sum()
        print(f'{n_pos}/{len(results)} ({100 * n_pos / len(results):.2f}%) predicted positive')
        results.to_csv(out_dir / (in_dir.name + '.csv'), index=False)


if __name__ == '__main__':

    # CLI used to run inference on a directory and save to
    # .csv in output directory. Run python run_classification_inference.py --help for
    # information about parameters.

    args = {
        'device': ('cuda:0', 'Device to run inference on. Ex: "cuda:0" or "cpu"'),
        'batch_size': (8, 'Number of videos to run inference on at once.'),
        'clip_length': (96, 'Number of frames to run inference on. The first N frames will be used.'),
        'verbose': (True, 'Print progress and statistics while running. y/n'),
        'model_path': (model_paths['amyloid'], f'Path to model weights. default={model_paths["amyloid"]}'),
        'threshold': (0.5, 'Model predictions above this level will be classified as positive.'),
    }
    parser = ArgumentParser()
    parser.add_argument('in_dir', type=str, help='Directory containing .avi\' to run inference on.')
    parser.add_argument('out_dir', type=str, help='Direcotry to output predictions to.')
    for k, (v, h) in args.items():
        if isinstance(v, bool):
            parser.add_argument('--' + k.replace('_', '-'), action=BoolAction, default=v, help=h)
        else:
            parser.add_argument('--' + k.replace('_', '-'), type=type(v), default=v, help=h)
    args.update({k.replace('-', '_'): v for k, v in vars(parser.parse_args()).items()})
    get_args = lambda l: {k: args[k][0] for k in l}

    # Run inference
    engine = A4cClassificationInferenceEngine(**get_args(['device', 'model_path']))
    engine.run_on_dir(**get_args(['in_dir', 'out_dir', 'batch_size', 'clip_length', 'verbose', 'threshold']))
