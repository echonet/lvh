from argparse import ArgumentParser
from operator import index
from pandas.core.algorithms import isin
import torch
from models import ClassificationModel as Model
from tqdm import tqdm
from pathlib import Path
from utils import BoolAction, read_clip, model_paths
import numpy as np
import pandas as pd


class ClassificationInferenceEngine:

    def __init__(self, model_path=model_paths['amyloid'], device='cuda:0', res=(112, 112)) -> None:
        self.model_path = model_path
        self.device = device
        self.res = res
        self.model = None
    
    def load_model(self):
        self.model = Model()
        self.model.to(self.device)
        return self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))

    def run_on_dir(self, in_dir, out_dir, batch_size=4, clip_length=96, verbose=True, threshold=0.5):
        if not isinstance(in_dir, Path):
            in_dir = Path(in_dir)
        if not isinstance(out_dir, Path):
            out_dir = Path(out_dir)
        if not out_dir.exists():
            out_dir.mkdir()
        
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
        
        results = {'Filename': [], 'Positive Confidence': []}
        for paths, clips in batch_gen():
            clips = torch.from_numpy(np.moveaxis(np.array(clips), -1, 1)).to(torch.float).to(self.device) / 255.0
            with torch.no_grad():
                preds = torch.sigmoid(self.model(clips)).detach().cpu().numpy()
                results['Filename'].append([p.name for p in paths])
                results['Positive Confidence'].append(preds[:, 0])
        results = pd.DataFrame({k: np.concatenate(v) for k, v in results.items()})
        n_pos = (results['Positive Confidence'] > threshold).sum()
        print(f'{n_pos}/{len(results)} ({100 * n_pos / len(results):.2f}%) predicted positive')
        results.to_csv(out_dir / (in_dir.name + '.csv'), index=False)


if __name__ == '__main__':

    args = {
        'device': 'cuda:0',
        'batch_size': 8,
        'clip_length': 96,
        'verbose': True,
        'model_path': model_paths['amyloid'],
        'threshold': 0.5
    }
    parser = ArgumentParser()
    parser.add_argument('in_dir', type=str)
    parser.add_argument('out_dir', type=str)
    for k, v in args.items():
        if isinstance(v, bool):
            parser.add_argument('--' + k.replace('_', '-'), action=BoolAction, default=v)
        else:
            parser.add_argument('--' + k.replace('_', '-'), type=type(v), default=v)
    args.update({k.replace('-', '_'): v for k, v in vars(parser.parse_args()).items()})
    get_args = lambda l: {k: args[k] for k in l}

    engine = ClassificationInferenceEngine(**get_args(['device', 'model_path']))
    print(engine.load_model())
    engine.run_on_dir(**get_args(['in_dir', 'out_dir', 'batch_size', 'clip_length', 'verbose', 'threshold']))
