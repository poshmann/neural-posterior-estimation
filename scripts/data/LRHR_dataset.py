from torch.utils.data import Dataset
import csv
import data.util as Util
import tifffile
import numpy as np
import os


def to_hwc(img):
    """Return a (H, W, C) array whatever layout the tile is stored in.
    The band axis is the one of length 4 (R, G, B, NIR). PlanetScope tiles are stored
    (H, W, C), the original Sentinel-2 tiles (C, H, W)."""
    img = np.asarray(img)
    if img.ndim == 2:
        return img[..., None]
    if img.shape[-1] == 4:
        return img
    if img.shape[0] == 4:
        return np.moveaxis(img, 0, -1)
    raise ValueError('Cannot find a 4-band axis in tile of shape {}'.format(img.shape))


class LRHRDataset(Dataset):
    """Paired HR / input tiles.

    RQ3 changes (see RQ3_CHANGES.md):
      * `hr_dir` / `sr_dir` can be set in the config (absolute, or relative to dataroot);
        default is the original `hr_<r>` / `sr_<l>_<r>` naming.
      * both tiles are converted to (H, W, C) by finding the 4-band axis, so Sentinel-2
        (C,H,W) and PlanetScope / fake-LR (H,W,C) files both load correctly.
      * HR and input are matched by filename stem, never by sort order.
    """

    def __init__(self, dataroot, datatype, l_resolution=16, r_resolution=128, split='train', data_len=-1,
                 need_LR=False, index_csv=None, hr_dir=None, sr_dir=None, lr_dir=None):
        self.datatype = datatype
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.data_len = data_len
        self.need_LR = need_LR
        self.split = split
        self.dataroot = dataroot
        self.index_csv = index_csv

        def resolve(d, default):
            d = d or default
            return d if os.path.isabs(d) else os.path.join(dataroot, d)

        sr_dir = resolve(sr_dir, 'sr_{}_{}'.format(l_resolution, r_resolution))
        hr_dir = resolve(hr_dir, 'hr_{}'.format(r_resolution))
        lr_dir = resolve(lr_dir, 'lr_{}'.format(l_resolution))

        if self.index_csv:
            stems = self._read_stems_from_csv(self.index_csv)
        else:
            stems = sorted(os.path.splitext(os.path.basename(p))[0]
                           for p in Util.get_paths_from_images(hr_dir))
        self.stems = stems
        self.sr_path = self._build_paths_from_stems(sr_dir, stems)
        self.hr_path = self._build_paths_from_stems(hr_dir, stems)
        if self.need_LR:
            self.lr_path = self._build_paths_from_stems(lr_dir, stems)

        self.dataset_len = len(self.hr_path)
        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.data_len, self.dataset_len)

    def __len__(self):
        return self.data_len

    def _read_stems_from_csv(self, csv_path):
        if not os.path.isfile(csv_path):
            raise FileNotFoundError('CSV file not found: {}'.format(csv_path))
        stems = []
        with open(csv_path, 'r', newline='') as handle:
            for row in csv.reader(handle):
                if not row or not row[0].strip():
                    continue
                value = row[0].strip()
                if value.lower() in {'filename', 'file', 'stem', 'id', 'name'}:
                    continue
                stems.append(os.path.splitext(os.path.basename(value))[0])
        if not stems:
            raise ValueError('No file stems found in first CSV column: {}'.format(csv_path))
        return stems

    def _build_paths_from_stems(self, image_dir, stems):
        # direct lookup: avoids walking a 115k-file folder for every split
        paths, missing = [], []
        for stem in stems:
            p = os.path.join(image_dir, stem + '.tif')
            if os.path.isfile(p):
                paths.append(p)
            else:
                missing.append(stem)
        if missing:
            raise ValueError('Missing {} stems in {}. First missing: {}'.format(
                len(missing), image_dir, ', '.join(missing[:10])))
        return paths

    def __getitem__(self, index):
        img_HR = to_hwc(tifffile.imread(self.hr_path[index])).astype(np.float32)
        img_SR = to_hwc(tifffile.imread(self.sr_path[index])).astype(np.float32)
        # top-left crop to 192 + [0,1] -> [-1,1] + random flips (train only)
        [img_SR, img_HR] = Util.transform_augment([img_SR, img_HR], split=self.split, min_max=(-1, 1))
        return {'HR': img_HR, 'SR': img_SR, 'Index': index,
                'HR_path': self.hr_path[index], 'SR_path': self.sr_path[index]}
