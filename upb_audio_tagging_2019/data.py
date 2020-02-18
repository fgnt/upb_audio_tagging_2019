
import json
import numbers
from pathlib import Path

import numpy as np
import samplerate
import soundfile
from cached_property import cached_property
from lazy_dataset import Dataset, FilterException
from lazy_dataset.core import DynamicTimeSeriesBucket
from scipy.interpolate import make_interp_spline
from scipy.signal import stft
from skimage.transform import warp
from tqdm import tqdm


def split_dataset(dataset, fold, nfolfds=5, seed=0):
    """

    Args:
        dataset:
        fold:
        nfolfds:
        seed:

    Returns:

    >>> split_dataset(np.array([1,2,3,4,5]), 0, nfolfds=2)
    [array([1, 3]), array([2, 4, 5])]
    >>> split_dataset(np.array([1,2,3,4,5]), 1, nfolfds=2)
    [array([1, 3]), array([2, 4, 5])]
    """
    indices = np.arange(len(dataset))
    np.random.RandomState(seed).shuffle(indices)
    folds = np.split(
        indices,
        np.linspace(0, len(dataset), nfolfds + 1)[1:-1].astype(np.int64)
    )
    validation_indices = folds.pop(fold)
    training_indices = np.concatenate(folds)
    return [
        dataset[sorted(indices.tolist())]
        for indices in (training_indices, validation_indices)
    ]


class MixUpDataset(Dataset):
    def __init__(
            self,
            input_dataset,
            mixin_dataset,
            mixin_probs
    ):
        """
        Combines examples from input_dataset and mixin_dataset into tuples.

        Args:
            input_dataset: lazy dataset providing example dict with key audio_length.
            mixin_dataset: lazy dataset providing example dict with key audio_length.
            mixin_probs: list of probabilities of the number of mixture components.
        """
        self.input_dataset = input_dataset
        self.mixin_dataset = mixin_dataset
        self.mixin_probs = mixin_probs
        self.input_ranks = {
            key: input_dataset[key]['audio_length']
            for key in input_dataset.keys()
        }
        self.mixin_ranks = {
            key: mixin_dataset[key]['audio_length']
            for key in mixin_dataset.keys()
        }
        self.ranked_mixins = sorted(
            self.mixin_ranks.items(), key=lambda x: x[1]
        )
        self.mixin_lookup = {}
        i = 0
        for key, value in sorted(self.input_ranks.items(), key=lambda x: x[1]):
            while i < len(self.ranked_mixins) and 0.9 * self.ranked_mixins[i][1] < value:
                i += 1
            self.mixin_lookup[key] = i

    def __len__(self):
        return len(self.input_dataset)

    def __iter__(self):
        for key in self.keys():
            yield self[key]

    def keys(self):
        return self.input_dataset.keys()

    def copy(self, freeze=False):
        return self.__class__(
            input_dataset=self.input_dataset.copy(freeze=freeze),
            mixin_dataset=self.mixin_dataset.copy(freeze=freeze),
            mixin_probs=self.mixin_probs
        )

    @property
    def indexable(self):
        return True

    def __getitem__(self, item):
        max_mixins = np.random.choice(
            len(self.mixin_probs), p=self.mixin_probs
        )
        if isinstance(item, str):
            key = item
        elif isinstance(item, numbers.Integral):
            key = self.keys()[item]
        else:
            return super().__getitem__(item)
        mixins = []
        max_mixin_idx = self.mixin_lookup[key]
        n_mixins = min(max_mixins, max_mixin_idx)
        n = 0
        c = 0
        while n < n_mixins:
            idx = np.random.choice(max_mixin_idx)
            try:
                mixin = self.mixin_dataset[self.ranked_mixins[int(idx)][0]]
                mixins.append(mixin)
                n += 1
            except FilterException:
                c += 1
            if c > 100*n_mixins:
                break
        return [self.input_dataset[item], *mixins]


class AudioReader:
    def __init__(
            self,
            input_sample_rate=44100,
            target_sample_rate=44100
    ):
        """
        Reads an audio from file, extracts normalized log mel spectrogram
        features and prepares nhot encodings of the active events.

        Args:
            input_sample_rate:
            target_sample_rate:
        """
        self.input_sample_rate = input_sample_rate
        self.target_sample_rate = target_sample_rate

    def __call__(self, example):
        if isinstance(example, (list, tuple)):
            example = [self.read_audio(ex) for ex in example]
            example = self.mixup(example)
        else:
            assert isinstance(example, dict)
            example = self.read_audio(example)
        return example

    def read_audio(self, example):
        if example['audio_length'] > 30.:
            raise FilterException
        audio_path = example["audio_path"]
        start_sample = 0
        if "audio_start_samples" in example:
            start_sample = example["audio_start_samples"]
        stop_sample = None
        if "audio_stop_samples" in example:
            stop_sample = example["audio_stop_samples"]

        audio, sr = soundfile.read(
            audio_path, start=start_sample, stop=stop_sample, always_2d=True
        )
        assert sr == self.input_sample_rate
        if self.target_sample_rate != sr:
            audio = samplerate.resample(
                audio, self.target_sample_rate / sr, "sinc_fastest"
            )
        audio = audio.T
        example["audio_data"] = audio
        return example

    def mixup(self, examples):
        assert len(examples) > 0
        if len(examples) == 1:
            return examples[0]

        ref_value = np.abs(examples[0]['audio_data']).max()
        events = set(examples[0]['events'])

        start_indices = [0]
        stop_indices = [examples[0]['audio_data'].shape[-1]]
        for example in examples[1:]:
            min_start = max(-example['audio_data'].shape[-1], np.max(stop_indices) - 30 * 44100)
            max_start = min(
                examples[0]['audio_data'].shape[-1],
                np.min(start_indices) + 30 * 44100 - example['audio_data'].shape[-1]
            )
            if max_start < min_start:
                raise FilterException
            start_indices.append(
                int(min_start + np.random.rand() * (min_start + max_start + 1))
            )
            stop_indices.append(start_indices[-1] + example['audio_data'].shape[-1])
        start_indices = np.array(start_indices)
        stop_indices = np.array(stop_indices)
        stop_indices -= start_indices.min()
        start_indices -= start_indices.min()

        mixed_audio = np.zeros((*examples[0]['audio_data'].shape[:-1], stop_indices.max()))
        for example, start, stop in zip(examples, start_indices, stop_indices):
            audio = example['audio_data']
            scale = ref_value / max(np.abs(audio).max(), 1e-3)
            scale *= (1. + np.random.rand()) ** (2*np.random.choice(2) - 1.)
            audio *= scale
            mixed_audio[..., start:stop] += audio

            events.update(example['events'])
        # mixed_audio += np.random.rand() * np.random.randn(mixed_audio.shape[-1]) * ref_value / 10.
        is_noisy = np.array([example['is_noisy'] for example in examples]).any()
        mix = {
            'example_id': examples[0]['example_id'],
            'dataset': examples[0]['dataset'],
            'audio_data': mixed_audio,
            'audio_length': mixed_audio.shape[-1] / self.target_sample_rate,
            'events': list(events),
            'is_noisy': is_noisy
        }
        return mix


class STFT:
    def __init__(
            self,
            frame_step: int,
            fft_length: int,
            frame_length: int = None,
            window: str = "blackman",
            pad: bool = True,
            always3d: bool = False
    ):
        """
        Transforms audio data to STFT.

        Args:
            frame_step:
            fft_length:
            frame_length:
            window:
            pad:
            always3d:

        >>> stft = STFT(160, 512)
        >>> audio_data=np.zeros((1,8000))
        >>> x = stft(audio_data)
        >>> x.shape
        (1, 51, 257)
        """
        self.frame_step = frame_step
        self.fft_length = fft_length
        self.frame_length = frame_length if frame_length is not None \
            else fft_length
        self.window = window
        self.pad = pad
        self.always3d = always3d

    def stft(self, x):
        x = stft(
            x,
            nperseg=self.frame_length,
            noverlap=self.frame_length - self.frame_step,
            nfft=self.fft_length,
            window=self.window,
            axis=-1,
            padded=self.pad
        )[-1]  # (..., F, T)
        x = np.moveaxis(x, -2, -1)

        if self.always3d:
            if x.ndim == 2:
                x = x[None]  # (C, T, F)
            assert x.ndim == 3

        return x

    def __call__(self, example):
        audio = example["audio_data"]
        x = self.stft(audio)
        example["features"] = np.stack([x.real, x.imag], axis=-1)
        return example


class MelTransform:
    def __init__(
            self,
            sample_rate: int,
            fft_length: int,
            n_mels: int = 40,
            fmin: int = 20,
            fmax: int = None,
            always3d: bool = False
    ):
        """
        Transforms linear spectrogram to (log) mel spectrogram.

        Args:
            sample_rate: sample rate of audio signal
            fft_length: fft_length used in stft
            n_mels: number of filters to be applied
            fmin: lowest frequency (onset of first filter)
            fmax: highest frequency (offset of last filter)
            always3d: always return 3d array (C, T, F)
        """
        self.sample_rate = sample_rate
        self.fft_length = fft_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.always3d = always3d

    @cached_property
    def fbanks(self):
        import librosa
        fbanks = librosa.filters.mel(
            n_mels=self.n_mels,
            n_fft=self.fft_length,
            sr=self.sample_rate,
            fmin=self.fmin,
            fmax=self.fmax,
            htk=True,
            norm=None
        )
        fbanks = fbanks / fbanks.sum(axis=-1, keepdims=True)
        return fbanks.T

    def transform(self, x):
        x = np.dot(x, self.fbanks)
        if self.always3d:
            if x.ndim == 2:
                x = x[None]
            assert x.ndim == 3
        return x

    def __call__(self, example):
        x = np.sum(example["features"]**2, axis=-1)
        x = self.transform(x)
        x = np.log(np.maximum(x, 1e-18))
        example["features"] = x
        return example


class Normalizer:
    def __init__(self, storage_dir=None):
        self.moments = None
        self.storage_dir = storage_dir

    def __call__(self, example):
        assert self.moments is not None
        if isinstance(self.moments, (list, tuple)):
            mean, scale = self.moments
        else:
            assert isinstance(self.moments, dict)
            dataset = example['dataset']
            mean, scale = self.moments[dataset]
        example['features'] -= mean
        example['features'] /= (scale + 1e-18)
        return example

    def initialize_norm(self, dataset_name=None, dataset=None, max_workers=0):
        filename = f"{dataset_name}_moments.json" if dataset_name \
            else "moments.json"
        filepath = None if self.storage_dir is None \
            else (Path(self.storage_dir) / filename).expanduser().absolute()
        if dataset is not None:
            dataset = dataset
            if max_workers > 0:
                dataset = dataset.prefetch(
                    max_workers, 2 * max_workers, catch_filter_exception=True
                )
        moments = read_moments(
            dataset, "features", center_axis=(0, 1), scale_axis=(0, 1, 2),
            filepath=filepath, verbose=True
        )
        if dataset_name is None:
            assert self.moments is None
            self.moments = moments
        else:
            if self.moments is None:
                self.moments = {}
            assert dataset_name not in self.moments
            self.moments[dataset_name] = moments


def read_moments(
        dataset=None, key=None, center_axis=None, scale_axis=None,
        filepath=None, verbose=False
):
    """
    Loads or computes the global mean (center) and scale over a dataset.

    Args:
        dataset: lazy dataset providing example dicts
        key: example dict key of the features to compute the moments from
        center_axis: axis of the feature array over which the mean (center) is computed
        scale_axis: axis of the feature array over which the scale is computed
        filepath: file to load/store the moments from/at
        verbose:

    Returns:

    """
    if filepath and Path(filepath).exists():
        with filepath.open() as fid:
            mean, scale = json.load(fid)
        if verbose:
            print(f'Restored moments from {filepath}')
    else:
        assert dataset is not None
        mean = 0
        mean_count = 0
        energy = 0
        energy_count = 0
        for example in tqdm(dataset, disable=not verbose):
            x = example[key]
            if center_axis is not None:
                if not mean_count:
                    mean = np.sum(x, axis=center_axis, keepdims=True)
                else:
                    mean += np.sum(x, axis=center_axis, keepdims=True)
                mean_count += np.prod(
                    np.array(x.shape)[np.array(center_axis)]
                )
            if scale_axis is not None:
                if not energy_count:
                    energy = np.sum(x**2, axis=scale_axis, keepdims=True)
                else:
                    energy += np.sum(x**2, axis=scale_axis, keepdims=True)
                energy_count += np.prod(
                    np.array(x.shape)[np.array(scale_axis)]
                )
        if center_axis is not None:
            mean /= mean_count
        if scale_axis is not None:
            energy /= energy_count
            scale = np.sqrt(np.mean(
                energy - mean ** 2, axis=scale_axis, keepdims=True
            ))
        else:
            scale = np.array(1.)

        if filepath:
            with filepath.open('w') as fid:
                json.dump(
                    (mean.tolist(), scale.tolist()), fid,
                    sort_keys=True, indent=4
                )
            if verbose:
                print(f'Saved moments to {filepath}')
    return np.array(mean), np.array(scale)


class Augmenter:
    def __init__(
            self,
            time_warping_factor_std=None,
            time_warping_cutoff_std=0.1,
            feature_warping_factor_std=None,
            feature_warping_cutoff_std=0.5,
            n_time_masks=0,
            max_masked_time_rate=0.2,
            max_masked_time_steps=70,
            n_feat_masks=0,
            max_masked_feature_rate=0.2,
            max_masked_features=16
    ):
        """
        Performs the following augmentations:
            frequency and time warping of a mel spectrogram,
            frequency and time masking of a mel spectrogram

        Args:
            time_warping_factor_std: standard deviation of the time warping factor
            time_warping_cutoff_std:
            feature_warping_factor_std: standard deviation of the feature warping factor (0.07 in our paper)
            feature_warping_cutoff_std: standard deviation of the cutoff frequency (0.5 in our paper)
            n_time_masks: number of time masks in a single spec
            max_masked_time_rate: maximum rate of masked frames
            max_masked_time_steps: maximum number of masked frames
            n_feat_masks: number of frequency masks in a single spec
            max_masked_feature_rate: maximum rate of masked mel bands
            max_masked_features: maximum number of masked mel bands
        """
        self.time_warping_factor_std = time_warping_factor_std
        self.time_warping_cutoff_std = time_warping_cutoff_std
        self.feature_warping_factor_std = feature_warping_factor_std
        self.feature_warping_cutoff_std = feature_warping_cutoff_std
        self.n_time_masks = n_time_masks
        self.max_masked_time_rate = max_masked_time_rate
        self.max_masked_time_steps = max_masked_time_steps
        self.n_feature_masks = n_feat_masks
        self.max_masked_feature_rate = max_masked_feature_rate
        self.max_masked_features = max_masked_features

    def __call__(self, example):
        example = self.warp(example)
        example = self.mask(example)
        return example

    def warp(self, example):
        x = example['features']
        if self.time_warping_factor_std is not None and self.time_warping_factor_std > 0.:
            seq_len = x.shape[-2]
            warping_factor = (
                1. + np.random.exponential(self.time_warping_factor_std)
            ) ** (2*np.random.choice(2) - 1.)
            cutoff = np.random.normal(0.5, self.time_warping_cutoff_std) * seq_len

            src_ctrl_point = min(cutoff, seq_len - cutoff) * 2 / (warping_factor + 1)
            dst_ctrl_point = warping_factor * src_ctrl_point
            if cutoff > seq_len // 2:
                src_ctrl_point = seq_len - src_ctrl_point
                dst_ctrl_point = seq_len - dst_ctrl_point

            # print(warping_factor, seq_len, src_ctrl_point, dst_ctrl_point)
            x = warp1d(
                x.squeeze(0),
                [0, src_ctrl_point, seq_len],
                [0, dst_ctrl_point, seq_len]
            )[None]
        if self.feature_warping_factor_std is not None and self.feature_warping_factor_std > 0.:
            n_features = x.shape[-1]
            warping_factor = (
                1. + np.random.exponential(self.feature_warping_factor_std)
            ) ** (2*np.random.choice(2) - 1.)
            cutoff = np.random.exponential(self.feature_warping_cutoff_std) * n_features

            src_ctrl_point = cutoff * 2 / (warping_factor + 1)
            dst_ctrl_point = warping_factor * src_ctrl_point
            # print(src_ctrl_point, dst_ctrl_point)
            x = warp1d(
                x.squeeze(0).T,
                [0, src_ctrl_point, n_features],
                [0, dst_ctrl_point, n_features]
            ).T[None]
        example['features'] = x
        return example

    def mask(self, example):
        x = example['features']
        for n in range(self.n_time_masks):
            seq_len = x.shape[-2]
            max_masked_steps = min(
                int(self.max_masked_time_rate * seq_len),
                self.max_masked_time_steps
            )
            masked_steps = int(np.random.rand() * (max_masked_steps + 1))
            start_idx = int(np.random.rand() * seq_len)
            masked_steps = min(
                masked_steps, seq_len - start_idx
            )
            stop_idx = start_idx + masked_steps
            x[..., start_idx:stop_idx, :] = 0.
        for n in range(self.n_feature_masks):
            n_features = x.shape[-1]
            max_masked_feats = min(
                int(self.max_masked_feature_rate * n_features),
                self.max_masked_features
            )
            masked_feats = int(np.random.rand() * (max_masked_feats + 1))
            start_idx = int(
                np.random.rand() * n_features
            )
            stop_idx = start_idx + masked_feats
            x[..., start_idx:stop_idx] = 0.
        example['features'] = x
        return example


def warp1d(
        spec,
        source_control_points,
        dest_control_points,
):
    T = spec.shape[0]
    flow = make_interp_spline(
        np.array(dest_control_points),
        np.array(source_control_points),
        k=1
    )(np.arange(T))

    def inverse_map(coordinates):
        y_coord = coordinates[:, 0]
        x_coord = flow[coordinates[:, 1].astype(np.int64)]
        transformed_coord = np.array([y_coord, x_coord]).T
        return transformed_coord

    return warp(spec, inverse_map)


class EventTimeSeriesBucket(DynamicTimeSeriesBucket):
    def __init__(self, init_example, **kwargs):
        """
        Extension of the DynamicTimeSeriesBucket such that each example in a
        batch has a unique event signature

        Args:
            init_example: first example in the bucket
            **kwargs: kwargs of DynamicTimeSeriesBucket
        """
        super().__init__(init_example, **kwargs)
        self.event_mat = init_example['events'][None]

    def assess(self, example):
        events = example['events']
        return (
            super().assess(example)
            and np.sum(self.event_mat * (1. - events), axis=-1).all()  # events do not mask events of other example in bucket
            and np.sum((1. - self.event_mat) * events, axis=-1).all()  # events are not masked by events of other example in bucket
        )

    def _append(self, example):
        super()._append(example)
        events = example['events']
        self.event_mat = np.concatenate([self.event_mat, events[None]])


class DatasetBalancedTimeSeriesBucket(DynamicTimeSeriesBucket):
    def __init__(self, init_example, min_examples, **kwargs):
        """
        Extension of the DynamicTimeSeriesBucket such that examples are
        balanced with respect to the dataset they originate from

        Args:
            init_example: first example in the bucket
            dataset_sizes:
            **kwargs: kwargs of DynamicTimeSeriesBucket
        """
        super().__init__(init_example, **kwargs)
        self.min_examples = min_examples
        self.counts = {key: 0 for key in min_examples}

    def assess(self, example):
        ds = example['dataset']
        return (
            super().assess(example)
            and (
                self.counts[ds] < self.min_examples[ds]
                or all([
                    self.counts[key] >= self.min_examples[key]
                    for key in self.counts
                ])
            )
        )

    def _append(self, example):
        super()._append(example)
        self.counts[example['dataset']] += 1
