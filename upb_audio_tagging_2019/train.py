import lazy_dataset
import numpy as np
import torch
from lazy_dataset.database import JsonDatabase
from padertorch import Trainer
from padertorch.contrib.je.data.transforms import (
    Collate, MultiHotLabelEncoder
)
from padertorch.train.optimizer import Adam
from sacred import Experiment as Exp
from sacred.commands import print_config
from sacred.observers import FileStorageObserver
from torch.nn import GRU
from upb_audio_tagging_2019.data import (
    split_dataset, MixUpDataset, AudioReader, STFT, MelTransform, Normalizer,
    Augmenter, DynamicTimeSeriesBucket, EventTimeSeriesBucket
)
from upb_audio_tagging_2019.model import CRNN, batch_norm_update
from upb_audio_tagging_2019.modules import CNN2d, CNN1d, fully_connected_stack
from upb_audio_tagging_2019.paths import exp_dir, jsons_dir
from upb_audio_tagging_2019.utils import timestamp

ex = Exp('upb_audio_tagging_2019')
storage_dir = exp_dir / timestamp()
observer = FileStorageObserver.create(str(storage_dir))
ex.observers.append(observer)


@ex.config
def config():
    debug = False

    # Data configuration
    use_noisy = True
    split = 0
    relabeled = False
    fold = None
    curated_reps = 7
    mixup_probs = [1/3, 2/3]
    audio_reader = {
        'input_sample_rate': 44100,
        'target_sample_rate': 44100,
    }
    stft = {
        'frame_step': 882,
        'frame_length': 1764,
        'fft_length': 2048,
    }
    mel_transform = {
        'sample_rate': audio_reader['target_sample_rate'],
        'fft_length': stft['fft_length'],
        'n_mels': 128,
        'fmin': 50,
        'fmax': 16000,
    }
    augmenter = {
        'time_warping_factor_std': None,
        'time_warping_cutoff_std': 0.1,
        'feature_warping_factor_std': 0.07,
        'feature_warping_cutoff_std': 0.5,
        'n_time_masks': 1,
        'n_feat_masks': 1,
    }
    num_workers = 8
    batch_size = 16
    prefetch_buffer = 20 * batch_size
    max_padding_rate = 0.2
    bucket_expiration = 2000 * batch_size
    event_bucketing = True

    # Trainer/Model configuration
    trainer = {
        'model': {
            'factory': CRNN,
            'cnn_2d': {
                'factory': CNN2d,
                'in_channels': 1,
                'hidden_channels': [16, 16, 32, 32, 64, 64, 128, 128, 256],
                'pool_size': [1, 2, 1, 2, 1, 2, 1, (2, 1), (2, 1)],
                'num_layers': 9,
                'out_channels': None,
                'kernel_size': 3,
                'norm': 'batch',
                'activation': 'relu',
                'gated': False,
                'dropout': .0,
            },
            'cnn_1d': {
                'factory': CNN1d,
                'in_channels': 1024,
                'hidden_channels': 256,
                'num_layers': 3,
                'out_channels': None,
                'kernel_size': 3,
                'norm': 'batch',
                'activation': 'relu',
                'dropout': .0
            },
            'enc': {
                'factory': GRU,
                'input_size': 256,
                'hidden_size': 256,
                'num_layers': 2,
                'batch_first': True,
                'bidirectional': False,
                'dropout': 0.,
            },
            'fcn': {
                'factory': fully_connected_stack,
                'input_size': 256,
                'hidden_size': 256,
                'output_size': 80,
                'activation': 'relu',
                'dropout': 0.,
            },
            'fcn_noisy': {
                'factory': fully_connected_stack,
                'input_size': 256,
                'hidden_size': 256,
                'output_size': 80,
                'activation': 'relu',
                'dropout': 0.,
            },
            'decision_boundary': .3
        },
        'optimizer': {
            'factory': Adam,
            'lr': 3e-4,
            'gradient_clipping': 15.,
            'weight_decay': 3e-5,
            'swa_start': 750 if debug else 150000,
            'swa_freq': 50 if debug else 1000,
            'swa_lr': 3e-4,
        },
        'storage_dir': storage_dir,
        'summary_trigger': (10 if debug else 100, 'iteration'),
        'checkpoint_trigger': (500 if debug else 5000, 'iteration'),
        'stop_trigger': (1000 if debug else 200000, 'iteration'),
    }
    Trainer.get_config(trainer)

    device = 0 if torch.cuda.is_available() else 'cpu'


@ex.capture
def get_datasets(
        use_noisy, split, relabeled, fold, curated_reps, mixup_probs,
        audio_reader, stft, mel_transform, augmenter, num_workers, batch_size,
        prefetch_buffer, max_padding_rate, bucket_expiration, event_bucketing,
        debug
):
    # prepare database
    database_json = jsons_dir / \
        f'fsd_kaggle_2019_split{split}{"_relabeled" if relabeled else ""}.json'
    db = JsonDatabase(database_json)

    def add_noisy_flag(example):
        example['is_noisy'] = example['dataset'] != 'train_curated'
        return example

    audio_reader = AudioReader(**audio_reader)
    stft = STFT(**stft)
    mel_transform = MelTransform(**mel_transform)
    normalizer = Normalizer(storage_dir=str(storage_dir))
    augmenter = Augmenter(**augmenter)

    curated_train_data = db.get_dataset('train_curated').map(add_noisy_flag)

    event_encoder = MultiHotLabelEncoder(
        label_key='events', storage_dir=storage_dir
    )
    event_encoder.initialize_labels(
        dataset=curated_train_data, verbose=True
    )

    if debug:
        curated_train_data = curated_train_data.shuffle()[:500]

    normalizer.initialize_norm(
        dataset_name='train_curated',
        dataset=curated_train_data.map(audio_reader).map(stft).map(mel_transform),
        max_workers=num_workers,
    )

    if fold is not None:
        curated_train_data, validation_set = split_dataset(
            curated_train_data, fold=fold, seed=0
        )
    else:
        validation_set = None

    if use_noisy:
        noisy_train_data = db.get_dataset('train_noisy').map(add_noisy_flag)
        if debug:
            noisy_train_data = noisy_train_data.shuffle()[:500]

        normalizer.initialize_norm(
            dataset_name='train_noisy',
            dataset=noisy_train_data.map(audio_reader).map(stft).map(mel_transform),
            max_workers=num_workers,
        )
        training_set = lazy_dataset.concatenate(curated_train_data, noisy_train_data)
    else:
        training_set = curated_train_data
    batch_norm_tuning_set = training_set

    if mixup_probs is not None:
        training_set = MixUpDataset(
            training_set, training_set, mixin_probs=mixup_probs
        )
    if curated_reps > 0:
        print('curated reps:', curated_reps)
        curated_train_data = lazy_dataset.from_dict({
            f'{example["example_id"]}_{i}': example
            for i in range(curated_reps)
            for example in curated_train_data
        })
        if mixup_probs is not None:
            curated_train_data = MixUpDataset(
                curated_train_data, curated_train_data, mixin_probs=mixup_probs
            )
        training_set = lazy_dataset.concatenate(
            training_set, curated_train_data
        )

    print('Length of training set', len(training_set))
    print('Length of validation set', len(validation_set))

    def finalize(example):
        x = example['features']
        example_ = {
            'example_id': example['example_id'],
            'dataset': example['dataset'],
            'is_noisy': np.array(example['is_noisy']).astype(np.float32),
            'features': x.astype(np.float32),
            'seq_len': x.shape[1],
        }
        if 'events' in example:
            example_['events'] = example['events']
        return example_

    bucket_cls = EventTimeSeriesBucket if event_bucketing \
        else DynamicTimeSeriesBucket

    def prepare_iterable(dataset, drop_incomplete=False):
        return dataset.map(event_encoder).map(finalize).prefetch(
            num_workers=num_workers, buffer_size=prefetch_buffer,
            catch_filter_exception=True
        ).batch_dynamic_bucket(
            bucket_cls=bucket_cls, batch_size=batch_size, len_key='seq_len',
            max_padding_rate=max_padding_rate, expiration=bucket_expiration,
            drop_incomplete=drop_incomplete, sort_key='seq_len',
            reverse_sort=True
        ).map(Collate())

    training_set = prepare_iterable(
        training_set.map(audio_reader).map(stft).map(mel_transform).map(normalizer).map(augmenter).shuffle(reshuffle=True),
        drop_incomplete=True
    )
    batch_norm_tuning_set = prepare_iterable(
        batch_norm_tuning_set.map(audio_reader).map(stft).map(mel_transform).map(normalizer),
        drop_incomplete=True
    )
    if validation_set is not None:
        validation_set = prepare_iterable(
            validation_set.map(audio_reader).map(stft).map(mel_transform).map(normalizer)
        )

    return training_set, validation_set, batch_norm_tuning_set


@ex.automain
def train(
        _run, trainer, device,
):
    print_config(_run)
    trainer = Trainer.from_config(trainer)
    train_iter, validate_iter, batch_norm_tuning_iter = get_datasets()
    if validate_iter is not None:
        trainer.register_validation_hook(validate_iter)
    trainer.train(train_iter, device=device)

    # finalize
    if trainer.optimizer.swa_start is not None:
        trainer.optimizer.swap_swa_sgd()
    batch_norm_update(
        trainer.model, batch_norm_tuning_iter,
        feature_key='features', device=device
    )
    torch.save(
        trainer.model.state_dict(),
        storage_dir / 'checkpoints' / 'ckpt_final.pth'
    )
