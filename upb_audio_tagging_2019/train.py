"""
Example call:

export STORAGE_ROOT=<your desired storage root>
python -m padertorch.contrib.examples.wavenet.train print_config
python -m padertorch.contrib.examples.wavenet.train
"""
import os
from collections import defaultdict

import lazy_dataset
import numpy as np
import tensorboardX
import torch
from lazy_dataset.database import JsonDatabase
from sacred import Experiment as Exp
from sacred.commands import print_config
from sacred.observers import FileStorageObserver
from torch.nn import GRU
from torch.optim import Adam
from torchcontrib.optim import SWA
from upb_audio_tagging_2019.data import (
    split_dataset, MixUpDataset, Extractor, Augmenter,
    DynamicTimeSeriesBucket, EventTimeSeriesBucket,
    Collate, batch_to_device
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
    fold = None
    curated_reps = 7
    mixup_probs = [1/3, 2/3]
    extractor = {
        'input_sample_rate': 44100,
        'target_sample_rate': 44100,
        'frame_step': 882,
        'frame_length': 1764,
        'fft_length': 2048,
        'n_mels': 128,
        'fmin': 50,
        'fmax': 16000,
        'storage_dir': storage_dir
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

    # Model configuration
    model = {
        'cnn_2d': {
            # 'factory': MultiScaleCNN2d,
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
            'input_size': 256, 'hidden_size': 256, 'num_layers': 2,
            'batch_first': True, 'bidirectional': False, 'dropout': 0.
        },
        'fcn': {
            'input_size': 256, 'hidden_size': 256, 'output_size': 80,
            'activation': 'relu', 'dropout': 0.
        },
        'fcn_noisy': {
            'input_size': 256, 'hidden_size': 256, 'output_size': 80,
            'activation': 'relu', 'dropout': 0.
        },
        'decision_boundary': .3
    }

    # Training configuration
    device = 0 if torch.cuda.is_available() else 'cpu'
    lr = 3e-4
    gradient_clipping = 15.
    weight_decay = 3e-5
    swa_start = 750 if debug else 150000
    swa_freq = 50 if debug else 1000
    swa_lr = lr
    summary_interval = 10 if debug else 100
    validation_interval = 500 if debug else 5000
    max_steps = 1000 if debug else 200000


@ex.capture
def get_datasets(
        use_noisy, split, fold, curated_reps, mixup_probs,
        extractor, augmenter, num_workers, batch_size, prefetch_buffer,
        max_padding_rate, bucket_expiration, event_bucketing, debug
):
    # prepare database
    database_json = jsons_dir / f'fsd_kaggle_2019_split{split}.json'
    db = JsonDatabase(database_json)

    def add_noisy_flag(example):
        example['is_noisy'] = example['dataset'] != 'train_curated'
        return example
    extractor = Extractor(**extractor)
    augmenter = Augmenter(extractor=extractor, **augmenter)

    curated_train_data = db.get_dataset('train_curated').map(add_noisy_flag)
    extractor.initialize_labels(curated_train_data)
    if debug:
        curated_train_data = curated_train_data.shuffle()[:len(curated_train_data)//10]
    extractor.initialize_norm(
        dataset_name='train_curated',
        dataset=curated_train_data,
        max_workers=num_workers
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
            noisy_train_data = noisy_train_data.shuffle()[:len(noisy_train_data)//10]
        extractor.initialize_norm(
            dataset_name='train_noisy',
            dataset=noisy_train_data,
            max_workers=num_workers
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

    bucket_cls = EventTimeSeriesBucket if event_bucketing \
        else DynamicTimeSeriesBucket

    def prepare_iterable(dataset, drop_incomplete=False):
        return dataset.prefetch(
            num_workers=num_workers, buffer_size=prefetch_buffer,
            catch_filter_exception=True
        ).batch_dynamic_bucket(
            bucket_cls=bucket_cls, batch_size=batch_size, len_key='seq_len',
            max_padding_rate=max_padding_rate, expiration=bucket_expiration,
            drop_incomplete=drop_incomplete, sort_key='seq_len',
            reverse_sort=True
        ).map(Collate())

    training_set = prepare_iterable(
        training_set.map(augmenter).shuffle(reshuffle=True),
        drop_incomplete=True
    )
    batch_norm_tuning_set = prepare_iterable(
        batch_norm_tuning_set.map(extractor), drop_incomplete=True
    )
    if validation_set is not None:
        validation_set = prepare_iterable(validation_set.map(extractor))

    return training_set, validation_set, batch_norm_tuning_set


@ex.automain
def train(
        _run, model, device, lr, gradient_clipping, weight_decay,
        swa_start, swa_freq, swa_lr,
        summary_interval, validation_interval, max_steps
):
    print_config(_run)
    os.makedirs(storage_dir / 'checkpoints', exist_ok=True)
    train_iter, validate_iter, batch_norm_tuning_iter = get_datasets()
    model = CRNN(
        cnn_2d=CNN2d(**model['cnn_2d']),
        cnn_1d=CNN1d(**model['cnn_1d']),
        enc=GRU(**model['enc']),
        fcn=fully_connected_stack(**model['fcn']),
        fcn_noisy=None if model['fcn_noisy'] is None
        else fully_connected_stack(**model['fcn_noisy']),
    )
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    model = model.to(device)
    model.train()
    optimizer = Adam(
        tuple(model.parameters()), lr=lr, weight_decay=weight_decay
    )
    if swa_start is not None:
        optimizer = SWA(
            optimizer, swa_start=swa_start, swa_freq=swa_freq, swa_lr=swa_lr
        )

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False

    # Summary
    summary_writer = tensorboardX.SummaryWriter(str(storage_dir))

    def get_empty_summary():
        return dict(
            scalars=defaultdict(list),
            histograms=defaultdict(list),
            images=dict(),
        )

    def update_summary(review, summary):
        review['scalars']['loss'] = review['loss'].detach()
        for key, value in review['scalars'].items():
            if torch.is_tensor(value):
                value = value.cpu().data.numpy()
            summary['scalars'][key].extend(
                np.array(value).flatten().tolist()
            )
        for key, value in review['histograms'].items():
            if torch.is_tensor(value):
                value = value.cpu().data.numpy()
            summary['histograms'][key].extend(
                np.array(value).flatten().tolist()
            )
        summary['images'] = review['images']

    def dump_summary(summary, prefix, iteration):
        summary = model.modify_summary(summary)

        # write summary
        for key, value in summary['scalars'].items():
            summary_writer.add_scalar(
                f'{prefix}/{key}', np.mean(value), iteration
            )
        for key, values in summary['histograms'].items():
            summary_writer.add_histogram(
                f'{prefix}/{key}', np.array(values), iteration
            )
        for key, image in summary['images'].items():
            summary_writer.add_image(
                f'{prefix}/{key}', image, iteration
            )
        return defaultdict(list)

    # Training loop
    train_summary = get_empty_summary()
    i = 0
    while i < max_steps:
        for batch in train_iter:
            optimizer.zero_grad()
            # forward
            batch = batch_to_device(batch, device=device)
            model_out = model(batch)

            # backward
            review = model.review(batch, model_out)
            review['loss'].backward()
            review['histograms']['grad_norm'] = torch.nn.utils.clip_grad_norm_(
                tuple(model.parameters()), gradient_clipping
            )
            optimizer.step()

            # update summary
            update_summary(review, train_summary)

            i += 1
            if i % summary_interval == 0:
                dump_summary(train_summary, 'training', i)
                train_summary = get_empty_summary()
            if i % validation_interval == 0 and validate_iter is not None:
                print('Starting Validation')
                model.eval()
                validate_summary = get_empty_summary()
                with torch.no_grad():
                    for batch in validate_iter:
                        batch = batch_to_device(batch, device=device)
                        model_out = model(batch)
                        review = model.review(batch, model_out)
                        update_summary(review, validate_summary)
                dump_summary(validate_summary, 'validation', i)
                print('Finished Validation')
                model.train()
            if i >= max_steps:
                break

    # finalize
    if swa_start is not None:
        optimizer.swap_swa_sgd()
    batch_norm_update(
        model, batch_norm_tuning_iter, feature_key='features', device=device
    )
    torch.save(
        model.state_dict(),
        storage_dir / 'checkpoints' / 'ckpt_final.pth'
    )
