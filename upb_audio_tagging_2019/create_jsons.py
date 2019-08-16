import concurrent.futures
import csv
import json
import os
from functools import partial
from pathlib import Path

import numpy as np
import soundfile
from natsort import natsorted
from upb_audio_tagging_2019.paths import jsons_dir


def read_audio_length_in_sec(filepath):
    with soundfile.SoundFile(str(filepath)) as f:
        return len(f) / f.samplerate


def construct_json(database_path: Path) -> dict:
    datasets = {}
    events = set()

    def create_example(example_id_tags, audio_dir):
        """
        Creates example dict for one example with example_id.
        """
        example_id, tags = example_id_tags
        audio_path = Path(audio_dir) / (example_id + '.wav')
        length = read_audio_length_in_sec(audio_path)
        assert length > 0
        example = {
            'audio_path': str(audio_path),
            'audio_length': length
        }
        if tags is not None:
            example['events'] = tags
        return (
            example_id,
            example
        )

    def add_dataset(dataset_name, tags_dict):

        # get a set of all available example_ids from the file system.
        audio_dir = database_path / dataset_name
        available = set(
            [wav_file.name.split('.wav')[0]
             for wav_file in audio_dir.glob('*.wav')]
        )
        datasets[dataset_name] = dict()

        original_length = len(tags_dict)
        # exclude all examples whose files are not available
        tags_items = filter(
            lambda x: x[0] in available,  tags_dict.items()
        )
        # sort
        tags_items = list(
            natsorted(tags_items, key=lambda x: x[0])
        )

        _create_example = partial(create_example, audio_dir=audio_dir)
        # parallelize with progress bar...
        with concurrent.futures.ThreadPoolExecutor() as ex:
            for example_id, example in ex.map(
                    _create_example, tags_items
            ):
                datasets[dataset_name][example_id] = example
                if 'events' in example:
                    events.update(example['events'])

        print(f'{original_length - len(datasets[dataset_name])}'
              f' from {original_length} files missing in {dataset_name}')

    # iterate over all segment files (datasets)
    for file in database_path.glob('*.csv'):
        if file.name == "sample_submission.csv":
            name = 'test'
        else:
            name = file.name[:-4]
        tags_dict = _read_csv_file(file)
        add_dataset(name, tags_dict)

    print('Number of event labels:', len(events))
    return {'datasets': datasets}


def _read_csv_file(csv_file):
    with csv_file.open() as fid:
        tags = {
            row[0][:-4]: row[1].split(',') if len(row) == 2 else None
            for row in csv.reader(fid)
            if len(row) > 1 and not row[0].startswith('fname')
        }
    return tags


def create_json():
    database_path = Path(os.environ['FSDKaggle2019DIR'])
    database = construct_json(database_path)
    os.makedirs(str(jsons_dir), exist_ok=True)
    json_path = jsons_dir / 'fsd_kaggle_2019.json'
    with json_path.open('w') as fid:
        json.dump(database, fid, indent=4, sort_keys=True)


def create_split(seed=0):
    json_path = jsons_dir / 'fsd_kaggle_2019.json'
    with json_path.open() as fid:
        datasets = json.load(fid)['datasets']

    train_noisy = {}
    npr = np.random.RandomState(seed)
    for key, example in datasets['train_noisy'].items():
        split_length = example['audio_length'] * (0.1 + 0.8*npr.rand())
        split_sample = int(split_length * 44100)
        train_noisy.update({
            f'{key}_1': {
                'audio_length': split_length,
                'audio_path': example['audio_path'],
                'events': example['events'],
                # 'uncertain_events': example['events'],
                'audio_stop_samples': split_sample
            },
            f'{key}_2': {
                'audio_length': example['audio_length'] - split_length,
                'audio_path': example['audio_path'],
                'events': example['events'],
                # 'uncertain_events': example['events'],
                'audio_start_samples': split_sample
            },
        })
    datasets['train_noisy'] = train_noisy
    json_path = jsons_dir / f'fsd_kaggle_2019_split{seed}.json'
    with json_path.open('w') as fid:
        json.dump({'datasets': datasets}, fid, indent=4, sort_keys=True)


if __name__ == "__main__":
    create_json()
    for seed in range(3):
        create_split(seed)
