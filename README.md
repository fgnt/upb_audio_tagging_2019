# upb_audio_tagging_2019: Convolutional Recurrent Neural Network and Data Augmentation for Audio Tagging with Noisy Labels and Minimal Supervision [\[pdf\]](http://dcase.community/documents/workshop2019/proceedings/DCASE2019Workshop_Ebbers_54.pdf)

This repository provides the source code for the 5-th place solution presented by Paderborn University for the Freesound Audio Tagging 2019 Kaggle Competition.
Our best submitted system achieved 75.5 % label-weighted label ranking average precision (lwlrap).
Later improvements due to sophisticated ensembling including Multi-Task-Learning led to 76.5 % lwlrap outperforming the winner of the competition.

Competition website: https://www.kaggle.com/c/freesound-audio-tagging-2019

If you are using this code please cite the following paper:

```
@inproceedings{Ebbers2019,
    author = "Ebbers, Janek and Häb-Umbach, Reinhold",
    title = "Convolutional Recurrent Neural Network and Data Augmentation for Audio Tagging with Noisy Labels and Minimal Supervision",
    booktitle = "Proceedings of the Detection and Classification of Acoustic Scenes and Events 2019 Workshop (DCASE2019)",
    address = "New York University, NY, USA",
    month = "October",
    year = "2019",
    pages = "64--68"
}

```

## Installation
Install requirements
```bash
$ pip install --user git+https://github.com/fgnt/lazy_dataset.git@ec06c1e8ff4ccb09420d2d641db8f6d9b1099a4f
$ pip install --user git+https://github.com/fgnt/paderbox.git@7b3b4e9d00e07664596108f987292b8c78d846b1
$ pip install --user git+https://github.com/fgnt/padertorch.git@88233a0c33ddcc33a6842a5f8dc6c24df84d9f09
```

Clone the repo
```bash
$ git clone https://github.com/fgnt/upb_audio_tagging_2019.git
$ cd upb_audio_tagging_2019
```

Install this package
```bash
$ pip install --user -e .
```

Create database description jsons
```bash
$ export FSDKaggle2019DIR=/path/to/fsd_kaggle_2019
$ python -m upb_audio_tagging_2019.create_jsons
```

## Training
We use sacred (https://sacred.readthedocs.io/en/latest/quickstart.html) for
configuration of a training. To train a model (without Multi-Task-Learning) run
```bash
$ python -m upb_audio_tagging_2019.train with model.fcn_noisy=None
```

It is assumed that the folder `exp` in this git is the simulation folder.
If you want to change the simulation dir, add a symlink to the folder where you
want to store the simulation results: `ln -s /path/to/sim/dir exp`.
For each training a new timestamped subdirectory is created.
Monitoring of the training is done using tensorboardX.
To view the training progress call `tensorboard --logdir exp/subdirname`.
After training finished there will be a checkpoint `ckpt_final.pth` in the
subdirectory.


## Reproduce paper results
### Training
To train the ensembles described in our paper you need to run the following
configurations:
#### with provided labels:
```bash
$ python -m upb_audio_tagging_2019.train with model.fcn_noisy=None split=0 curated_reps=7
$ python -m upb_audio_tagging_2019.train with model.fcn_noisy=None split=1 curated_reps=7
$ python -m upb_audio_tagging_2019.train with model.fcn_noisy=None split=2 curated_reps=7
$ python -m upb_audio_tagging_2019.train with model.fcn_noisy=None split=0 curated_reps=5
$ python -m upb_audio_tagging_2019.train with model.fcn_noisy=None split=1 curated_reps=5
$ python -m upb_audio_tagging_2019.train with model.fcn_noisy=None split=2 curated_reps=5
$ python -m upb_audio_tagging_2019.train with model.fcn_noisy=None split=0 curated_reps=3
$ python -m upb_audio_tagging_2019.train with model.fcn_noisy=None split=1 curated_reps=3
$ python -m upb_audio_tagging_2019.train with model.fcn_noisy=None split=2 curated_reps=3
```
#### with Multi-Task-Learning:
```bash
$ python -m upb_audio_tagging_2019.train with split=0 curated_reps=7
$ python -m upb_audio_tagging_2019.train with split=1 curated_reps=7
$ python -m upb_audio_tagging_2019.train with split=2 curated_reps=7
$ python -m upb_audio_tagging_2019.train with split=0 curated_reps=5
$ python -m upb_audio_tagging_2019.train with split=1 curated_reps=5
$ python -m upb_audio_tagging_2019.train with split=2 curated_reps=5
$ python -m upb_audio_tagging_2019.train with split=0 curated_reps=3
$ python -m upb_audio_tagging_2019.train with split=1 curated_reps=3
$ python -m upb_audio_tagging_2019.train with split=2 curated_reps=3
```

#### with Relabeling:
```bash
$ python -m upb_audio_tagging_2019.train with model.fcn_noisy=None split=0 relabeled=True curated_reps=6
$ python -m upb_audio_tagging_2019.train with model.fcn_noisy=None split=1 relabeled=True curated_reps=6
$ python -m upb_audio_tagging_2019.train with model.fcn_noisy=None split=2 relabeled=True curated_reps=6
$ python -m upb_audio_tagging_2019.train with model.fcn_noisy=None split=0 relabeled=True curated_reps=4
$ python -m upb_audio_tagging_2019.train with model.fcn_noisy=None split=1 relabeled=True curated_reps=4
$ python -m upb_audio_tagging_2019.train with model.fcn_noisy=None split=2 relabeled=True curated_reps=4
$ python -m upb_audio_tagging_2019.train with model.fcn_noisy=None split=0 relabeled=True curated_reps=2
$ python -m upb_audio_tagging_2019.train with model.fcn_noisy=None split=1 relabeled=True curated_reps=2
$ python -m upb_audio_tagging_2019.train with model.fcn_noisy=None split=2 relabeled=True curated_reps=2
```
##### Reproduce relabeling:
The relabeling procedure requires to train 15 models for relabeling
```bash
$ python -m upb_audio_tagging_2019.train with split=0 fold=0 curated_reps=9
$ python -m upb_audio_tagging_2019.train with split=0 fold=1 curated_reps=9
$ python -m upb_audio_tagging_2019.train with split=0 fold=2 curated_reps=9
$ python -m upb_audio_tagging_2019.train with split=0 fold=3 curated_reps=9
$ python -m upb_audio_tagging_2019.train with split=0 fold=4 curated_reps=9
$ python -m upb_audio_tagging_2019.train with split=1 fold=0 curated_reps=9
$ python -m upb_audio_tagging_2019.train with split=1 fold=1 curated_reps=9
$ python -m upb_audio_tagging_2019.train with split=1 fold=2 curated_reps=9
$ python -m upb_audio_tagging_2019.train with split=1 fold=3 curated_reps=9
$ python -m upb_audio_tagging_2019.train with split=1 fold=4 curated_reps=9
$ python -m upb_audio_tagging_2019.train with split=2 fold=0 curated_reps=9
$ python -m upb_audio_tagging_2019.train with split=2 fold=1 curated_reps=9
$ python -m upb_audio_tagging_2019.train with split=2 fold=2 curated_reps=9
$ python -m upb_audio_tagging_2019.train with split=2 fold=3 curated_reps=9
$ python -m upb_audio_tagging_2019.train with split=2 fold=4 curated_reps=9
```
Perform relabeling: Code not yet available


### Inference:
Kaggle kernel not yet public
