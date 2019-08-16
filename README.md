# upb_audio_tagging_2019: Convolutional Recurrent Neural Network and Data Augmentation for Audio Tagging with Noisy Labels and Minimal Supervision [\[pdf\]](http://dcase.community/documents/challenge2019/technical_reports/DCASE2019_Ebbers_92_t2.pdf)

This repository provides the source code for the 5-th place solution presented by Paderborn University for the Freesound Audio Tagging 2019 Kaggle Competition.
Our best submitted system achieved 75.5 % label-weighted label ranking average precision (lwlrap).
Later improvements due to sophisticated ensembling including Multi-Task-Learning led to 76.5 % lwlrap outperforming the winner of the competition.

Competition website: https://www.kaggle.com/c/freesound-audio-tagging-2019

If you are using this code please cite the following paper:

```
@Article{ebbers2019dcase,
  author    = {Ebbers, Janek and Haeb-Umbach, Reinhold},
  title     = {{Convolutional Recurrent Neural Network and Data Augmentation for Audio Tagging with Noisy Labels and Minimal Supervision}},
  year      = {2019}
}
```

## Installation
Install requirements
```bash
$ pip install --user git+https://github.com/fgnt/lazy_dataset.git
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
$ python -m upb_audio_tagging_2019.train with model.fcn_noisy=None curated_reps=7 split=0
$ python -m upb_audio_tagging_2019.train with model.fcn_noisy=None curated_reps=7 split=1
$ python -m upb_audio_tagging_2019.train with model.fcn_noisy=None curated_reps=7 split=2
$ python -m upb_audio_tagging_2019.train with model.fcn_noisy=None curated_reps=5 split=0
$ python -m upb_audio_tagging_2019.train with model.fcn_noisy=None curated_reps=5 split=1
$ python -m upb_audio_tagging_2019.train with model.fcn_noisy=None curated_reps=5 split=2
$ python -m upb_audio_tagging_2019.train with model.fcn_noisy=None curated_reps=3 split=0
$ python -m upb_audio_tagging_2019.train with model.fcn_noisy=None curated_reps=3 split=1
$ python -m upb_audio_tagging_2019.train with model.fcn_noisy=None curated_reps=3 split=2
```
#### with Multi-Task-Learning:
```bash
$ python -m upb_audio_tagging_2019.train with curated_reps=7 split=0
$ python -m upb_audio_tagging_2019.train with curated_reps=7 split=1
$ python -m upb_audio_tagging_2019.train with curated_reps=7 split=2
$ python -m upb_audio_tagging_2019.train with curated_reps=5 split=0
$ python -m upb_audio_tagging_2019.train with curated_reps=5 split=1
$ python -m upb_audio_tagging_2019.train with curated_reps=5 split=2
$ python -m upb_audio_tagging_2019.train with curated_reps=3 split=0
$ python -m upb_audio_tagging_2019.train with curated_reps=3 split=1
$ python -m upb_audio_tagging_2019.train with curated_reps=3 split=2
```

#### with Relabeling:
The relabeling procedure is kind of requires to train 15 models for relabeling
##### Train models for relabeling:
```bash
$ python -m upb_audio_tagging_2019.train with curated_reps=9 split=0 fold=0
$ python -m upb_audio_tagging_2019.train with curated_reps=9 split=0 fold=1
$ python -m upb_audio_tagging_2019.train with curated_reps=9 split=0 fold=2
$ python -m upb_audio_tagging_2019.train with curated_reps=9 split=0 fold=3
$ python -m upb_audio_tagging_2019.train with curated_reps=9 split=0 fold=4
$ python -m upb_audio_tagging_2019.train with curated_reps=9 split=1 fold=0
$ python -m upb_audio_tagging_2019.train with curated_reps=9 split=1 fold=1
$ python -m upb_audio_tagging_2019.train with curated_reps=9 split=1 fold=2
$ python -m upb_audio_tagging_2019.train with curated_reps=9 split=1 fold=3
$ python -m upb_audio_tagging_2019.train with curated_reps=9 split=1 fold=4
$ python -m upb_audio_tagging_2019.train with curated_reps=9 split=2 fold=0
$ python -m upb_audio_tagging_2019.train with curated_reps=9 split=2 fold=1
$ python -m upb_audio_tagging_2019.train with curated_reps=9 split=2 fold=2
$ python -m upb_audio_tagging_2019.train with curated_reps=9 split=2 fold=3
$ python -m upb_audio_tagging_2019.train with curated_reps=9 split=2 fold=4
```

##### Perform relabeling:

##### Train on relabeled data:

### Inference:
