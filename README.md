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

Clone the repo
```bash
$ git clone https://github.com/fgnt/upb_audio_tagging_2019.git
$ cd upb_audio_tagging_2019
```
Use the environmental variable FSDKaggle2019DIR to direct the repository to the challenge data:
```bash
$ export FSDKaggle2019DIR=/path/to/fsd_kaggle_2019
```

Install this package
```bash
$ pip install --user -e .
```
