import numpy as np
import torch
from einops import rearrange
from padertorch.data import example_to_device
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision.utils import make_grid
from upb_audio_tagging_2019.lwlrap import lwlrap_from_precisions
from upb_audio_tagging_2019.lwlrap import positive_class_precisions


class CRNN(nn.Module):
    def __init__(
            self, cnn_2d, cnn_1d, enc, fcn, fcn_noisy, *, decision_boundary=.5
    ):
        super().__init__()
        self._cnn_2d = cnn_2d
        self._cnn_1d = cnn_1d
        self._enc = enc
        self._fcn = fcn
        self._fcn_noisy = fcn_noisy
        self.decision_boundary = decision_boundary

    def cnn_2d(self, x, seq_len=None):
        if self._cnn_2d is not None:
            x = self._cnn_2d(x)
            if seq_len is not None:
                in_shape = [(128, n) for n in seq_len]
                out_shape = self._cnn_2d.get_out_shape(in_shape)
                seq_len = [s[-1] for s in out_shape]
        if x.dim() != 3:
            assert x.dim() == 4
            x = rearrange(x, 'b c f t -> b (c f) t')
        return x, seq_len

    def cnn_1d(self, x, seq_len=None):
        if self._cnn_1d is not None:
            x = self._cnn_1d(x)
            if seq_len is not None:
                seq_len = self._cnn_1d.get_out_shape(seq_len)
        return x, seq_len

    def enc(self, x, seq_len=None):
        if isinstance(self._enc, nn.RNNBase):
            if self._enc.batch_first:
                x = rearrange(x, 'b f t -> b t f')
            else:
                x = rearrange(x, 'b f t -> t b f')
            if seq_len is not None:
                x = pack_padded_sequence(
                    x, seq_len, batch_first=self._enc.batch_first
                )
            x, _ = self._enc(x)
            if seq_len is not None:
                x = pad_packed_sequence(x, batch_first=self._enc.batch_first)[0]
            if not self._enc.batch_first:
                x = rearrange(x, 't b f -> b t f')
        else:
            raise NotImplementedError
        return TakeLast()(x, seq_len=seq_len)

    def out(self, x, is_noisy):
        y = self._fcn(x)
        if self._fcn_noisy is not None:
            y_noisy = self._fcn_noisy(x)
            while is_noisy.dim() < y.dim():
                is_noisy = is_noisy.unsqueeze(-1)
            y = is_noisy * y_noisy + (1. - is_noisy) * y
        return nn.Sigmoid()(y)

    def forward(self, inputs):
        x = inputs['features'].transpose(-2, -1)
        seq_len = inputs['seq_len']
        is_noisy = inputs['is_noisy']
        x, seq_len = self.cnn_2d(x, seq_len)
        x, seq_len = self.cnn_1d(x, seq_len)
        x = self.enc(x, seq_len)
        return self.out(x, is_noisy)

    def review(self, inputs, outputs):
        # compute loss
        x = inputs['features'].transpose(-2, -1)
        targets = inputs['events']
        if outputs.dim() == 3:  # (B, T, K)
            if targets.dim() == 2:   # (B, K)
                targets = targets.unsqueeze(1).expand(outputs.shape)
            outputs = outputs.contiguous().view((-1, outputs.shape[-1]))
            targets = targets.contiguous().view((-1, targets.shape[-1]))
        assert outputs.dim() == targets.dim() == 2
        bce = nn.BCELoss(reduction='none')(outputs, targets).sum(-1)

        # create review including metrics and visualizations
        labels, label_ranked_precisions = positive_class_precisions(
            targets.cpu().data.numpy(),
            outputs.cpu().data.numpy()
        )
        decision = (outputs.detach() > self.decision_boundary).float()
        true_pos = (decision * targets).sum()
        false_pos = (decision * (1.-targets)).sum()
        false_neg = ((1.-decision) * targets).sum()
        review = dict(
            loss=bce.mean(),
            scalars=dict(
                labels=labels,
                label_ranked_precisions=label_ranked_precisions,
                true_pos=true_pos.cpu().data.numpy(),
                false_pos=false_pos.cpu().data.numpy(),
                false_neg=false_neg.cpu().data.numpy()
            ),
            histograms=dict(),
            images=dict(
                features=x[:3]
            )
        )
        return review

    def modify_summary(self, summary):
        # compute lwlrap
        if 'labels' in summary['scalars']:
            labels = summary['scalars'].pop('labels')
            label_ranked_precisions = summary['scalars'].pop(
                'label_ranked_precisions'
            )
            summary['scalars']['lwlrap'] = lwlrap_from_precisions(
                label_ranked_precisions, labels
            )[0]

        # compute precision, recall and fscore for each decision boundary
        if 'true_pos' in summary['scalars']:
            tp = np.sum(summary['scalars'].pop('true_pos'))
            fp = np.sum(summary['scalars'].pop('false_pos'))
            fn = np.sum(summary['scalars'].pop('false_neg'))
            p = tp/(tp+fp)
            r = tp/(tp+fn)
            summary['scalars'][f'precision'] = p
            summary['scalars'][f'recall'] = r
            summary['scalars'][f'fscore'] = 2*(p*r)/(p+r)

        for key, scalar in summary['scalars'].items():
            summary['scalars'][key] = np.mean(scalar)

        for key, image in summary['images'].items():
            if image.dim() == 4 and image.shape[1] > 1:
                image = image[:, 0]
            if image.dim() == 3:
                image = image.unsqueeze(1)
            summary['images'][key] = make_grid(
                image.flip(2),  normalize=True, scale_each=False, nrow=1
            )
        return summary


class TakeLast(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x, seq_len=None):
        if seq_len is None:
            x = x[:, -1]
        else:
            x = x[torch.arange(x.shape[0]), seq_len - 1]
        return x


def batch_norm_update(
        model, dataset, feature_key, batch_dim=0,
        device=0 if torch.cuda.is_available() else 'cpu'
):
    r"""Updates BatchNorm running_mean, running_var buffers in the model.

    It performs one pass over data in `loader` to estimate the activation
    statistics for BatchNorm layers in the model.

    Args:
        dataset: dataset to compute the activation statistics on.
            Each data batch should be either a dict, or a list/tuple.

        model: model for which we seek to update BatchNorm statistics.

        feature_key: key to get an input tensor to read batch_size from

        device: If set, data will be transferred to :attr:`device`
            before being passed into :attr:`model`.
    """
    if not _check_bn(model):
        return
    was_training = model.training
    model.train()

    model.to(device)

    momenta = {}
    model.apply(_reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    with torch.no_grad():
        for i, example in enumerate(dataset):
            example = example_to_device(example, device)
            b = example[feature_key].size(batch_dim)

            momentum = b / float(n + b)
            for module in momenta.keys():
                module.momentum = momentum

            model(example)

            n += b

    model.apply(lambda module: _set_momenta(module, momenta))
    model.train(was_training)


# BatchNorm utils
def _check_bn_apply(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def _check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn_apply(module, flag))
    return flag[0]


def _reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]
