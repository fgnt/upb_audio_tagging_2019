import numpy as np


# Core calculation of label precisions for one test sample.
def positive_class_precisions(target_mat, score_mat):
    """Calculate precisions for each true class.

    Args:
      target_mat: np.array of (num_samples, num_classes) bools indicating which classes are true.
      score_mat: np.array of (num_samples, num_classes) giving the individual classifier scores.

    Returns:
      class_indices: np.array of indices of the true classes.
      precision_at_hits: np.array of precisions corresponding to each of those
        classes.
    """
    num_samples, num_classes = score_mat.shape
    class_indices = np.cumsum(np.ones_like(score_mat), axis=-1) - 1
    target_mat = target_mat > 0
    # Only calculate precisions if there are some true classes.
    if not target_mat.any():
        return np.array([]), np.array([])
    # Retrieval list of classes for this sample.
    retrieved_classes = np.argsort(score_mat, axis=-1)[:, ::-1]
    sort_idx = (np.arange(num_samples)[:, None], retrieved_classes)
    class_indices = class_indices[sort_idx]
    target_mat = target_mat[sort_idx]
    # Num hits for every truncated retrieval list.
    retrieved_cumulative_hits = np.cumsum(target_mat, axis=-1)
    ranks = np.cumsum(np.ones_like(retrieved_cumulative_hits), axis=-1)
    # Precision of retrieval list truncated at each hit, in order of pos_labels.
    precision_at_hits = (retrieved_cumulative_hits[target_mat] / ranks[target_mat])
    return class_indices[target_mat].astype(np.int), precision_at_hits


def lwlrap_from_precisions(
        precision_at_hits, class_indices, num_classes=None
):
    """Calculate label-weighted label-ranking average precision.

    Arguments:
        precision_at_hits:
        class_indices:

    Returns:
      lwlrap: overall unbalanced lwlrap which is simply
        np.sum(per_class_lwlrap * weight_per_class)
      per_class_lwlrap: np.array of (num_classes,) giving the lwlrap for each
        class.
      weight_per_class: np.array of (num_classes,) giving the prior of each
        class within the truth labels.

    """

    if num_classes is None:
        num_classes = np.max(class_indices) + 1
    per_class_lwlrap = np.zeros(num_classes)
    np.add.at(per_class_lwlrap, class_indices, precision_at_hits)
    labels_per_class = np.zeros(num_classes)
    np.add.at(labels_per_class, class_indices, 1)
    per_class_lwlrap /= np.maximum(1, labels_per_class)
    weight_per_class = labels_per_class / float(np.sum(labels_per_class))
    lwlrap = np.sum(per_class_lwlrap * weight_per_class)
    return lwlrap, per_class_lwlrap, weight_per_class


# All-in-one calculation of per-class lwlrap.
def lwlrap(target_mat, score_mat, event_wise=False):
    """Calculate label-weighted label-ranking average precision.

    Arguments:
      target_mat: np.array of (num_samples, num_classes) giving boolean ground-truth
        of presence of that class in that sample.
      score_mat: np.array of (num_samples, num_classes) giving the classifier-under-
        test's real-valued score for each class for each sample.

    Returns:
      lwlrap: overall unbalanced lwlrap which is simply
        np.sum(per_class_lwlrap * weight_per_class)


    >>> num_samples = 100
    >>> num_labels = 20

    >>> truth = np.random.rand(num_samples, num_labels) > 0.5
    >>> truth[0:1, :] = False # Ensure at least some samples with no truth labels.
    >>> scores = np.random.rand(num_samples, num_labels)

    >>> lwlrap_, per_class_lwlrap, weight_per_class = lwlrap(truth, scores)
    """
    assert target_mat.shape == score_mat.shape
    pos_class_indices, precision_at_hits = positive_class_precisions(
        target_mat, score_mat
    )
    lwlrap_, per_class_lwlrap, weight_per_class = lwlrap_from_precisions(
        precision_at_hits, pos_class_indices, num_classes=target_mat.shape[1]
    )
    if event_wise:
        return per_class_lwlrap
    else:
        return lwlrap_
