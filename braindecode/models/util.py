# Authors: Robin Schirrmeister <robintibor@gmail.com>
#          Hubert Banville <hubert.jbanville@gmail.com>
#
# License: BSD (3-clause)
import inspect

import numpy as np
import torch
from scipy.special import log_softmax
from sklearn.utils import deprecated

import braindecode.models as models


@deprecated(
    "will be removed in version 1.0. Use EEGModuleMixin.to_dense_prediction_model method directly "
    "on the model object."
)
def to_dense_prediction_model(model, axis=(2, 3)):
    """
    Transform a sequential model with strides to a model that outputs
    dense predictions by removing the strides and instead inserting dilations.
    Modifies model in-place.

    Parameters
    ----------
    model: torch.nn.Module
        Model which modules will be modified
    axis: int or (int,int)
        Axis to transform (in terms of intermediate output axes)
        can either be 2, 3, or (2,3).

    Notes
    -----
    Does not yet work correctly for average pooling.
    Prior to version 0.1.7, there had been a bug that could move strides
    backwards one layer.

    """
    if not hasattr(axis, "__len__"):
        axis = [axis]
    assert all([ax in [2, 3] for ax in axis]), "Only 2 and 3 allowed for axis"
    axis = np.array(axis) - 2
    stride_so_far = np.array([1, 1])
    for module in model.modules():
        if hasattr(module, "dilation"):
            assert module.dilation == 1 or (module.dilation == (1, 1)), (
                "Dilation should equal 1 before conversion, maybe the model is "
                "already converted?"
            )
            new_dilation = [1, 1]
            for ax in axis:
                new_dilation[ax] = int(stride_so_far[ax])
            module.dilation = tuple(new_dilation)
        if hasattr(module, "stride"):
            if not hasattr(module.stride, "__len__"):
                module.stride = (module.stride, module.stride)
            stride_so_far *= np.array(module.stride)
            new_stride = list(module.stride)
            for ax in axis:
                new_stride[ax] = 1
            module.stride = tuple(new_stride)


@deprecated(
    "will be removed in version 1.0. Use EEGModuleMixin.get_output_shape method directly on the "
    "model object."
)
def get_output_shape(model, in_chans, input_window_samples):
    """Returns shape of neural network output for batch size equal 1.

    Returns
    -------
    output_shape: tuple
        shape of the network output for `batch_size==1` (1, ...)
    """
    with torch.no_grad():
        dummy_input = torch.ones(
            1,
            in_chans,
            input_window_samples,
            dtype=next(model.parameters()).dtype,
            device=next(model.parameters()).device,
        )
        output_shape = model(dummy_input).shape
    return output_shape


def _pad_shift_array(x, stride=1):
    """Zero-pad and shift rows of a 3D array.

    E.g., used to align predictions of corresponding windows in
    sequence-to-sequence models.

    Parameters
    ----------
    x : np.ndarray
        Array of shape (n_rows, n_classes, n_windows).
    stride : int
        Number of non-overlapping elements between two consecutive sequences.

    Returns
    -------
    np.ndarray :
        Array of shape (n_rows, n_classes, (n_rows - 1) * stride + n_windows)
        where each row is obtained by zero-padding the corresponding row in
        ``x`` before and after in the last dimension.
    """
    if x.ndim != 3:
        raise NotImplementedError(
            f"x must be of shape (n_rows, n_classes, n_windows), got {x.shape}"
        )
    x_padded = np.pad(x, ((0, 0), (0, 0), (0, (x.shape[0] - 1) * stride)))
    orig_strides = x_padded.strides
    new_strides = (
        orig_strides[0] - stride * orig_strides[2],
        orig_strides[1],
        orig_strides[2],
    )
    return np.lib.stride_tricks.as_strided(x_padded, strides=new_strides)


def aggregate_probas(logits, n_windows_stride=1):
    """Aggregate predicted probabilities with self-ensembling.

    Aggregate window-wise predicted probabilities obtained on overlapping
    sequences of windows using multiplicative voting as described in
    [Phan2018]_.

    Parameters
    ----------
    logits : np.ndarray
        Array of shape (n_sequences, n_classes, n_windows) containing the
        logits (i.e. the raw unnormalized scores for each class) for each
        window of each sequence.
    n_windows_stride : int
        Number of windows between two consecutive sequences. Default is 1
        (maximally overlapping sequences).

    Returns
    -------
    np.ndarray :
        Array of shape ((n_rows - 1) * stride + n_windows, n_classes)
        containing the aggregated predicted probabilities for each window
        contained in the input sequences.

    References
    ----------
    .. [Phan2018] Phan, H., Andreotti, F., Cooray, N., Chén, O. Y., &
        De Vos, M. (2018). Joint classification and prediction CNN framework
        for automatic sleep stage classification. IEEE Transactions on
        Biomedical Engineering, 66(5), 1285-1296.
    """
    log_probas = log_softmax(logits, axis=1)
    return _pad_shift_array(log_probas, stride=n_windows_stride).sum(axis=0).T


models_dict = {}

# For the models inside the init model, go through all the models
# check those have the EEGMixin class inherited. If they are, add them to the
# list.


def _init_models_dict():
    for m in inspect.getmembers(models, inspect.isclass):
        if (
            issubclass(m[1], models.base.EEGModuleMixin)
            and m[1] != models.base.EEGModuleMixin
        ):
            models_dict[m[0]] = m[1]


################################################################
# Test cases for models
#
# This list should be updated whenever a new model is added to
# braindecode (otherwise `test_completeness__models_test_cases`
# will fail).
# Each element in the list should be a tuple with structure
# (model_class, required_params, signal_params), such that:
#
# model_name: str
#   The name of the class of the model to be tested.
# required_params: list[str]
#   The signal-related parameters that are needed to initialize
#   the model.
# signal_params: dict | None
#   The characteristics of the signal that should be passed to
#   the model tested in case the default_signal_params are not
#   compatible with this model.
#   The keys of this dictionary can only be among those of
#   default_signal_params.
################################################################
models_mandatory_parameters = [
    ("ATCNet", ["n_chans", "n_outputs", "n_times"], None),
    ("BDTCN", ["n_chans", "n_outputs"], None),
    ("Deep4Net", ["n_chans", "n_outputs", "n_times"], None),
    ("DeepSleepNet", ["n_outputs"], None),
    ("EEGConformer", ["n_chans", "n_outputs", "n_times"], None),
    ("EEGInceptionERP", ["n_chans", "n_outputs", "n_times", "sfreq"], None),
    ("EEGInceptionMI", ["n_chans", "n_outputs", "n_times", "sfreq"], None),
    ("EEGITNet", ["n_chans", "n_outputs", "n_times"], None),
    ("EEGNetv1", ["n_chans", "n_outputs", "n_times"], None),
    ("EEGNetv4", ["n_chans", "n_outputs", "n_times"], None),
    ("EEGResNet", ["n_chans", "n_outputs", "n_times"], None),
    ("ShallowFBCSPNet", ["n_chans", "n_outputs", "n_times"], None),
    (
        "SleepStagerBlanco2020",
        ["n_chans", "n_outputs", "n_times"],
        # n_chans dividable by n_groups=2:
        dict(chs_info=[dict(ch_name=f"C{i}", kind="eeg") for i in range(1, 5)]),
    ),
    ("SleepStagerChambon2018", ["n_chans", "n_outputs", "n_times", "sfreq"], None),
    (
        "SleepStagerEldele2021",
        ["n_outputs", "n_times", "sfreq"],
        dict(sfreq=100, n_times=3000, chs_info=[dict(ch_name="C1", kind="eeg")]),
    ),  # 1 channel
    ("TIDNet", ["n_chans", "n_outputs", "n_times"], None),
    ("USleep", ["n_chans", "n_outputs", "n_times", "sfreq"], dict(sfreq=128)),
    ("BIOT", ["n_chans", "n_outputs", "sfreq"], None),
    ("AttentionBaseNet", ["n_chans", "n_outputs", "n_times"], None),
    ("Labram", ["n_chans", "n_outputs", "n_times"], None),
    ("EEGSimpleConv", ["n_chans", "n_outputs", "sfreq"], None),
    ("SPARCNet", ["n_chans", "n_outputs", "n_times"], None),
    ("ContraWR", ["n_chans", "n_outputs", "sfreq"], dict(sfreq=200)),
    ("EEGNeX", ["n_chans", "n_outputs", "n_times"], None),
    ("TSceptionV1", ["n_chans", "n_outputs", "n_times", "sfreq"], dict(sfreq=200)),
    ("EEGTCNet", ["n_chans", "n_outputs", "n_times"], None),
    ("SyncNet", ["n_chans", "n_outputs", "n_times"], None),
    ("MSVTNet", ["n_chans", "n_outputs", "n_times"], None),
    ("EEGMiner", ["n_chans", "n_outputs", "n_times", "sfreq"], dict(sfreq=200)),
    ("CTNet", ["n_chans", "n_outputs", "n_times"], None),
    ("SincShallowNet", ["n_chans", "n_outputs", "n_times", "sfreq"], dict(sfreq=250)),
    ("SCCNet", ["n_chans", "n_outputs", "n_times", "sfreq"], dict(sfreq=200)),
    ("SignalJEPA", ["chs_info"], None),
    ("SignalJEPA_Contextual", ["chs_info", "n_times", "n_outputs"], None),
    ("SignalJEPA_PostLocal", ["n_chans", "n_times", "n_outputs"], None),
    ("SignalJEPA_PreLocal", ["n_chans", "n_times", "n_outputs"], None),
]

################################################################
# List of models that are not meant for classification
#
# Their output shape may difer from the expected output shape
# for classification models.
################################################################
non_classification_models = [
    "SignalJEPA",
]
