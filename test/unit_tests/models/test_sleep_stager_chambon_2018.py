import numpy as np
import torch

from braindecode.models import SleepStagerChambon2018


def test_sleep_stager_chambon_2018_classification_and_features():
    n_channels = 2
    sfreq = 100
    input_size_s = 30
    n_classes = 5
    n_examples = 4

    rng = np.random.RandomState(42)
    X = rng.randn(n_examples, n_channels, int(sfreq * input_size_s))
    X = torch.from_numpy(X.astype(np.float32))

    # Classification output
    model = SleepStagerChambon2018(
        n_chans=n_channels,
        sfreq=sfreq,
        input_window_seconds=input_size_s,
        n_outputs=n_classes,
        return_features=False,
    )
    model.eval()
    y = model(X)
    assert y.shape == (n_examples, n_classes)

    # Feature return
    model_feats = SleepStagerChambon2018(
        n_chans=n_channels,
        sfreq=sfreq,
        input_window_seconds=input_size_s,
        n_outputs=n_classes,
        return_features=True,
    )
    model_feats.eval()
    feats = model_feats(X)
    assert feats.shape == (n_examples, model_feats.len_last_layer)
    y_from_feats = model_feats.final_layer(feats)
    assert y_from_feats.shape == (n_examples, n_classes)
