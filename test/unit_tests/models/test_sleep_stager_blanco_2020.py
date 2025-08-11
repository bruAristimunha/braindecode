import numpy as np
import torch

from braindecode.models import SleepStagerBlanco2020


def test_sleep_stager_blanco_2020_classification_and_features():
    n_channels = 2
    sfreq = 50
    input_size_s = 30
    n_classes = 3
    n_examples = 4

    rng = np.random.RandomState(42)
    X = rng.randn(n_examples, n_channels, int(sfreq * input_size_s))
    X = torch.from_numpy(X.astype(np.float32))

    # Classification output
    model = SleepStagerBlanco2020(
        n_chans=n_channels,
        sfreq=sfreq,
        n_groups=2,
        input_window_seconds=input_size_s,
        n_outputs=n_classes,
        return_features=False,
    )
    model.eval()
    y = model(X)
    assert y.shape == (n_examples, n_classes)

    # Feature return
    model_feats = SleepStagerBlanco2020(
        n_chans=n_channels,
        sfreq=sfreq,
        n_groups=2,
        input_window_seconds=input_size_s,
        n_outputs=n_classes,
        return_features=True,
    )
    model_feats.eval()
    feats = model_feats(X)
    assert feats.shape == (n_examples, model_feats.len_last_layer)
    y_from_feats = model_feats.final_layer(feats)
    assert y_from_feats.shape == (n_examples, n_classes)
