import numpy as np
import torch

from braindecode.models import SleepStagerEldele2021


def test_sleep_stager_eldele_2021_classification_and_features():
    sfreq = 100
    input_size_s = 30
    n_classes = 5
    n_examples = 4
    n_channels = 1

    rng = np.random.RandomState(42)
    X = rng.randn(n_examples, n_channels, int(sfreq * input_size_s))
    X = torch.from_numpy(X.astype(np.float32))

    # Classification output
    model = SleepStagerEldele2021(
        sfreq=sfreq,
        n_outputs=n_classes,
        input_window_seconds=input_size_s,
        return_features=False,
    )
    model.eval()
    y = model(X)
    assert y.shape == (n_examples, n_classes)

    # Feature return
    model_feats = SleepStagerEldele2021(
        sfreq=sfreq,
        n_outputs=n_classes,
        input_window_seconds=input_size_s,
        return_features=True,
    )
    model_feats.eval()
    feats = model_feats(X)
    assert feats.shape == (n_examples, model_feats.len_last_layer)
    y_from_feats = model_feats.final_layer(feats)
    assert y_from_feats.shape == (n_examples, n_classes)
