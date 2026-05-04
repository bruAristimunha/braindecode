import os

import mne
import numpy as np
import torch

from braindecode.classifier import EEGClassifier
from braindecode.datasets import BaseConcatDataset, RawDataset
from braindecode.models import ShallowFBCSPNet
from braindecode.preprocessing import (
    Preprocessor,
    create_windows_from_events,
    preprocess,
)
from braindecode.util import create_mne_dummy_raw, set_random_seeds


def test_e2e_training_pipeline(tmpdir):
    """
    Integration test for a full end-to-end training pipeline:
    1. Data generation (dummy)
    2. Preprocessing
    3. Windowing
    4. Training (EEGClassifier)
    5. Prediction
    6. Saving/Loading state
    """
    set_random_seeds(seed=42, cuda=False)

    # 1. Create dummy data
    n_channels = 4
    n_times = 2000 # 20 seconds at 100Hz
    sfreq = 100
    raw, _ = create_mne_dummy_raw(n_channels, n_times, sfreq, random_state=42)

    # Add annotations for 4 trials, 2 classes
    annotations = mne.Annotations(
        onset=[1, 5, 9, 13],
        duration=[2, 2, 2, 2],
        description=['class1', 'class1', 'class2', 'class2']
    )
    raw.set_annotations(annotations)

    ds = RawDataset(raw)
    concat_ds = BaseConcatDataset([ds])

    # 2. Preprocess
    preprocessors = [
        Preprocessor('pick_types', eeg=True, stim=False),
        Preprocessor(lambda x: x * 1e6), # Scale to microvolts
    ]
    preprocess(concat_ds, preprocessors)

    # 3. Create Windows
    # window_size = 1s = 100 samples
    # each 2s trial will produce 2 windows
    windows_ds = create_windows_from_events(
        concat_ds,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        window_size_samples=100,
        window_stride_samples=100,
        drop_last_window=False,
        mapping={'class1': 0, 'class2': 1}
    )

    # 4. Model & Classifier
    n_classes = 2
    model = ShallowFBCSPNet(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=100,
        final_conv_length='auto'
    )

    # We use a very small net for speed
    clf = EEGClassifier(
        model,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        train_split=None,
        batch_size=4,
        max_epochs=2,
    )

    # 5. Train
    # y=None tells skorch to use labels from the dataset
    clf.fit(windows_ds, y=None)

    # 6. Predict
    preds = clf.predict(windows_ds)
    assert len(preds) == 8


    # 7. Save and Load
    # Note: save_params/load_params is the skorch way
    f_params = os.path.join(tmpdir, "model_params.pt")
    f_optimizer = os.path.join(tmpdir, "optimizer.pt")
    f_history = os.path.join(tmpdir, "history.json")

    clf.save_params(f_params=f_params, f_optimizer=f_optimizer, f_history=f_history)

    # Create a new classifier and load
    new_clf = EEGClassifier(
        model,
        criterion=torch.nn.CrossEntropyLoss,
    )
    new_clf.initialize()
    new_clf.load_params(f_params=f_params, f_optimizer=f_optimizer, f_history=f_history)

    new_preds = new_clf.predict(windows_ds)
    np.testing.assert_allclose(preds, new_preds)

    # Verify history was loaded
    assert len(new_clf.history) == 2
