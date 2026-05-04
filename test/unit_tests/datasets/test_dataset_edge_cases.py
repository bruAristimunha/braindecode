import mne
import numpy as np
import pytest

from braindecode.datasets import BaseConcatDataset, RawDataset
from braindecode.preprocessing import Preprocessor, preprocess


@pytest.fixture
def datasets_with_mismatched_channels():
    rng = np.random.RandomState(42)
    # DS 1: ch0, ch1
    info1 = mne.create_info(ch_names=["ch0", "ch1"], sfreq=100, ch_types="eeg")
    raw1 = mne.io.RawArray(rng.randn(2, 1000), info1)
    ds1 = RawDataset(raw1, description={"id": 1})

    # DS 2: ch1, ch0 (reversed order)
    info2 = mne.create_info(ch_names=["ch1", "ch0"], sfreq=100, ch_types="eeg")
    raw2 = mne.io.RawArray(rng.randn(2, 1000), info2)
    ds2 = RawDataset(raw2, description={"id": 2})

    # DS 3: ch0, ch2 (mismatched subset)
    info3 = mne.create_info(ch_names=["ch0", "ch2"], sfreq=100, ch_types="eeg")
    raw3 = mne.io.RawArray(rng.randn(2, 1000), info3)
    ds3 = RawDataset(raw3, description={"id": 3})

    return [ds1, ds2, ds3]

def test_concat_dataset_mismatched_channels(datasets_with_mismatched_channels):
    """
    Test that BaseConcatDataset can be created with mismatched channels,
    but ensure preprocessing that expects consistency fails or handles it.
    """
    ds_list = datasets_with_mismatched_channels
    concat_ds = BaseConcatDataset(ds_list)

    # Preprocessing with a standard MNE function that exists in both but different order
    # MNE's pick_channels should work fine on each individual raw
    preprocessors = [Preprocessor('pick_channels', ch_names=['ch0'])]
    preprocess(concat_ds, preprocessors)

    # Now all should have only 'ch0'
    for ds in concat_ds.datasets:
        assert ds.raw.ch_names == ['ch0']

def test_set_description_edge_cases():
    rng = np.random.RandomState(42)
    info = mne.create_info(ch_names=["ch0"], sfreq=100, ch_types="eeg")
    raw = mne.io.RawArray(rng.randn(1, 100), info)
    ds1 = RawDataset(raw, description={"subject": 1})
    ds2 = RawDataset(raw, description={"subject": 2})
    concat_ds = BaseConcatDataset([ds1, ds2])

    # 1. Mismatched length
    with pytest.raises(ValueError, match="Length of values .* does not match"):
        concat_ds.set_description({"new_col": [1, 2, 3]})

    # 2. Overwrite without permission
    with pytest.raises(AssertionError, match="already in description"):
        concat_ds.set_description({"subject": [3, 4]}, overwrite=False)

    # 3. Successful overwrite
    concat_ds.set_description({"subject": [3, 4]}, overwrite=True)
    assert concat_ds.description["subject"].tolist() == [3, 4]

def test_base_concat_dataset_empty_list():
    """Ensure proper error when creating empty concat dataset."""
    with pytest.raises(ValueError, match="datasets should not be an empty iterable"):
        BaseConcatDataset([])

def test_transform_propagation():
    """Ensure transforms are correctly set and kept across datasets."""
    rng = np.random.RandomState(42)
    info = mne.create_info(ch_names=["ch0"], sfreq=100, ch_types="eeg")
    raw = mne.io.RawArray(rng.randn(1, 100), info)
    ds = RawDataset(raw, description={"id": 1})
    concat_ds = BaseConcatDataset([ds])

    def dummy_transform(x):
        return x * 2

    concat_ds.transform = dummy_transform
    assert concat_ds.datasets[0].transform == dummy_transform

    # After split, transform should persist
    splits = concat_ds.split(by=[0])
    assert splits["0"].datasets[0].transform == dummy_transform
