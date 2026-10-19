"""Deep-learning runners pass known labels to the CV splitter."""

import numpy as np
import pytest

from trustcv.frameworks._labels import split_labels


def test_tuple_dataset_gives_its_labels():
    y = np.tile([0, 1], 5)
    assert np.array_equal(split_labels((np.zeros((10, 2)), y)), y)


def test_dataset_targets_attribute_is_used_without_loading_items():
    class DS:
        targets = [0, 1, 1, 0]

        def __len__(self):
            return 4

        def __getitem__(self, i):
            raise AssertionError("items must not be loaded")

    assert list(split_labels(DS())) == [0, 1, 1, 0]


def test_monai_dicts_use_scalar_labels_only():
    assert list(split_labels([{"image": "a.nii", "label": 1}, {"image": "b.nii", "label": 0}])) == [1, 0]
    assert split_labels([{"image": "a.nii", "label": "a_seg.nii"}]) is None  # segmentation mask path


def test_unknown_data_gives_none():
    assert split_labels(object()) is None


def test_torch_runner_supports_stratified_splitters():
    torch = pytest.importorskip("torch")
    from trustcv.frameworks.pytorch import TorchCVRunner
    from trustcv.splitters import RepeatedKFold, StratifiedGroupKFold

    rng = np.random.RandomState(0)
    X = torch.tensor(rng.normal(size=(60, 4)), dtype=torch.float32)
    y = torch.tensor(np.tile([0, 1], 30))
    groups = np.repeat(np.arange(20), 3)

    def model_fn():
        return torch.nn.Sequential(torch.nn.Linear(4, 2))

    for splitter, kwargs in [(StratifiedGroupKFold(n_splits=3), {"groups": groups}),
                             (RepeatedKFold(n_splits=3, n_repeats=1, stratify=True), {})]:
        result = TorchCVRunner(model_fn, splitter).run((X, y), epochs=1, loss_fn=torch.nn.CrossEntropyLoss(), **kwargs)
        assert len(result.scores) == 3
