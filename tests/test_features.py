import unittest

import numpy as np

from maldi_t3ss.features import build_feature_spaces, fit_predict_ensemble


class FeatureTests(unittest.TestCase):
    def test_feature_selection_and_scaling_use_training_data_only(self):
        rng = np.random.default_rng(5)
        isolate = rng.random((12, 100), dtype=np.float32)
        replicate = np.repeat(isolate, 2, axis=0) + rng.normal(0, 0.01, (24, 100))
        mapping = np.repeat(np.arange(12), 2)
        spaces = build_feature_spaces(isolate, replicate, mapping, 2001)
        y = np.array([0, 1] * 6)
        train, test = np.arange(8), np.arange(8, 12)
        original, selected = fit_predict_ensemble(spaces, y, train, test, 5, 0.5, 1.0, 9)
        isolate_changed = isolate.copy()
        isolate_changed[test] *= 1000
        changed = build_feature_spaces(isolate_changed, replicate, mapping, 2001)
        _, selected_changed = fit_predict_ensemble(changed, y, train, test, 5, 0.5, 1.0, 9)
        self.assertTrue(all(np.array_equal(selected[name], selected_changed[name]) for name in selected))
        self.assertTrue(np.isfinite(original).all())

