import unittest

import numpy as np

from maldi_t3ss.validation import allele_cluster_groups, grouped_splits


class ValidationTests(unittest.TestCase):
    def test_grouped_splits_are_lineage_disjoint(self):
        y = np.array([0, 0, 1, 1] * 5)
        groups = np.repeat([f"g{i}" for i in range(10)], 2)
        for train, test in grouped_splits(y, groups, 5, 7):
            self.assertFalse(set(groups[train]).intersection(groups[test]))
            self.assertEqual(set(y[train]), {0, 1})
            self.assertEqual(set(y[test]), {0, 1})

    def test_allele_clustering_does_not_join_missing_profiles(self):
        profiles = np.array(
            [
                ["1", "2", "3", "4", "5", "6", "7"],
                ["1", "2", "3", "4", "5", "6", "8"],
                ["", "2", "3", "4", "5", "6", "7"],
            ],
            dtype=object,
        )
        groups = allele_cluster_groups(profiles, 1)
        self.assertEqual(groups[0], groups[1])
        self.assertNotEqual(groups[2], groups[0])

