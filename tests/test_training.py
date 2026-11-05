"""Small synthetic tests; these values are not robot measurements."""
import copy
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd

from train_v3 import train
from v3.detectors import threshold
from v3.feature_helpers import describe


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame(dict(cycle_id=[f"synthetic_{i}" for i in range(8)],
            cohort=["train"]*4+["validation"]*2+["evaluation"]*2,
            f1=[1.,2.,np.nan,4.,5.,6.,100.,200.], f2=[2.,1.,4.,3.,6.,5.,50.,60.]))
        self.config = dict(seed=20261031,configurations={"test":["f1","f2"]},
                           algorithms=["Statistical","IsolationForest","PCA"])

    def test_evaluation_values_do_not_change_fitting_or_threshold(self):
        changed = self.data.copy()
        changed.loc[changed.cohort.eq("evaluation"),["f1","f2"]] = 1e9
        with tempfile.TemporaryDirectory() as temporary:
            outputs = [Path(temporary)/"a",Path(temporary)/"b"]
            for frame,out in zip([self.data,changed],outputs):
                train(frame,self.config,out)
            for algorithm in self.config["algorithms"]:
                bundles = [joblib.load(out/"models"/f"test_{algorithm}.joblib") for out in outputs]
                self.assertEqual(bundles[0]["threshold"],bundles[1]["threshold"])
                np.testing.assert_array_equal(bundles[0]["detector"].scaler.mean_,bundles[1]["detector"].scaler.mean_)
                np.testing.assert_array_equal(bundles[0]["detector"].imputer.statistics_,[2.,2.5])

    def test_duplicate_cycles_and_metadata_predictors_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            bad = self.data.copy()
            bad.loc[1,"cycle_id"] = bad.loc[0,"cycle_id"]
            with self.assertRaises(ValueError):
                train(bad,self.config,Path(temporary)/"duplicate")
            config = copy.deepcopy(self.config)
            config["configurations"]["test"] = ["cohort"]
            with self.assertRaises(ValueError):
                train(self.data,config,Path(temporary)/"metadata")

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileExistsError):
                train(self.data,self.config,temporary)

    def test_original_quantile_and_population_statistics(self):
        self.assertEqual(threshold(np.array([1.,2.,3.,4.])),4.)
        values = describe([1.,2.,3.],"signal")
        self.assertAlmostEqual(values["signal_std"],np.std([1.,2.,3.],ddof=0))
        self.assertEqual(values["signal_abs_p95_nearest_rank"],3.)


if __name__ == "__main__":
    unittest.main()
