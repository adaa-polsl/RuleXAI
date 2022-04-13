import unittest
import rulekit

from rulekit.main import RuleKit
import pandas as pd
from rulekit import RuleKit
from rulekit.classification import RuleClassifier
from rulekit.regression import RuleRegressor
from rulekit.survival import SurvivalRules
from rulekit.params import Measures
from scipy.io import arff
import numpy as np
import os 
from pathlib import Path
from rulexai.explainer import RuleExplainer


class TestConditionsImportances(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        RuleKit.init()

    
    def test_classification(self):
        classification_resources = "resources/classification/"
        dataset_path = classification_resources + "iris.arff"
        results_path = classification_resources + "results.csv" 
        results_split_path = classification_resources + "results_split.csv" 
        dataset_path = os.path.join(Path(__file__).resolve().parent, dataset_path)
        results_path = os.path.join(Path(__file__).resolve().parent, results_path)
        results_split_path = os.path.join(Path(__file__).resolve().parent, results_split_path)

        train_df = pd.DataFrame(arff.loadarff(dataset_path)[0])

        # code to change encoding of the file
        tmp_df = train_df.select_dtypes([object])
        tmp_df = tmp_df.stack().str.decode("utf-8").unstack()
        for col in tmp_df:
            train_df[col] = tmp_df[col].replace({"?": None})

        x = train_df.drop(["class"], axis=1)
        y = train_df["class"]

        # RuleKit
        clf = RuleClassifier(
            induction_measure=Measures.C2,
            pruning_measure=Measures.C2,
            voting_measure=Measures.C2,
        )
        clf.fit(x, y)


        # RuleXai
        explainer = RuleExplainer(clf, x, y, type="classification")
        explainer.explain()
        conditions_importances = explainer.condition_importances_
        conditions_importances.replace("-", np.nan, inplace = True)
        conditions_importances = conditions_importances.round(4)

        importances_ground_truth = pd.read_csv(results_path, sep = ";")
        importances_ground_truth = importances_ground_truth.round(4)

        # with splitting conditions to basic
        explainer.explain(basic_conditions=True)
        basic_conditions_importances= explainer.condition_importances_
        basic_conditions_importances.replace("-", np.nan, inplace = True)
        basic_conditions_importances = basic_conditions_importances.round(4)

        basic_importances_ground_truth = pd.read_csv(results_split_path, sep = ";")
        basic_importances_ground_truth = basic_importances_ground_truth.round(4)

        assert conditions_importances.equals(importances_ground_truth)
        assert basic_conditions_importances.equals(basic_importances_ground_truth)


    def test_regression(self):

        regression_resources = "resources/regression/"
        dataset_path = regression_resources + "diabetes.arff"
        results_path = regression_resources + "results.csv" 
        results_split_path = regression_resources + "results_split.csv" 
        dataset_path = os.path.join(Path(__file__).resolve().parent, dataset_path)
        results_path = os.path.join(Path(__file__).resolve().parent, results_path)
        results_split_path = os.path.join(Path(__file__).resolve().parent, results_split_path)

        train_df = pd.DataFrame(arff.loadarff(dataset_path)[0])

        # code to change encoding of the file
        tmp_df = train_df.select_dtypes([object])
        if (tmp_df.size != 0):  
            tmp_df = tmp_df.stack().str.decode("utf-8").unstack()
            for col in tmp_df:
                train_df[col] = tmp_df[col].replace({"?": None})

        x = train_df.drop(["class"], axis=1)
        y = train_df["class"]

        df = train_df

        # RuleKit
        reg = RuleRegressor(
            induction_measure=Measures.C2,
            pruning_measure=Measures.C2,
            voting_measure=Measures.C2,
        )
        reg.fit(x, y)


        # RuleXai
        explainer = RuleExplainer(reg, x, y, type="regression")
        explainer.explain()
        conditions_importances = explainer.condition_importances_
        conditions_importances = conditions_importances.round(4)

        importances_ground_truth = pd.read_csv(results_path, sep = ";")
        importances_ground_truth = importances_ground_truth.round(4)

        # with splitting conditions to basic
        explainer.explain(basic_conditions=True)
        basic_conditions_importances= explainer.condition_importances_
        basic_conditions_importances = basic_conditions_importances.round(4)

        basic_importances_ground_truth = pd.read_csv(results_split_path, sep = ";")
        basic_importances_ground_truth = basic_importances_ground_truth.round(4)

        assert conditions_importances.equals(importances_ground_truth)
        assert basic_conditions_importances.equals(basic_importances_ground_truth)


 

if __name__ == '__main__':
    unittest.main()