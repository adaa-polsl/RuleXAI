from rulexai import rule
import unittest
import rulekit
from rulekit.main import RuleKit
from rulexai.explainer import RuleExplainer
from sklearn.datasets import load_iris
import pandas as pd
from rulekit.classification import RuleClassifier
from rulekit.params import Measures
from sklearn.tree import DecisionTreeClassifier
from Orange.data.table import Table
from Orange.data.pandas_compat import table_to_frame
from Orange.classification import CN2UnorderedLearner

import os 
from pathlib import Path
import numpy as np

class TestFunctionalities(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        RuleKit.init()

    def test_creating_model_with_rules_RuleKit(self):
        data = load_iris()
        x = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.DataFrame(data.target.astype(str), columns=["label"]) 

        clf = RuleClassifier(
            induction_measure=Measures.C2,
            pruning_measure=Measures.C2,
            voting_measure=Measures.C2,
        )
        clf.fit(x, data.target.astype(str))

        explainer = RuleExplainer(clf, x, y, type="classification")

        assert len(clf.model.rules) == len(explainer.model.rules)

    
    def test_creating_model_with_rules_Sklearn(self):
        clf = DecisionTreeClassifier(random_state=0)
        data = load_iris()
        clf.fit(data.data, data.target)

        x = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.DataFrame(data.target.astype(str), columns=["class"]) 

        explainer = RuleExplainer(clf, x, y, type="classification")
        
        assert len(explainer.model.rules) > 0 

    def test_creating_model_with_rules_Orange(self):
        data = Table("iris")
        learner = CN2UnorderedLearner()
        clf = learner(data)
        df= table_to_frame(data)
        x = df.drop(['iris'], axis=1)
        y = df['iris']

        explainer = RuleExplainer(clf, x, y, type="classification")
        
        assert len(explainer.model.rules) > 0 


    def test_features_importances(self):

        results_path = "resources/features_importances.csv"
        results_path = os.path.join(Path(__file__).resolve().parent, results_path)

        data = load_iris()
        x = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.DataFrame(data.target.astype(str), columns=["label"]) 

        clf = RuleClassifier(
            induction_measure=Measures.C2,
            pruning_measure=Measures.C2,
            voting_measure=Measures.C2,
        )
        clf.fit(x, data.target.astype(str))

        explainer = RuleExplainer(clf, x, y, type="classification")
        explainer.explain()

        features_importances = explainer.feature_importances_
        features_importances.replace("-", np.nan, inplace = True)
        features_importances = features_importances.round(4)


        features_importances_ground_truth = pd.read_csv(results_path, sep = ";")
        features_importances_ground_truth.replace("-", np.nan, inplace = True)
        features_importances_ground_truth = features_importances_ground_truth.round(4)

        assert features_importances.equals(features_importances_ground_truth)


    def test_prepare_binary_dataset(self):
        data = load_iris()
        x = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.DataFrame(data.target.astype(str), columns=["label"]) 

        clf = RuleClassifier(
            induction_measure=Measures.C2,
            pruning_measure=Measures.C2,
            voting_measure=Measures.C2,
        )
        clf.fit(x, data.target.astype(str))

        explainer = RuleExplainer(clf, x, y, type="classification")
        explainer.explain()

        binary_dataset = explainer.fit_transform(x)
        binary_dataset_10 = explainer.transform(x.iloc[0:10])

 
        binary_dataset_TOP_50 = explainer.fit_transform(x, selector=0.5)

        assert binary_dataset.shape[0] == x.shape[0]
        assert binary_dataset_10.shape[0] == 10
        assert binary_dataset.shape[1] == 14
        assert binary_dataset_TOP_50.shape[0] == x.shape[0]
        assert binary_dataset_TOP_50.shape[1] == 7



    def test_get_rules_covering_example_and_get_rules(self):
        data = load_iris()
        x = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.DataFrame(data.target.astype(str), columns=["label"]) 

        clf = RuleClassifier(
            induction_measure=Measures.C2,
            pruning_measure=Measures.C2,
            voting_measure=Measures.C2,
        )
        clf.fit(x, data.target.astype(str))

        explainer = RuleExplainer(clf, x, y, type="classification")

        rules_covering_example_0 = explainer.get_rules_covering_example(x.iloc[0,:], y.iloc[0,:])
        rules_covering_example_100 = explainer.get_rules_covering_example(x.iloc[100,:], y.iloc[100,:])

        assert len(rules_covering_example_0) == 1
        assert len(rules_covering_example_100) == 2
        assert len(explainer.get_rules()) == len(explainer.get_rules_with_basic_conditions())
        assert len(explainer.get_rules()) == len(clf.model.rules)


if __name__ == '__main__':
    unittest.main()