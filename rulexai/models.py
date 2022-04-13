from rulekit.operator import BaseOperator
import numpy as np
from .rule import Rule, CompoundCondition, ElementaryCondition
from rulekit import RuleKit
from rulekit.classification import RuleClassifier
from rulekit.regression import RuleRegressor
from rulekit.params import Measures
import pandas as pd
from typing import Dict
import importlib

class Model:
    def __init__(
        self, model, feature_names=None, class_names=None, label_name=None
    ) -> None:
        self.model = model
        self.rules = self.get_rules(self.model, feature_names, class_names, label_name)
        self.column_indexes: Dict[str, int] = {column_name:i for i, column_name in enumerate(list(feature_names))}


    def _map_rules_from_RuleKit(self, rules):
        preprocessed_Rules = []
        for rule in rules:
                                      
            preprocessed_Rules.append(self._map_rule_from_RuleKit(rule.__str__()))
        return preprocessed_Rules

    def _map_rules_from_sklearn(self, rules):
        preprocessed_Rules = []
        for rule in rules:
                                      
            preprocessed_Rules.append(self._map_rule_from_sklearn(rule.__str__()))
        return preprocessed_Rules

    def _map_rule_from_sklearn(self, rule):
        rule = rule[3:]
        premise, consequence = rule.split(" THEN ")
        conditions = premise.split(" AND ")
        compoundCondition = CompoundCondition()

        for condition in conditions:
            if "<=" in condition:
                attribute, value = condition.split(" <= ")
                elementaryCondition = ElementaryCondition(
                    attribute, ElementaryCondition.minus_inf, float(value), False, True, column_index=self.column_indexes[attribute]
                                           
                )
            elif "<" in condition:
                attribute, value = condition.split(" < ")
                elementaryCondition = ElementaryCondition(
                    attribute, ElementaryCondition.minus_inf, float(value), False, False, column_index=self.column_indexes[attribute]
                                            
                )
            elif ">=" in condition:
                attribute, value = condition.split(" >= ")
                elementaryCondition = ElementaryCondition(
                                     
                    attribute, float(value), ElementaryCondition.inf, True, False, column_index=self.column_indexes[attribute]
                )
            elif ">" in condition:
                attribute, value = condition.split(" > ")
                elementaryCondition = ElementaryCondition(
                                     
                    attribute, float(value), ElementaryCondition.inf, False, False, column_index=self.column_indexes[attribute]
                )
            else:
                attribute, value = condition.split(" = ")
                elementaryCondition = ElementaryCondition(attribute, str(value), column_index=self.column_indexes[attribute])
                                          

            compoundCondition.add_subcondition(elementaryCondition)

        consequence_att, consequence_value = consequence.split(" = ")
        consequence_value = consequence_value[1:-1]
        consequence = ElementaryCondition(consequence_att, consequence_value, column_index=self.column_indexes[attribute])

        return self._preprocessRule(Rule(compoundCondition, consequence))

    def _map_rules_from_list(self, rules):
        preprocessed_Rules = []
        for rule in rules:
            preprocessed_Rules.append(self._map_string_rule(rule))
        return preprocessed_Rules

    def _map_string_rule(self, rule):
        rule = rule[3:]
        premise, consequence = rule.split(" THEN ")
        conditions = premise.split(" AND ")
        compoundCondition = CompoundCondition()

        for condition in conditions:
            attribute, valueset = condition.split(" = ")
            if "," in valueset:
                left, right = valueset.split(",")
                leftClosed = False if left[0] == "(" else True
                rightClosed = False if right[-1:] == ")" else True
                left = left[1:]
                right = right[1:-1]
                elementaryCondition = ElementaryCondition(
                                                  
                    attribute, float(left), float(right), leftClosed, rightClosed, column_index=self.column_indexes[attribute]
                )
            else:
                value = valueset[1:-1]
                elementaryCondition = ElementaryCondition(attribute, str(value), column_index=self.column_indexes[attribute])
                                          

            compoundCondition.add_subcondition(elementaryCondition)

        consequence_att, consequence_value = consequence.split(" = ")
        consequence_value = consequence_value[1:-1]
        consequence = ElementaryCondition(consequence_att, consequence_value, column_index=self.column_indexes[attribute])

        return self._preprocessRule(Rule(compoundCondition, consequence))

    def _preprocessRule(self, rule):
        conditions_for_attributes = {}

        for condition in rule.premise.subconditions:
            attr = condition.attribute

            if attr in conditions_for_attributes.keys():
                old_condition = conditions_for_attributes[attr]
                condition = old_condition.get_intersection(condition)
            conditions_for_attributes[attr] = condition

        premise = CompoundCondition()
        premise.add_subconditions(conditions_for_attributes.values())
        rule.premise = premise

        return rule

    def _get_rules_from_DT(self, tree, feature_names, class_names, label_name):
        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != self._tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]
        paths = []
        path = []

        def recurse(node, path, paths):
            if tree_.feature[node] != self._tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                p1, p2 = list(path), list(path)
                p1 += [f"{name} <= {threshold}"]
                recurse(tree_.children_left[node], p1, paths)
                p2 += [f"{name} > {threshold}"]
                recurse(tree_.children_right[node], p2, paths)
            else:
                path += [(tree_.value[node], tree_.n_node_samples[node])]
                paths += [path]

        recurse(0, path, paths)

        # sort by samples count
        samples_count = [p[-1][1] for p in paths]
        ii = list(np.argsort(samples_count))
        paths = [paths[i] for i in reversed(ii)]

        rules = []
        for path in paths:
            rule = "IF "
            for p in path[:-1]:
                if rule != "IF ":
                    rule += " AND "
                rule += str(p)
            rule += " THEN "
            if class_names is None:
                rule += (
                                        
                    f"{label_name} = " + "{" + str(path[-1][0][0][0]) + "}"
                )
            else:
                classes = path[-1][0][0]
                l = np.argmax(classes)
                rule += f"{label_name} = " + "{" + f"{class_names[l]}" + "}"
            rules += [rule]

        return rules


class ClassificationModel(Model):
    def __init__(
        self, model, feature_names=None, class_names=None, label_name=None
    ) -> None:
        self.column_indexes: Dict[str, int] = {column_name:i for i, column_name in enumerate(list(feature_names))}
        super().__init__(
            model,
            feature_names=feature_names,
            class_names=class_names,
            label_name=label_name,
        )

    def get_rules(self, model, feature_names=None, class_names=None, label_name=None):

        if isinstance(model, BaseOperator):
            return self._map_rules_from_RuleKit(model.model.rules)
        if isinstance(model, list):
            return self._map_rules_from_list(model)
        sklearn_tree_module = importlib.import_module("sklearn.tree")
        BaseDecisionTree = getattr(sklearn_tree_module, "BaseDecisionTree")
        if isinstance(model, BaseDecisionTree):
            self._tree = getattr(sklearn_tree_module, "_tree")
            rules = self._get_rules_from_DT(
                self.model, feature_names, class_names, label_name
            )
            return self._map_rules_from_sklearn(rules)
        orange_module = importlib.import_module("Orange.classification.rules")
        _RuleClassifier = getattr(orange_module, "_RuleClassifier")
        if isinstance(model, _RuleClassifier):
            return self._map_rules_from_Orange(model.rule_list)

    def _map_rule_from_RuleKit(self, rule):
        rule = rule[3:]
        premise, consequence = rule.split(" THEN ")
        conditions = premise.split(" AND ")

        compoundCondition = CompoundCondition()

        for condition in conditions:
            attribute, valueset = condition.split(" = ")
            if "," in valueset:
                left, right = valueset.split(",")
                leftClosed = False if left[0] == "(" else True
                rightClosed = False if right[-1:] == ")" else True
                left = left[1:]
                right = right[1:-1]
                elementaryCondition = ElementaryCondition(
                                                  
                    attribute, float(left), float(right), leftClosed, rightClosed, column_index=self.column_indexes[attribute]
                )
            else:
                value = valueset[1:-1]
                elementaryCondition = ElementaryCondition(attribute, str(value), column_index=self.column_indexes[attribute])
                                          

            compoundCondition.add_subcondition(elementaryCondition)

        consequence_att, consequence_value = consequence.split(" = ")
        consequence_value = consequence_value[1:-1]
        consequence = ElementaryCondition(consequence_att, consequence_value, column_index=self.column_indexes[attribute])

        return Rule(compoundCondition, consequence)

    def _map_rules_from_Orange(self, rules):
        preprocessed_Rules = []
        for rule in rules:
            if "TRUE" not in rule.__str__():
                                          
                preprocessed_Rules.append(self._map_rule_from_Orange(rule.__str__()))
        return preprocessed_Rules

    def _map_rule_from_Orange(self, rule):
        rule = rule[3:]
        premise, consequence = rule.split(" THEN ")
        conditions = premise.split(" AND ")
        compoundCondition = CompoundCondition()

        for condition in conditions:
            if "<=" in condition:
                attribute, value = condition.split("<=")
                elementaryCondition = ElementaryCondition(
                    attribute, ElementaryCondition.minus_inf, float(value), False, True, column_index=self.column_indexes[attribute]
                                           
                )
            elif "<" in condition:
                attribute, value = condition.split("<")
                elementaryCondition = ElementaryCondition(
                    attribute, ElementaryCondition.minus_inf, float(value), False, False, column_index=self.column_indexes[attribute]
                                            
                )
            elif ">=" in condition:
                attribute, value = condition.split(">=")
                elementaryCondition = ElementaryCondition(
                                     
                    attribute, float(value), ElementaryCondition.inf, True, False, column_index=self.column_indexes[attribute]
                )
            elif ">" in condition:
                attribute, value = condition.split(">")
                elementaryCondition = ElementaryCondition(
                                     
                    attribute, float(value), ElementaryCondition.inf, False, False, column_index=self.column_indexes[attribute]
                )
            elif "=" in condition:
                attribute, value = condition.split("=")
                elementaryCondition = ElementaryCondition(attribute, str(value), column_index=self.column_indexes[attribute])
                                          
            else:
                elementaryCondition = ElementaryCondition("all", "TRUE")

            compoundCondition.add_subcondition(elementaryCondition)

        consequence_att, consequence_value = consequence.split("=")
        consequence_value = consequence_value[:-1]
        consequence = ElementaryCondition(consequence_att, consequence_value, column_index=self.column_indexes[attribute])

        return self._preprocessRule(Rule(compoundCondition, consequence))


class RegressionModel(Model):
    def __init__(self, model, feature_names=None, label_name=None) -> None:
        self.column_indexes: Dict[str, int] = {column_name:i for i, column_name in enumerate(list(feature_names))}
        super().__init__(
            model, feature_names=feature_names, class_names=None, label_name=label_name
        )

    def get_rules(self, model, feature_names=None, class_names=None, label_name=None):
        if isinstance(model, BaseOperator):
            return self._map_rules_from_RuleKit(model.model.rules)
        if isinstance(model, list):
            return self._map_rules_from_list(model)
        sklearn_tree_module = importlib.import_module("sklearn.tree")
        BaseDecisionTree = getattr(sklearn_tree_module, "BaseDecisionTree")
        if isinstance(model, BaseDecisionTree):
            self._tree = getattr(sklearn_tree_module, "_tree")
            rules = self._get_rules_from_DT(
                self.model, feature_names, class_names, label_name
            )
            return self._map_rules_from_sklearn(rules)


    def _map_rule_from_RuleKit(self, rule):
        rule = rule[3:]
        premise, consequence = rule.split(" THEN ")
        conditions = premise.split(" AND ")

        compoundCondition = CompoundCondition()
                                                        
        for condition in conditions:
            attribute, valueset = condition.split(" = ")
            if "," in valueset:
                left, right = valueset.split(",")
                leftClosed = False if left[0] == "(" else True
                rightClosed = False if right[-1:] == ")" else True
                left = left[1:]
                right = right[1:-1]
                elementaryCondition = ElementaryCondition(
                                                  
                    attribute, float(left), float(right), leftClosed, rightClosed, column_index=self.column_indexes[attribute]
                )
            else:
                value = valueset[1:-1]
                elementaryCondition = ElementaryCondition(attribute, str(value), column_index=self.column_indexes[attribute])
                                          

            compoundCondition.add_subcondition(elementaryCondition)

        consequence_att, consequence_value = consequence.split(" = ")
        consequence_val, consequence_range = consequence_value.split(" ")
        consequence_val = consequence_val[1:-1]
                                          
        consequence = ElementaryCondition(consequence_att, float(consequence_val), column_index=self.column_indexes[attribute])

        return Rule(compoundCondition, consequence)

    def _map_rule_from_sklearn(self, rule):
        rule = rule[3:]
        premise, consequence = rule.split(" THEN ")
        conditions = premise.split(" AND ")
        compoundCondition = CompoundCondition()

        for condition in conditions:
            if "<=" in condition:
                attribute, value = condition.split(" <= ")
                elementaryCondition = ElementaryCondition(
                    attribute, ElementaryCondition.minus_inf, float(value), False, True, column_index=self.column_indexes[attribute]
                                           
                )
            elif "<" in condition:
                attribute, value = condition.split(" < ")
                elementaryCondition = ElementaryCondition(
                    attribute, ElementaryCondition.minus_inf, float(value), False, False, column_index=self.column_indexes[attribute]
                                            
                )
            elif ">=" in condition:
                attribute, value = condition.split(" >= ")
                elementaryCondition = ElementaryCondition(
                                     
                    attribute, float(value), ElementaryCondition.inf, True, False, column_index=self.column_indexes[attribute]
                )
            elif ">" in condition:
                attribute, value = condition.split(" > ")
                elementaryCondition = ElementaryCondition(
                                     
                    attribute, float(value), ElementaryCondition.inf, False, False, column_index=self.column_indexes[attribute]
                )
            else:
                attribute, value = condition.split(" = ")
                elementaryCondition = ElementaryCondition(attribute, str(value), column_index=self.column_indexes[attribute])
                                          

            compoundCondition.add_subcondition(elementaryCondition)

        consequence_att, consequence_value = consequence.split(" = ")
        consequence_value = consequence_value[1:-1]
                                          
        consequence = ElementaryCondition(consequence_att, float(consequence_value), column_index=self.column_indexes[attribute])

        return self._preprocessRule(Rule(compoundCondition, consequence))

    def _map_string_rule(self, rule):
        rule = rule[3:]
        premise, consequence = rule.split(" THEN ")
        conditions = premise.split(" AND ")
        compoundCondition = CompoundCondition()

        for condition in conditions:
            attribute, valueset = condition.split(" = ")
            if "," in valueset:
                left, right = valueset.split(",")
                leftClosed = False if left[0] == "(" else True
                rightClosed = False if right[-1:] == ")" else True
                left = left[1:]
                right = right[1:-1]
                elementaryCondition = ElementaryCondition(
                                                  
                    attribute, float(left), float(right), leftClosed, rightClosed, column_index=self.column_indexes[attribute]
                )
            else:
                value = valueset[1:-1]
                elementaryCondition = ElementaryCondition(attribute, str(value), column_index=self.column_indexes[attribute])
                                          

            compoundCondition.add_subcondition(elementaryCondition)

        consequence_att, consequence_value = consequence.split(" = ")
        consequence_value = consequence_value[1:-1]
                                          
        consequence = ElementaryCondition(consequence_att, float(consequence_value), column_index=self.column_indexes[attribute])

        return self._preprocessRule(Rule(compoundCondition, consequence))


class SurvivalModel(Model):
    def __init__(self, model, feature_names=None, survival_status_name=None) -> None:
        self.column_indexes: Dict[str, int] = {column_name:i for i, column_name in enumerate(list(feature_names))}
        super().__init__(
            model,
            feature_names=feature_names,
            class_names=None,
            label_name=survival_status_name,
        )

    def get_rules(self, model, feature_names=None, class_names=None, label_name=None):
        if isinstance(model, BaseOperator):
            return self._map_rules_from_RuleKit(model.model.rules)
        if isinstance(model, list):
            return self._map_rules_from_list(model)
        sksurv_tree_module = importlib.import_module("sksurv.tree")
        SurvivalTree = getattr(sksurv_tree_module, "SurvivalTree")
        if isinstance(model, SurvivalTree):
            sklearn_tree_module = importlib.import_module("sklearn.tree")
            self._tree = getattr(sklearn_tree_module, "_tree")
            rules = self._get_rules_from_DT(
                self.model, feature_names, class_names, label_name
            )
            return self._map_rules_from_sklearn(rules)


    def _map_rule_from_RuleKit(self, rule):
        rule = rule[3:]
        premise, consequence = rule.split(" THEN ")
        conditions = premise.split(" AND ")
        compoundCondition = CompoundCondition()

        for condition in conditions:
            attribute, valueset = condition.split(" = ")
            if "," in valueset:
                left, right = valueset.split(",")
                leftClosed = False if left[0] == "(" else True
                rightClosed = False if right[-1:] == ")" else True
                left = left[1:]
                right = right[1:-1]
                elementaryCondition = ElementaryCondition(
                                                  
                    attribute, float(left), float(right), leftClosed, rightClosed, column_index=self.column_indexes[attribute]
                )
            else:
                value = valueset[1:-1]
                elementaryCondition = ElementaryCondition(attribute, str(value), column_index=self.column_indexes[attribute])
                                          

            compoundCondition.add_subcondition(elementaryCondition)

        consequence_att, consequence_value = consequence.split(" = ")
        consequence_val = consequence_value[1:-1]
        consequence = ElementaryCondition(consequence_att, consequence_val, column_index=self.column_indexes[attribute])

        return Rule(compoundCondition, consequence)


class BlackBoxModel:
    def __init__(self, X_model, model_predictions, type) -> None:
        RuleKit.init()

        if type == "regression":
            rulekit_model = RuleRegressor(
            induction_measure=Measures.C2,
            pruning_measure=Measures.C2,
            voting_measure=Measures.C2,
        )
        else:
            rulekit_model = RuleClassifier(
            induction_measure=Measures.C2,
            pruning_measure=Measures.C2,
            voting_measure=Measures.C2,
        )
        self.type = type
        self.y = model_predictions
        self.rulekit_model = rulekit_model


    def get_rules_model(self, X_org):
        
        if isinstance(self.y, pd.DataFrame):
            label_name = self.y.columns[0]
            y = self.y.to_numpy().reshape(self.y.size)
        else:
            label_name = self.y.name
            y=self.y

        for column in X_org.select_dtypes('object').columns.tolist():
            X_org[column] = X_org[column].replace({np.nan: None})
        self.rulekit_model.fit(X_org,y)

        if type == "regression":
            model = RegressionModel(self.rulekit_model,label_name)
        else:          
            model = ClassificationModel(self.rulekit_model, X_org.columns, np.unique(self.y), label_name)

        return model






        



