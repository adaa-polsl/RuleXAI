from .rule import (
    CompoundCondition,
    CompoundConditionWithCombiningOperators,
    ClassificationRule,
    ElementaryCondition,
    RegressionRule,
    Rule,
    SurvivalRule,
)
import pandas as pd
import numpy as np
from typing import List
from operator import attrgetter
import math
from typing import Dict
import importlib

class ConditionImportance:
    def __init__(self, rules, data, labels, if_split, measure = None) -> None:
        self.data = data
        self.labels = labels
        self.dataset = pd.concat([self.data, self.labels], axis=1)
        self.if_split = if_split
        self.measure = measure
        self.column_indexes: Dict[str, int] = {column_name:i for i, column_name in enumerate(list(data.columns))}
        if if_split:
            self.rules = self.split_conditions_in_rules(rules)
        else:
            self.rules = rules

    def _get_conditions_with_rules(self, rules):
        conditions_with_rules = dict()

        for rule in rules:
            rule_conditions = rule.premise.get_subconditions()

            for condition in rule_conditions:
                if condition in conditions_with_rules.keys():
                    conditions_rules = conditions_with_rules[condition]
                else:
                    conditions_rules = []
                conditions_rules.append(rule)
                conditions_with_rules[condition] = conditions_rules

        return conditions_with_rules

    def _calculate_conditions_qualities(self, conditions_with_rules):
        conditions_qualities = []
        for condition in conditions_with_rules.keys():
            sum = 0
            for rule in conditions_with_rules[condition]:
                sum += self._calculate_index_simplified(condition, rule)
            conditions_qualities.append(ConditionEvaluation(condition, sum))

        return conditions_qualities


    def _calculateMeasure(self, rule):
        p, n, P, N = rule.covers(self.dataset)

        if self.measure == "Correlation":
            if (P - p + N - n == 0):
                return  0
            else: 
                return (p * N - P * n) / math.sqrt(P * N * (p + n) * (P - p + N - n))

        elif self.measure == "Lift":
            if (p == 0 and n == 0) or P == 0:
                return 0
            else:
                return p * (P + N) / ((p + n) * P)

        else: # C2
            if (p == 0 and n == 0) or P == 0 or N == 0:
                return 0
            else:
                return (((P + N) * p / (p + n) - P) / N) * ((1 + p / P) / 2)  # C2

    def _condition_importances_to_DataFrame(self, condition_importances):
        importances_df = pd.DataFrame()
        importances_df["conditions"] = pd.Series(
            [str(cnd) for cnd in condition_importances.keys()]
        )
        importances_df["importances"] = pd.Series(condition_importances.values())

        return importances_df

    def split_conditions_in_rules(self, rules):
        conditions_with_rules = self._get_conditions_with_rules(rules)
        conditions = self._split_conditions_into_basic(conditions_with_rules.keys())

        return self._get_rules_with_splitted_conditions(conditions, rules)

    def _split_conditions_into_basic(self, conditions: List[ElementaryCondition]):
        splittedConditions = []
        conditions_for_attributes = dict()
        for condition in conditions:
            if condition.right is None: #attribute is Nominal
                splittedConditions.append(condition)
            else:
                attr = condition.attribute
                if attr in conditions_for_attributes:
                    attribute_conditions = conditions_for_attributes[attr]
                else: 
                    attribute_conditions = []
                attribute_conditions.append(condition)
                conditions_for_attributes[attr] = attribute_conditions

        for conditions_for_attribute in conditions_for_attributes.values():
            if len(conditions_for_attribute) > 1:
                basic_conditions = self._get_basic_conditions_for_attribute(conditions_for_attribute)
                splittedConditions.extend(basic_conditions)
            else:
                splittedConditions.extend(conditions_for_attribute)

        return splittedConditions
    
    def _get_basic_conditions_for_attribute(self, conditions: List[ElementaryCondition]):
        basic_conditions = []
        points = []
        id = 0 
        attribute = conditions[0].attribute
        for condition in conditions:
            point = Point(id, condition.left, "Left", condition.leftClosed)
            points.append(point)
            point = Point(id, condition.right, "Right", condition.rightClosed)
            points.append(point)
            id += 1

        min_point_first = min(points, key = attrgetter('value'))
        all_points_len = len(points)
        points = [point for point in points if point.value != min_point_first.value]
        number_of_firts_mins = all_points_len - len(points)

        while(len(points) > 0):
            min_point_second = min(points, key = attrgetter('value'))
            points_with_second_min_len = len(points)
            points = [point for point in points if point.value != min_point_second.value]
            number_of_second_mins = points_with_second_min_len - len(points)

            if(min_point_first.value == float('-inf')):
                leftClosed = False
            else:
                leftClosed = True

            condition = ElementaryCondition(attribute = attribute, left = min_point_first.value, right= min_point_second.value, leftClosed = leftClosed, rightClosed= False, column_index=self.column_indexes[attribute])
            basic_conditions.append(condition)

            if (min_point_first.condition_id == min_point_second.condition_id) and number_of_firts_mins == 1 and number_of_second_mins == 1 and len(points) > 1:
                min_point_first = min(points, key = attrgetter('value'))
                points_with_first_min_len = len(points)
                points = [point for point in points if point.value != min_point_first.value]
                number_of_firts_mins = points_with_first_min_len - len(points)
            else:
                min_point_first = min_point_second
                min_point_first.is_closed = not min_point_second.is_closed

        return basic_conditions

    
    def _get_rules_with_splitted_conditions(self, basic_conditions, rules: List[ClassificationRule]):
        rules_with_basic_conditions = []
        for rule in rules:
            compoundCondition = self._create_compound_condition_for_rule(basic_conditions, rule)
            rules_with_basic_conditions.append(Rule(compoundCondition, rule.consequence))
        return rules_with_basic_conditions

    def _create_compound_condition_for_rule(self,basic_conditions, rule: Rule):
        subconditions = rule.premise.get_subconditions()
        compoundCondition = CompoundConditionWithCombiningOperators()
        for condition_basic in basic_conditions:
            i = 0
            condition_added = False
            while(i<len(subconditions) and not condition_added):
                condition_from_rule = subconditions[i]
                if self._check_if_condition_contains_basic_condition(condition_from_rule, condition_basic):
                    compoundCondition.add_subcondition(condition_basic)
                    condition_added = True
                i += 1
        return compoundCondition

    def _check_if_condition_contains_basic_condition(self,condition_from_rule: ElementaryCondition, basic_condition: ElementaryCondition):
        if not condition_from_rule.attribute == basic_condition.attribute:
            return False
        elif condition_from_rule.right == None: #nominal attribute
            if condition_from_rule.left == basic_condition.left:
                return True
            else:
                return False
        else:
            if ((condition_from_rule.left <= basic_condition.left) and (condition_from_rule.right >= basic_condition.right)):
                return True
            else:
                return False
            

class ClassificationConditionImportance(ConditionImportance):
    def __init__(self, rules, data, labels, if_split, measure) -> None:
        self.column_indexes: Dict[str, int] = {column_name:i for i, column_name in enumerate(list(data.columns))}
        super().__init__(rules, data, labels, if_split, measure)

    def condition_importances(self):
        rules_by_class = self._split_rules_by_decision_class(self.rules)

        condition_importances_for_classes = dict()

        for class_name in rules_by_class.keys():
            class_rules = rules_by_class[class_name]
            conditions_with_rules = self._get_conditions_with_rules(class_rules)
            conditions_qualities = self._calculate_conditions_qualities(
                conditions_with_rules
            )
            condition_importances_for_classes[class_name] = conditions_qualities

        conditions_importances = self._calculate_conditions_importances(
            condition_importances_for_classes
        )
        return self._condition_importances_to_DataFrame(conditions_importances)

    def _split_rules_by_decision_class(self, rules):
        rules_by_class = dict()

        for rule in rules:
            class_name = rule.consequence.left
            if class_name in rules_by_class.keys():
                class_rules = rules_by_class[class_name]
            else:
                class_rules = []

            class_rules.append(rule)
            rules_by_class[class_name] = class_rules

        return rules_by_class

    def _calculate_index_simplified(self, condition, rule):
        rule = ClassificationRule(rule.premise, rule.consequence)

        rule_conditions = []
        rule_conditions.extend(rule.premise.get_subconditions())
        number_of_conditions = len(rule_conditions)
        rule_conditions.remove(condition)

        if self.if_split:
            premise_without_evaluated_condition = CompoundConditionWithCombiningOperators()
        else:
            premise_without_evaluated_condition = CompoundCondition()
        
        premise_without_evaluated_condition.add_subconditions(rule_conditions)
        rule_without_evaluated_condition = ClassificationRule(
            premise_without_evaluated_condition, rule.consequence
        )

        factor = 1.0 / number_of_conditions

        if len(rule_conditions) == 0:
            return factor * (
                self._calculateMeasure(rule)
                - self._calculateMeasure(rule_without_evaluated_condition)
            )
        else:
            premise_with_only_evaluated_condition = CompoundCondition()
            premise_with_only_evaluated_condition.add_subcondition(condition)
            rule_with_only_evaluated_condition = ClassificationRule(
                premise_with_only_evaluated_condition, rule.consequence
            )

            return factor * (
                self._calculateMeasure(rule)
                - self._calculateMeasure(rule_without_evaluated_condition)
                + self._calculateMeasure(rule_with_only_evaluated_condition)
            )

    def _calculate_conditions_importances(self, condition_qualities_for_classes):
        conditions_importances = dict()

        for evaluated_class in condition_qualities_for_classes.keys():

            conditions_importances_for_class = dict()
            conditions_for_evaluated_class = condition_qualities_for_classes[
                evaluated_class
            ]

            for condition in conditions_for_evaluated_class:
                sum = condition.quality
                """ 
                for class_name in condition_qualities_for_classes.keys():
                    if class_name != evaluated_class:
                        conditions_from_other_class = condition_qualities_for_classes[
                            class_name
                        ]
                        evaluated_condition_from_other_class = list(
                            filter(
                                lambda cnd: cnd.condition == condition.condition,
                                conditions_from_other_class,
                            )
                        )
                        if len(evaluated_condition_from_other_class) > 0:
                            sum -= evaluated_condition_from_other_class[0].quality
                """
                conditions_importances_for_class[condition.condition] = sum

            conditions_importances[evaluated_class] = dict(
                sorted(
                    conditions_importances_for_class.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            )

        return conditions_importances

    def _condition_importances_to_DataFrame(self, condition_importances):
        importances_df = pd.DataFrame()
        for class_name in condition_importances.keys():
            importances_df_tmp = pd.DataFrame()
            importances_df_tmp[class_name + " | conditions_names"] = pd.Series(
                [str(cnd) for cnd in condition_importances[class_name].keys()]
            )
            importances_df_tmp[class_name + " | importances"] = pd.Series(
                condition_importances[class_name].values()
            )
            importances_df = pd.concat(
                [importances_df, importances_df_tmp], ignore_index=False, axis=1
            )
        return importances_df.replace(np.nan, "-")

    def _get_rules_with_splitted_conditions(self, basic_conditions, rules: List[ClassificationRule]):
        rules_with_basic_conditions = []
        for rule in rules:
            compoundCondition = self._create_compound_condition_for_rule(basic_conditions, rule)
            rules_with_basic_conditions.append(ClassificationRule(compoundCondition, rule.consequence))
        return rules_with_basic_conditions



class RegressionConditionImportance(ConditionImportance):
    def __init__(self, rules, data, labels, if_split, measure) -> None:
        self.column_indexes: Dict[str, int] = {column_name:i for i, column_name in enumerate(list(data.columns))}
        super().__init__(rules, data, labels, if_split, measure)

    def condition_importances(self):
        conditions_with_rules = self._get_conditions_with_rules(self.rules)
        conditions_qualities = self._calculate_conditions_qualities(
            conditions_with_rules
        )
        conditions_importances = dict()

        for condition_evaluation in conditions_qualities:
            conditions_importances[
                condition_evaluation.condition
            ] = condition_evaluation.quality

        conditions_importances = dict(
            sorted(
                conditions_importances.items(), key=lambda item: item[1], reverse=True
            )
        )

        return self._condition_importances_to_DataFrame(conditions_importances)

    def _calculate_index_simplified(self, condition, rule):
        rule = RegressionRule(rule.premise, rule.consequence)

        rule_conditions = []
        rule_conditions.extend(rule.premise.get_subconditions())
        number_of_conditions = len(rule_conditions)
        rule_conditions.remove(condition)

        if self.if_split:
            premise_without_evaluated_condition = CompoundConditionWithCombiningOperators()
        else:
            premise_without_evaluated_condition = CompoundCondition()
        premise_without_evaluated_condition.add_subconditions(rule_conditions)
        rule_without_evaluated_condition = RegressionRule(
            premise_without_evaluated_condition, rule.consequence
        )

        factor = 1.0 / number_of_conditions

        if len(rule_conditions) == 0:
            return factor * (
                self._calculateMeasure(rule)
                - self._calculateMeasure(rule_without_evaluated_condition)
            )
        else:
            premise_with_only_evaluated_condition = CompoundCondition()
            premise_with_only_evaluated_condition.add_subcondition(condition)
            rule_with_only_evaluated_condition = RegressionRule(
                premise_with_only_evaluated_condition, rule.consequence
            )

            return factor * (
                self._calculateMeasure(rule)
                - self._calculateMeasure(rule_without_evaluated_condition)
                + self._calculateMeasure(rule_with_only_evaluated_condition)
            )

    def _get_rules_with_splitted_conditions(self, basic_conditions, rules: List[RegressionRule]):
        rules_with_basic_conditions = []
        for rule in rules:
            compoundCondition = self._create_compound_condition_for_rule(basic_conditions, rule)
            rules_with_basic_conditions.append(RegressionRule(compoundCondition, rule.consequence))
        return rules_with_basic_conditions

class SurvivalConditionImportance(ConditionImportance):
    def __init__(self, rules, data, labels, if_split, measure) -> None:
        self.column_indexes: Dict[str, int] = {column_name:i for i, column_name in enumerate(list(data.columns))}
        super().__init__(rules, data, labels, if_split, measure)
        lifelines = importlib.import_module("lifelines.statistics")
        self.logrank_test = getattr(lifelines, "logrank_test")
        

    def condition_importances(self):
        conditions_with_rules = self._get_conditions_with_rules(self.rules)
        conditions_qualities = self._calculate_conditions_qualities(
            conditions_with_rules
        )
        conditions_importances = dict()

        for condition_evaluation in conditions_qualities:
            conditions_importances[
                condition_evaluation.condition
            ] = condition_evaluation.quality

        conditions_importances = dict(
            sorted(
                conditions_importances.items(), key=lambda item: item[1], reverse=True
            )
        )

        return self._condition_importances_to_DataFrame(conditions_importances)

    def _calculate_index_simplified(self, condition, rule):
        rule = SurvivalRule(rule.premise, rule.consequence)

        rule_conditions = []
        rule_conditions.extend(rule.premise.get_subconditions())
        number_of_conditions = len(rule_conditions)
        rule_conditions.remove(condition)

        if self.if_split:
            premise_without_evaluated_condition = CompoundConditionWithCombiningOperators()
        else:
            premise_without_evaluated_condition = CompoundCondition()

        premise_without_evaluated_condition.add_subconditions(rule_conditions)
        rule_without_evaluated_condition = SurvivalRule(
            premise_without_evaluated_condition, rule.consequence
        )

        factor = 1.0 / number_of_conditions

        if len(rule_conditions) == 0:
            return factor * (
                self._calculateMeasure(rule)
                - self._calculateMeasure(rule_without_evaluated_condition)
            )
        else:
            premise_with_only_evaluated_condition = CompoundCondition()
            premise_with_only_evaluated_condition.add_subcondition(condition)
            rule_with_only_evaluated_condition = SurvivalRule(
                premise_with_only_evaluated_condition, rule.consequence
            )

            return factor * (
                self._calculateMeasure(rule)
                - self._calculateMeasure(rule_without_evaluated_condition)
                + self._calculateMeasure(rule_with_only_evaluated_condition)
            )

    def _calculateMeasure(self, rule):
        p, n, P, N, covered_indices = rule.covers(self.dataset, return_positives=True
        )
        # covered_indices -> in survival rules all examples are classified as positives
        uncovered_indices = [
            id for id in range(self.data.shape[0]) if id not in covered_indices
        ]

        results = self.logrank_test(
            self.data["survival_time"][
                self.data["survival_time"].index.isin(covered_indices)
            ],
            self.data["survival_time"][
                self.data["survival_time"].index.isin(uncovered_indices)
            ],
            event_observed_A=self.labels[self.labels.index.isin(covered_indices)],
            event_observed_B=self.labels[self.labels.index.isin(uncovered_indices)],
        )

        return results.test_statistic


    def _get_rules_with_splitted_conditions(self, basic_conditions, rules: List[SurvivalRule]):
        rules_with_basic_conditions = []
        for rule in rules:
            compoundCondition = self._create_compound_condition_for_rule(basic_conditions, rule)
            rules_with_basic_conditions.append(SurvivalRule(compoundCondition, rule.consequence))
        return rules_with_basic_conditions



class ConditionEvaluation:
    def __init__(self, condition, quality) -> None:
        self.condition = condition
        self.quality = quality


class Point:
    def __init__(self, condition_id: int, value: float, side: str, is_closed: bool) -> None:
        self.condition_id = condition_id
        self.value = value
        self.side = side
        self.is_closed = is_closed