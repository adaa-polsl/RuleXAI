import enum
import numpy as np
from typing import List
np.warnings.filterwarnings('ignore')

# creating enumerations using class
class LogicalOperator(enum.Enum):
    conjuction = 1
    alternative = 2


class ElementaryCondition:

    inf = float("inf")
    minus_inf = float("-inf")

    def __init__(
        self,
        attribute: str,
        left: float,
        right: float = None,
        leftClosed: bool = None,
        rightClosed: bool = None,
        column_index: int = None,
    ) -> None:
        self.attribute = attribute
        self.left = left
        self.right = right
        self.leftClosed = leftClosed
        # if right is None its means that attribute is Nominal
        self.rightClosed = rightClosed
        self._column_index = column_index

    def covered_mask(self, X: np.ndarray) -> np.ndarray:
        if self.right is not None:
            left_part: np.ndarray = (X[:, self._column_index] >= self.left) if self.leftClosed else (X[:, self._column_index] > self.left)
            right_part: np.ndarray = (X[:, self._column_index] <= self.right) if self.rightClosed else (X[:, self._column_index] < self.right)
            return left_part & right_part
        else:
            return (X[:, self._column_index] == self.left)

    def uncovered_mask(self, X: np.ndarray) -> np.ndarray:
        return np.logical_not(self.covered_mask(X))

    def negative_covered_mask(self, X: np.ndarray, y: np.ndarray, decision) -> np.ndarray:
        return self.covered_mask(X) & (y[:] != decision)

    def positive_covered_mask(self, X: np.ndarray, y: np.ndarray, decision) -> np.ndarray:
        a =  self.covered_mask(X) & (y[:] == decision)
        return a
    def get_intersection(self, other_condition):
        return ElementaryCondition(
            self.attribute,
            np.max([self.left, other_condition.left]),
            np.min([self.right, other_condition.right]),
            other_condition.leftClosed
            if self.left < other_condition.left
            else self.leftClosed,
            other_condition.rightClosed
            if self.right > other_condition.right
            else self.rightClosed,
            column_index=self._column_index
        )

    def evaluate(self, ex):
        value = ex[self.attribute]
        if self.right == None:
            return str(value) == self.left
        else:
            return ((value >= self.left and self.leftClosed) or value > self.left) and (
                (value <= self.right and self.rightClosed) or value < self.right
            )
    def evaluate_mask(self, X_t: np.ndarray, X: np.ndarray, column_index: int = None):
        X_t[:, column_index] = self.covered_mask(X)
        return X_t

    def __str__(self):
        if self.right == None:
            s = "".join(["{", str(self.left), "}"])
        else:
            s = "".join(
                [
                    ("<" if self.leftClosed else "("),
                    ("-inf" if (self.left == self.minus_inf) else str(self.left)),
                    ", ",
                    ("inf" if (self.right == self.inf) else str(self.right)),
                    (">" if self.rightClosed else ")"),
                ]
            )
        return "".join([self.attribute, " = ", s])

    def __eq__(self, other):
        if not isinstance(other, ElementaryCondition):
            return False
        return (
            (self.attribute == other.attribute)
            and (self.left == other.left)
            and (self.right == other.right)
            and (self.leftClosed == other.leftClosed)
            and (self.rightClosed == other.rightClosed)
        )

    def __hash__(self):
        return hash(
            (self.attribute, self.left, self.right,
             self.leftClosed, self.rightClosed)
        )


class CompoundCondition:

    def __init__(self) -> None:
        self.operator = LogicalOperator.conjuction
        self.subconditions = []

    def covered_mask(self, X: np.ndarray) -> np.ndarray:
        if len(self.subconditions) == 0:
            return np.full(X.shape[0], fill_value=False)
        covered_mask = (self.subconditions[0].covered_mask(X))
        if self.operator == LogicalOperator.conjuction:
            for i in range(1, len(self.subconditions)):
                covered_mask &= self.subconditions[i].covered_mask(X)
        elif self.operator == LogicalOperator.alternative:
            for i in range(1, len(self.subconditions)):
                covered_mask |= self.subconditions[i].covered_mask(X)
        return covered_mask

    def uncovered_mask(self, X: np.ndarray) -> np.ndarray:
        if len(self.subconditions) == 0:
            return np.full(X.shape[0], fill_value=True)
        uncovered_mask = (self.subconditions[0].uncovered_mask(X))
        if self.operator == LogicalOperator.conjuction:
            for i in range(1, len(self.subconditions)):
                uncovered_mask &= self.subconditions[i].uncovered_mask(X)
        elif self.operator == LogicalOperator.alternative:
            for i in range(1, len(self.subconditions)):
                uncovered_mask |= self.subconditions[i].uncovered_mask(X)
        return uncovered_mask

    def positive_covered_mask(self, X: np.ndarray, y: np.ndarray, decision) -> np.ndarray:
        if len(self.subconditions) == 0:
            return np.full(X.shape[0], fill_value=False)
        positive_covered_mask = (
            self.subconditions[0].positive_covered_mask(X, y, decision))
        if self.operator == LogicalOperator.conjuction:
            for i in range(1, len(self.subconditions)):
                positive_covered_mask &= self.subconditions[i].positive_covered_mask(
                    X, y, decision)
        elif self.operator == LogicalOperator.alternative:
            for i in range(1, len(self.subconditions)):
                positive_covered_mask |= self.subconditions[i].positive_covered_mask(
                    X, y, decision)
        return positive_covered_mask

    def negative_covered_mask(self, X: np.ndarray, y: np.ndarray, decision) -> np.ndarray:
        if len(self.subconditions) == 0:
            return np.full(X.shape[0], fill_value=False)
        negative_covered_mask = (
            self.subconditions[0].negative_covered_mask(X, y, decision))
        if self.operator == LogicalOperator.conjuction:
            for i in range(1, len(self.subconditions)):
                negative_covered_mask &= self.subconditions[i].negative_covered_mask(
                    X, y, decision)
        elif self.operator == LogicalOperator.alternative:
            for i in range(1, len(self.subconditions)):
                negative_covered_mask |= self.subconditions[i].negative_covered_mask(
                    X, y, decision)
        return negative_covered_mask

    def add_subcondition(self, cnd: ElementaryCondition):
        self.subconditions.append(cnd)

    def add_subconditions(self, cnds: List[ElementaryCondition]):
        self.subconditions.extend(cnds)

    def get_subconditions(self):
        return self.subconditions

    def set_logical_operator(self, operator: LogicalOperator):
        self.operator = operator

    def __str__(self):
        s = ""
        operator = " AND " if (
            self.operator == LogicalOperator.conjuction) else " OR "

        for i in range(len(self.subconditions)):
            s += self.subconditions[i].__str__()
            if i != (len(self.subconditions) - 1):
                s += operator
        return s

    def evaluate(self, ex):
        for condition in self.subconditions:
            partial = condition.evaluate(ex)
            if self.operator == LogicalOperator.conjuction and partial == False:
                return False
            elif self.operator == LogicalOperator.alternative and partial == True:
                return True
        return True if (self.operator == LogicalOperator.conjuction) else False


class CompoundConditionWithCombiningOperators(CompoundCondition):

    def __init__(self) -> None:
        self.operator = LogicalOperator.conjuction
        self.subCompoundConditions = dict()
    
    def covered_mask(self, X: np.ndarray) -> np.ndarray:
        subCompoundConditions_keys = list(self.subCompoundConditions.keys())
        if len(self.subCompoundConditions) == 0:
            return np.full(X.shape[0], fill_value=False)
        covered_mask = (self.subCompoundConditions[subCompoundConditions_keys[0]].covered_mask(X))
        if self.operator == LogicalOperator.conjuction:
            for i in range(1, len(subCompoundConditions_keys)):
                covered_mask &= self.subCompoundConditions[subCompoundConditions_keys[i]].covered_mask(X)
        elif self.operator == LogicalOperator.alternative:
            for i in range(1, len(subCompoundConditions_keys)):
                covered_mask |= self.subCompoundConditions[subCompoundConditions_keys[i]].covered_mask(X)
        return covered_mask

    def uncovered_mask(self, X: np.ndarray) -> np.ndarray:
        subCompoundConditions_keys = list(self.subCompoundConditions.keys())
        if len(self.subCompoundConditions) == 0:
            return np.full(X.shape[0], fill_value=True)
        uncovered_mask = (self.subCompoundConditions[subCompoundConditions_keys[0]].uncovered_mask(X))
        if self.operator == LogicalOperator.conjuction:
            for i in range(1, len(subCompoundConditions_keys)):
                uncovered_mask &= self.subCompoundConditions[subCompoundConditions_keys[i]].uncovered_mask(X)
        elif self.operator == LogicalOperator.alternative:
            for i in range(1, len(subCompoundConditions_keys)):
                uncovered_mask |= self.subCompoundConditions[subCompoundConditions_keys[i]].uncovered_mask(X)
        return uncovered_mask

    def positive_covered_mask(self, X: np.ndarray, y: np.ndarray, decision) -> np.ndarray:
        subCompoundConditions_keys = list(self.subCompoundConditions.keys())
        if len(self.subCompoundConditions) == 0:
            return np.full(X.shape[0], fill_value=False)
        positive_covered_mask = (
            self.subCompoundConditions[subCompoundConditions_keys[0]].positive_covered_mask(X, y, decision))
        if self.operator == LogicalOperator.conjuction:
            for i in range(1, len(subCompoundConditions_keys)):
                positive_covered_mask &= self.subCompoundConditions[subCompoundConditions_keys[i]].positive_covered_mask(
                    X, y, decision)
        elif self.operator == LogicalOperator.alternative:
            for i in range(1, len(subCompoundConditions_keys)):
                positive_covered_mask |= self.subCompoundConditions[subCompoundConditions_keys[i]].positive_covered_mask(
                    X, y, decision)
        return positive_covered_mask

    def negative_covered_mask(self, X: np.ndarray, y: np.ndarray, decision) -> np.ndarray:
        subCompoundConditions_keys = list(self.subCompoundConditions.keys())
        if len(self.subCompoundConditions) == 0:
            return np.full(X.shape[0], fill_value=False)
        negative_covered_mask = (
            self.subCompoundConditions[subCompoundConditions_keys[0]].negative_covered_mask(X, y, decision))
        if self.operator == LogicalOperator.conjuction:
            for i in range(1, len(subCompoundConditions_keys)):
                negative_covered_mask &= self.subCompoundConditions[subCompoundConditions_keys[i]].negative_covered_mask(
                    X, y, decision)
        elif self.operator == LogicalOperator.alternative:
            for i in range(1, len(subCompoundConditions_keys)):
                negative_covered_mask |= self.subCompoundConditions[subCompoundConditions_keys[i]].negative_covered_mask(
                    X, y, decision)
        return negative_covered_mask

    def add_subcondition(self, cnd: ElementaryCondition):
        attr = cnd.attribute
        if attr in self.subCompoundConditions.keys():
            compoundCondition = self.subCompoundConditions[attr]
        else:
            compoundCondition = CompoundCondition()
            compoundCondition.set_logical_operator(LogicalOperator.alternative)
        compoundCondition.add_subcondition(cnd)
        self.subCompoundConditions[attr] = compoundCondition

    def add_subconditions(self, cnds: List[ElementaryCondition]):
        for condition in cnds:
            self.add_subcondition(condition)

    def get_subconditions(self):
        all_conditions_list = []
        for compundCondition in self.subCompoundConditions.values():
            all_conditions_list.extend(compundCondition.get_subconditions())
        return all_conditions_list

    def set_logical_operator(self, operator: LogicalOperator):
        self.operator = operator

    def __str__(self):
        s = ""
        operator = " AND " if (
            self.operator == LogicalOperator.conjuction) else " OR "
        operator_internal = " OR " if (
            self.operator == LogicalOperator.conjuction) else " AND "

        for compound_condition in self.subCompoundConditions.values():
            s += "["
            for condition_base in compound_condition.get_subconditions():
                s += condition_base.__str__() + operator_internal
            s = s[0: len(s) - len(operator_internal)]
            s += "]" + operator

        s = s[0: len(s) - len(operator)]
        return s

    def evaluate(self, ex):
        for compound_condition in self.subCompoundConditions.values():
            partial = compound_condition.evaluate(ex)
            if self.operator == LogicalOperator.conjuction and partial == False:
                return False
            elif self.operator == LogicalOperator.alternative and partial == True:
                return True
        return True if (self.operator == LogicalOperator.conjuction) else False


class Rule:

    def __init__(
        self, premise: CompoundCondition, consequence: ElementaryCondition
    ) -> None:
        self.premise = premise
        self.consequence = consequence

    def __str__(self) -> str:
        return "".join(
            ["IF ", self.premise.__str__(), " THEN ", self.consequence.__str__()]
        )


class ClassificationRule(Rule):

    def __init__(
        self, premise: CompoundCondition, consequence: ElementaryCondition
    ) -> None:
        super().__init__(premise, consequence)
        self.decision = consequence.left

    def covers(self, X_df):
        x = X_df.to_numpy()
        y = x[:, -1].astype(str)
        x = x[:, 0:-1]

        P = y[y == self.decision].shape[0]
        N = y.shape[0] - P
        p = x[self.premise.positive_covered_mask(x, y, self.decision)].shape[0]
        n = x[self.premise.negative_covered_mask(x, y, self.decision)].shape[0]

        return p, n, P, N


class RegressionRule(Rule):

    def __init__(
        self, premise: CompoundCondition, consequence: ElementaryCondition
    ) -> None:
        super().__init__(premise, consequence)

    def covers(self, x):
        P = 0
        N = 0
        p = 0
        n = 0

        sum_y = 0.0
        sum_y2 = 0.0

        label_name = self.consequence.attribute
        x.sort_values(label_name, inplace=True)

        orderedNegatives = []
        negatives = []
        positives = []

        for i in range(x.shape[0]):
            ex = x.iloc[i]
            N += 1

            if self.premise.evaluate(ex):  # if covered
                n += 1
                orderedNegatives.append(i)
                negatives.append(i)

                y = ex[label_name]

                sum_y += y
                sum_y2 += y * y

        if len(negatives) == 0:
            return p, n, P, N

        mean_y = sum_y / n
        stddev_y = np.sqrt(sum_y2 / n - mean_y * mean_y)

        medianId = orderedNegatives[len(orderedNegatives) // 2]

        median_y = x.iloc[medianId][label_name]

        # update positives
        for i in range(x.shape[0]):
            ex = x.iloc[i]

            # if inside epsilon
            if np.abs(ex[label_name] - median_y) <= stddev_y:
                N -= 1
                P += 1

                # if covered
                if self.premise.evaluate(ex):
                    negatives.remove(i)
                    n -= 1
                    positives.append(i)
                    p += 1

        return p, n, P, N


class SurvivalRule(Rule):

    def __init__(
        self, premise: CompoundCondition, consequence: ElementaryCondition
    ) -> None:
        super().__init__(premise, consequence)

    def covers(self, x, return_positives: bool = False):
        P = 0
        N = 0
        p = 0
        n = 0
        if return_positives:
            positives = []

        for i in range(x.shape[0]):
            ex = x.iloc[i]
            P += 1
            if self.premise.evaluate(ex):
                p += 1
                if return_positives:
                    positives.append(i)
        if return_positives:
            return p, n, P, N, positives
        else:
            return p, n, P, N
