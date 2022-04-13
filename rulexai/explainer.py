from .importances import (
    ClassificationConditionImportance,
    RegressionConditionImportance,
    SurvivalConditionImportance,
    ConditionImportance,
)
from .models import ClassificationModel, RegressionModel, SurvivalModel, BlackBoxModel
import pandas as pd
import numpy as np
from typing import Union, List
import matplotlib.pyplot as plt

from .reduct import Reduct

Labels = Union[pd.DataFrame, pd.Series]



class BaseExplainer:
    """:meta private:"""
    def __init__(
        self, model, X: pd.DataFrame, y: Labels, type: str = "classification"
    ) -> None:


        self.model = model
        self.X = X
        self.y = y
        self.type = type

        self.condition_importances_ = None
        self.feature_importances_ = None
        self.if_basic_conditions = None

        self.condition_importance_class = None                            
        self._conditions_importances_for_training_set = None 
        self._basic_conditions_importances_for_training_set = None


    def explain(self, measure: str = "C2", basic_conditions: bool = False):
        """Compute conditions importances. The importances of a conditions are computed base on: \n
        Marek Sikora: Redefinition of Decision Rules Based on the Importance of Elementary Conditions Evaluation. Fundam. Informaticae 123(2): 171-197 (2013) \n
        https://dblp.org/rec/journals/fuin/Sikora13.html

        Parameters
        ----------
        measure: str
            Specifies the measure that is used to evaluate the quality of the rules. Possible measures for classification and regression problem are: C2, Lift, Correlation. Default: C2. It is not possible to select a measure for the survival problem, the LogRank test is used by default 
        basic_conditions : bool
            Specifies whether to evaluate the conditions contained in the input rules, or to break the conditions in the rules into base conditions so that individual conditions do not overlap
        Returns
        -------
        self: Explainer
            Fitted explainer with calculated conditions

        """
        self.if_basic_conditions = basic_conditions
        self.condition_importances_ = self._determine_condition_importances(measure)
        self.feature_importances_ = self._determine_feature_importances(self.condition_importances_)

        return self

    def fit_transform(
        self, X: pd.DataFrame, selector=None, y=None, POS=None) -> pd.DataFrame:

        """Creates a dataset based on given dataset in which the examples, instead of being described by the original attributes, will be described with the specified conditions - it will be a set with binary attributes determining whether a given example meets a given condition. It can be considered as kind of dummification.
        Thanks to this function you can discretize data and get rid of missing values. It can be used as prestep for others algorithms.

        Parameters
        ----------
        X: pd.DataFrame
            The input samples from which you want to create binary dataset. Should have the same columns and columns order as X specified when creating Explainer
        selector : string/float
            Specifies on what basis to select the conditions from the rules that will be included as attributes in the transformed set.   
            If None all conditions will be included in the transformed set. If number 0-1 percent of the most important conditions will be selected based on condition importance ranking. If "reduct" the reduct of the conditions set will be selected. Preferably, the option with the percentage of most important conditions will be selected.
        y: Union[pd.DataFrame, pd.Series]
            Only if selector = "reduct".The target values for input sample, used in the determination of the reduct
        POS: float
            Only if selector = "reduct".Target reduct POS
        Returns
        -------
        X_transformed: pd.DataFrame
            Transformed dataset

        """
        
        helper = ConditionImportance(self.model.rules, X, None, self.if_basic_conditions)
        if self.if_basic_conditions:
            rules = helper.split_conditions_in_rules(self.model.rules)
            conditions = helper._get_conditions_with_rules(rules)
        else:
            conditions = helper._get_conditions_with_rules(
                self.model.rules
            )

        binary_dataset = self._prepare_binary_dataset(X, conditions)

        if selector=="reduct":
            reduct = Reduct()
            binary_dataset = reduct.get_reduct(binary_dataset,y,POS)
        elif not selector is None:
            binary_dataset = self._get_top_conditions(binary_dataset, selector)
        
        chosen_conditions_names = binary_dataset.columns
        self.conditions_names = chosen_conditions_names
        self.conditions = []
        for condition in conditions:
            if str(condition) in chosen_conditions_names:
                self.conditions.append(condition)

        return binary_dataset

    def transform(
        self, X: pd.DataFrame) -> pd.DataFrame:

        """Creates a dataset based on given dataset in which the examples, instead of being described by the original attributes, will be described with the specified conditions - it will be a set with binary attributes determining whether a given example meets a given condition. It can be considered as kind of dummification.
        Thanks to this function you can discretize data and get rid of missing values. It can be used as prestep for others algorithms.

        Parameters
        ----------
        X: pd.DataFrame
            The input samples from which you want to create binary dataset. Should have the same columns and columns order as X given in fit_transform
        Returns
        -------
        X_transformed: pd.DataFrame
            Transformed dataset

        """
        transformed_dataset = self._prepare_binary_dataset(X, self.conditions)
            
        return transformed_dataset[self.conditions_names]

   
    def get_rules_covering_example(self, x: pd.DataFrame, y: Labels) -> List[str]:
        """Return rules that covers the given example

        Parameters
        ----------
        x : pd.DataFrame
            The input sample.
        y : Union[pd.DataFrame, pd.Series]
            The target values for input sample.
        Returns
        -------
        rules: List[str]
            Rules that covers the given example

        """
        rules_covering_example = []
        for rule in self.model.rules:
            if rule.premise.evaluate(x):
                rules_covering_example.append(rule)

        return rules_covering_example


    def local_explainability(self, x: pd.DataFrame, y: Labels, plot: bool = False):
        """Displays information about the local explanation of the example: the rules that cover the given example and the importance of the conditions contained in these rules
        
        Parameters
        ----------
        x : pd.DataFrame
            The input sample.
        y : Union[pd.DataFrame, pd.Series]
            The target values for input sample.
        plot : bool
            If True the importance of the conditions will also be shown in the chart. Default: False
        """
        rules_covering_example = self.get_rules_covering_example(x,y)

        print("Example:")
        print(pd.concat([x,y]))
        print("")

        print("Rules that covers this example:")
        for rule in rules_covering_example:
            print(rule)
        print("")
            
        conditions_importances = self.condition_importances_.copy()   
        
        classes_with_conditions = dict()
        for rule in rules_covering_example:
            

            if rule.consequence.left in classes_with_conditions.keys():
                conditions = classes_with_conditions[rule.consequence.left]
            else:
                conditions = []

            conditions.extend(rule.premise.get_subconditions())
            conditions = list(map(str, conditions))
            classes_with_conditions[rule.consequence.left] = conditions



        if self.type == "classification":
            importances_for_covering_rules = pd.DataFrame()
            for j in range(0, conditions_importances.shape[1], 2):
                class_in_consequences = False
                tmp_df = conditions_importances.iloc[:, j : j + 2]
                for cl in classes_with_conditions.keys():
                    if cl in tmp_df.columns[0]:
                        class_in_consequences = True
                        class_name = cl

                if class_in_consequences:
                    tmp_df.loc[~tmp_df[tmp_df.columns[0]].isin(classes_with_conditions[class_name]), tmp_df.columns[0]] = np.NaN
                    tmp_df.dropna(inplace=True)
                    tmp_df.reset_index(drop = True, inplace = True)
                    importances_for_covering_rules = pd.concat([importances_for_covering_rules, tmp_df], ignore_index=False, axis=1)

            importances_for_covering_rules = importances_for_covering_rules.replace(np.nan, "-")
        else:
            conditions_importances.loc[~conditions_importances[conditions_importances.columns[0]].isin(conditions), conditions_importances.columns[0]] = np.NaN
            conditions_importances.dropna(inplace=True)
            importances_for_covering_rules = conditions_importances.reset_index(drop = True)



        print("Importances of the conditions from rules covering the example")
        print(importances_for_covering_rules)

        if plot:
            self.plot_importances(importances_for_covering_rules)

        return importances_for_covering_rules

    def get_rules(self):
        """Return rules from model

        Returns
        -------
        rules: List[str]
            Rules from model
        """
        rules = []
        for rule in self.model.rules:
            rules.append(rule.__str__())
        return rules


    def get_rules_with_basic_conditions(self):
        """Return rules from model with conditions broken down into base conditions so that individual conditions do not overlap

        Returns
        -------
        rules: List[str]
            Rules from the model containing the base conditions 
        """
        rules_with_basic_conditions = []
        helper = ConditionImportance(self.model.rules, self.X, self.y, True)
        rules = helper.rules
        for rule in rules:
            rules_with_basic_conditions.append(rule.__str__())
        return rules_with_basic_conditions


    def plot_importances(self, importances: pd.DataFrame):
        """Plot importances
        Parameters
        ----------
        importances : pd.DataFrame
            Feature/Condition importances to plot.
        """
        
        if "attributes" in importances.columns[0]:
            title = "Feature Importance"
        else:
            title = "Condition Importance"

        plots_number = int(importances.shape[1]/2)

        if self.type == "classification" and plots_number > 1:

            fig, axs = plt.subplots(1,plots_number, sharex = True)
            i = 0
            ticks_number = importances.shape[0]
            y_ticks = np.arange(0, ticks_number)
            for j in range(0, importances.shape[1], 2):

                tmp_df = importances.iloc[:, j : j + 2]
                tmp_df = tmp_df.replace("-", np.nan).dropna()
                tmp_df.sort_values(inplace=True, by=tmp_df.columns[1])
                    
                ticks_values = tmp_df.iloc[:, 1].to_list()
                ticks_all = [0 for _ in range(len(ticks_values),ticks_number)]
                ticks_all.extend(ticks_values)  
                labels = tmp_df.iloc[:, 0].to_list()
                labels_all = ["" for _ in range(len(labels),ticks_number)]
                labels_all.extend(labels)

                colors = ["green" if y >= 0 else "red" for y in ticks_all]

                axs[i].barh(y_ticks, ticks_all, color = colors)
                axs[i].set_yticks(y_ticks)
                axs[i].set_yticklabels(labels_all)
                class_name = tmp_df.columns[0].split(" | ")[0]
                axs[i].set_title(f"Importance for class: {class_name}")
                axs[i].set_xlabel(f"Importance")
                i+=1

            fig.subplots_adjust(wspace = 0.5)
            plt.show()

        else:

            tmp_df = importances.sort_values(by=importances.columns[1])     
            colors = ["green" if y >= 0 else "red" for y in tmp_df.iloc[:,1].to_list()] 
            y_ticks = np.arange(0, len(tmp_df))
            fig, ax = plt.subplots()
            ax.barh(y_ticks, tmp_df.iloc[:, 1], color = colors)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(tmp_df.iloc[:, 0])
            ax.set_title(f"{title}")
            ax.set_xlabel(f"Importance")

            plt.show()


    
    def _determine_condition_importances(self, measure: str = "C2", X = None, y = None):
        if X is None:
            X = self.X
        if y is None:
            y = self.y

        if self.type == "regression":
            self.condition_importance_class = RegressionConditionImportance(
                rules = self.model.rules, data = X, labels = y, if_split = self.if_basic_conditions, measure = measure
            )
        elif self.type == "survival":
            self.condition_importance_class = SurvivalConditionImportance(
                rules = self.model.rules, data = X, labels = y, if_split = self.if_basic_conditions, measure = measure
            )
        else:
            self.condition_importance_class = ClassificationConditionImportance(
                rules = self.model.rules, data = X, labels = y, if_split = self.if_basic_conditions, measure = measure
            )
        return self.condition_importance_class.condition_importances()

    def _determine_feature_importances(self, conditions_importances):
        feature_importances = pd.DataFrame()

        if self.type == "classification":
            
            for j in range(0, conditions_importances.shape[1], 2):
                class_importances = (
                    conditions_importances.iloc[:, j : j + 2]
                    .replace("-", np.nan)
                    .dropna()
                )
                importances_df_tmp = pd.DataFrame()
                class_importances.iloc[:, 0] = class_importances.iloc[:, 0].apply(
                    lambda x: x.split(" = ")[0]
                )
                class_importances = (
                    class_importances.groupby(class_importances.columns[0])
                    .sum()
                    .reset_index()
                )
                class_importances.sort_values(
                    class_importances.columns[1], ascending=False, inplace=True
                )
                class_importances.reset_index(drop=True, inplace= True)

                class_name, _ = class_importances.columns[0].split(" | ")
                importances_df_tmp[class_name + " | attributes"] = pd.Series(class_importances[
                    class_importances.columns[0]
                ])
                importances_df_tmp[class_name + " | importances"] = pd.Series(class_importances[
                    class_importances.columns[1]
                ])
           
                feature_importances = pd.concat(
                    [feature_importances, importances_df_tmp],
                    ignore_index=False,
                    axis=1
                )
             
            feature_importances = feature_importances.replace(np.nan, "-")
                            
        else:

            importances_df = conditions_importances.copy()
            importances_df.iloc[:, 0] = importances_df.iloc[:, 0].apply(
                lambda x: x.split(" = ")[0]
            )
            importances_df = (
                                       
                importances_df.groupby(importances_df.columns[0]).sum().reset_index()
            )
            importances_df.sort_values(
                importances_df.columns[1], ascending=False, inplace=True
            )

            importances_df.rename(
                columns={
                    importances_df.columns[0]: "attributes",
                    importances_df.columns[1]: "importances",
                },
                inplace=True,
            )

            feature_importances = importances_df.replace(np.nan, "-")

        return feature_importances

    def _prepare_binary_dataset(
        self, X: pd.DataFrame, conditions) -> pd.DataFrame:

        x = X.to_numpy()
        binary_dataset_arr = np.zeros((x.shape[0], len(conditions)), dtype=int)
        conditions_names = []

        for i, condition in enumerate(conditions):
            condition.evaluate_mask(binary_dataset_arr, x, column_index=i)
            conditions_names.append(str(condition))

        binary_dataset = pd.DataFrame(binary_dataset_arr, columns=conditions_names)
            
        return binary_dataset

    def _get_top_conditions(self,binary_dataset, percent):
        if self.type == "classification":
            importances_TOP = []
            for j in range(0, self.condition_importances_.shape[1] + 0, 2):
                class_importances = (
                    self.condition_importances_.iloc[:, j]
                    .replace("-", np.nan)
                    .dropna()
                )
                class_importances_TOP_number = np.round(
                    (percent) * class_importances.shape[0]
                )

                if class_importances_TOP_number == 0:
                    class_importances_TOP_number = 1

                class_importances_TOP = class_importances.loc[
                    0 : class_importances_TOP_number - 1
                ]
                importances_TOP.extend(list(class_importances_TOP))

            importances_TOP_list = list(set(importances_TOP))
        else:

            importances_TOP_number = np.round(
                (percent) * self.condition_importances_.shape[0]
            )
            if importances_TOP_number == 0:
                importances_TOP_number = 1

            importances_TOP = self.condition_importances_.loc[
                0 : importances_TOP_number - 1
            ]
            importances_TOP_list = importances_TOP["conditions"].to_list()
            
        return binary_dataset[importances_TOP_list]


class RuleExplainer(BaseExplainer):
    def __init__(
        self, model, X: pd.DataFrame, y: Labels, type: str = "classification"
    ) -> None:
        """RuleExplainer

        Parameters
        ----------
        model : Model = Union[RuleClassifier, RuleRegressor, SurvivalRules, CN2UnorderedClassifier, CN2SDUnorderedClassifier, DecisionTreeClassifier, DecisionTreeRegressor, SurvivalTree, List[str]]
            Model to be analyzed. RuleXai supports the following Rule models:
             - RuleKit(https://adaa-polsl.github.io/RuleKit-python/): RuleClassifier, RuleRegressor, SurvivalRules
             - Orange (https://orangedatamining.com/): CN2UnorderedClassifier, CN2SDUnorderedClassifier
            It can also extract rules from decision trees:
             - scikit-learn (https://scikit-learn.org/stable/): DecisionTreeClassifier, DecisionTreeRegressor
             - scikit-survival (https://scikit-survival.readthedocs.io/en/stable/): SurvivalTree
            Or you can provide a list of rules as:
             - classification:
                IF attribute1 = (-inf, value) AND  ...  AND attribute2 = <value1, value2) THEN label_atrribute = {class_name}
             - regression:
                IF attribute1 = (-inf, value) AND  ...  AND attribute2 = <value1, value2) THEN target_attribute = {value}
             - survival:
                IF attribute1 = (-inf, value) AND  ...  AND attribute2 = <value1, value2) THEN survival_status_attribute = {survival_status}
        X : pd.DataFrame
            The training dataset used during provided model training
        y : Union[pd.DataFrame, pd.Series]
            The target values (class labels, real number, survival status) used during provided model training
        type : str = None
            The type of problem that the provided model solves. You can choose between:
                - "classification"
                - "regression"
                - "survival"
                default: "classification"
        Attributes
        ----------
        condition_importances_ : pd.DataFrame
            Computed conditions importances
        feature_importances_ : pd.DataFrame
            Feature importances computed base on conditions importances
        """

        if isinstance(y, pd.DataFrame):
            label_name = y.columns[0]
        else:
            label_name = y.name

        if type == "regression":
            model = RegressionModel(model = model, feature_names= X.columns, label_name=label_name)
        elif type == "survival":
            model = SurvivalModel(model=model, fetures_names = X.columns, survival_status_name=label_name)
        else:
                                        
            model = ClassificationModel(model = model, feature_names = X.columns, class_names = np.unique(y), label_name = label_name)

        super().__init__(model, X, y, type)

class Explainer(BaseExplainer):
    def __init__(
        self, X: pd.DataFrame, model_predictions: Labels, type: str = "classification"
    ) -> None:
        """Explainer

        Parameters
        ----------
        X : pd.DataFrame
            The training dataset used during provided model training
        model_predictions : Union[pd.DataFrame, pd.Series]
            The training dataset used during provided model training
        type : str
            The type of problem that the provided model solves. You can choose between:
                - "classification"
                - "regression"
            default: "classification"
        Attributes
        ----------
        condition_importances_ : pd.DataFrame
            Computed conditions importances on given dataset
        feature_importances_ : pd.DataFrame
            Feature importances computed base on conditions importances
        """

        
        if (not isinstance(model_predictions, pd.DataFrame)) and (not isinstance(model_predictions, pd.Series)) : 
              model_predictions =  pd.DataFrame(model_predictions, columns=["class"])                
        self._bb_model = BlackBoxModel(X, model_predictions, type)
        super().__init__(None, X, model_predictions, type)

    def explain(self, measure: str = "C2", basic_conditions: bool = False, X_org = None):
        """Compute conditions importances. The importances of a conditions are computed base on: \n
        Marek Sikora: Redefinition of Decision Rules Based on the Importance of Elementary Conditions Evaluation. Fundam. Informaticae 123(2): 171-197 (2013) \n
        https://dblp.org/rec/journals/fuin/Sikora13.html

        Parameters
        ----------
        measure: str
            Specifies the measure that is used to evaluate the quality of the rules. Possible measures for classification and regression problem are: C2, Lift, Correlation. Default: C2. It is not possible to select a measure for the survival problem, the LogRank test is used by default 
        basic_conditions : bool
            Specifies whether to evaluate the conditions contained in the input rules, or to break the conditions in the rules into base conditions so that individual conditions do not overlap
        X_org:
            The dataset on which the rule-based model should be built. It can be the set on which the black-box model was learned or this set before preprocessing (imputation of missing values, dummification, scaling), because such a set can be handled by the rule model 
        Returns
        -------
        self: Explainer
            Fitted explainer with calculated conditions

        """
        self.if_basic_conditions = basic_conditions

        if X_org is None:
            self.model = self._bb_model.get_rules_model(self.X)
            self.condition_importances_ = self._determine_condition_importances(measure)
        else:
            self.model = self._bb_model.get_rules_model(X_org)
            self.condition_importances_ = self._determine_condition_importances(measure,X_org)
 
        self.feature_importances_ = self._determine_feature_importances(self.condition_importances_)

        return self