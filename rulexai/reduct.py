import pandas as pd



class Reduct:

    def __init__(self) -> None:
        pass

    def calculate_POS(self, reduct: pd.DataFrame, y: pd.DataFrame):
        POS = []
        
        reduct["if_duplicated"] = reduct.duplicated(keep=False)
        positive_region = [reduct.iloc[i,:].to_numpy() for i in range(reduct.shape[0]) if reduct.if_duplicated.iloc[i] == False]

        reduct["label"] = y
        df_duplicated = reduct[reduct.if_duplicated == True]
        df_duplicated = df_duplicated.drop(["if_duplicated"], axis=1)

        df_duplicated.sort_values(by = df_duplicated.columns.to_list(), inplace = True)
        df_duplicated.reset_index(inplace= True, drop = True)

        df_labels = df_duplicated["label"]
        df_duplicated = df_duplicated.drop(["label"], axis = 1)

        start_equivalence = 0
        for i in range(1, df_duplicated.shape[0]):
            if (not df_duplicated.iloc[i-1,:].equals(df_duplicated.iloc[i,:])) or i == (df_duplicated.shape[0] - 1):
                
                if i == (df_duplicated.shape[0] - 1):
                    end_equivalence = i + 1
                else: 
                    end_equivalence = i
                
                Xj = df_duplicated.iloc[start_equivalence:end_equivalence, :]
                Xj = Xj.assign(label = df_labels.iloc[start_equivalence:end_equivalence])

                if len(Xj["label"].unique()) == 1:
                    positive_region.extend([Xj.iloc[i,:].to_numpy() for i in range(Xj.shape[0])])
                
                start_equivalence = end_equivalence

        return positive_region

    
    def eliminate_irrelevant_attributes(self, R: pd.DataFrame, y: pd.DataFrame):

        POS_R = self.calculate_POS(R.copy(), y)
        m = len(POS_R)
        A = R.columns.to_list()
        for a in A:
            POS_R_a = self.calculate_POS(R.drop([a], axis = 1), y)
            m_a = len(POS_R_a)
            if m_a == m:
                R = R.drop([a], axis = 1)
        
        return R

    def eliminate_irrelevant_attributes_based_on_user_POS(self,R: pd.DataFrame, y: pd.DataFrame, user_POS: float):

        POS_R = self.calculate_POS(R.copy(), y)
        all_objects_number = len(R)
        m = len(POS_R)/all_objects_number
        
        if user_POS > m:
            print("Warning! The set POS is greater than the reduct POS. The reduct will not be limited")
            return R 

        A = R.columns.to_list()
        for a in A:
            POS_R_a = self.calculate_POS(R.drop([a], axis = 1), y)
            m_a = len(POS_R_a)/all_objects_number
            if m_a >= user_POS:
                R = R.drop([a], axis = 1)
        return R

    # Jonhson's algorithm
    def semi_minimal_reduct(self, x: pd.DataFrame):

        repeate = True
        R = pd.DataFrame()
        L = [x]
        A = x.columns.to_list()

        while(repeate):

            W_U_R = dict()
            for a in A:            
                W_U_R[a] = 0
                for Xi in L:
                    cardinalities = Xi[a].value_counts()
                    W_X_i = (pow(cardinalities.sum(), 2) -
                            sum([pow(xi, 2) for xi in cardinalities]))/2
                    W_U_R[a] += W_X_i

            # attribute with the largest W_U_R(a)
            a = max(W_U_R, key=lambda key: W_U_R[key])

            A.remove(a)
            R[a] = x[a].copy()

            tmp_L = []
            for Xi in L:
                tmp_L.extend([Xi[Xi[a] == value] for value in Xi[a].unique()])
            L = tmp_L

            if W_U_R[a] == 0 or len(A) == 0:
                repeate = False

        return R


    def get_reduct(self, x: pd.DataFrame, y: pd.DataFrame = None, POS: float = None):
        
        R = self.semi_minimal_reduct(x)
        if y is None:
            return R 

        if POS is None:
            R = self.eliminate_irrelevant_attributes(R, y)
        else:
            R = self.eliminate_irrelevant_attributes_based_on_user_POS(R,y,POS)
        return R

  