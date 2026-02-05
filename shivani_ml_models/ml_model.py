import pandas as pd
from sklearn.model_selection import train_test_split
# SVM and RandomForest 
from sklearn import svm as sk_svm
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn import datasets, ensemble
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.utils.fixes import parse_version
from sklearn.metrics import mean_squared_error, r2_score

class Data():
    def __init__(self, csv_path, target):
        self.data = pd.read_csv(csv_path)
        self.target = target
        self.model_data = []
        self.y_true = []
        self.X_trainval = []
        self.X_test = []
        self.y_trainval = []
        self.y_test = []
        # TODO include validation set too??
    def split_data(self, test_size, seed):
        y = self.data[self.target]
        X = self.data.drop(columns=[self.target, 'MolecularFormula', 'CID', 'SMILES', 'ConnectivitySMILES', 'InChIKey'])
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=seed,
        )
        self.X_trainval = X_trainval
        self.X_test = X_test
        self.y_trainval = y_trainval
        self.y_test = y_test
        return self.X_trainval, self.X_test, self.y_trainval, self.y_test
    
# super basic models w/o finetuning 

class SVM():
    def __init__(self, X_train, y_train, X_test, y_test):
        self.svm_model = sk_svm.SVR()
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_pred = []
    def run_model(self):
        self.svm_model.fit(self.X_train, self.y_train)
        self.y_pred = self.svm_model.predict(self.X_test)
        return self.svm_model
    def eval(self):
        mse = mean_squared_error(self.y_test, self.y_pred)
        r2 = r2_score(self.y_test, self.y_pred)
        return {'mse': mse, 'r2': r2}


class RandomForest():
    def __init__(self, X_train, y_train, X_test, y_test):
        self.rf = RandomForestRegressor(max_depth=2, random_state=0)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_pred = []
    def run_model(self):
        self.rf.fit(self.X_train, self.y_train)
        self.y_pred = self.rf.predict(self.X_test)
        return self.rf
    def eval(self):
        mse = mean_squared_error(self.y_test, self.y_pred)
        r2 = r2_score(self.y_test, self.y_pred)
        return {'mse': mse, 'r2': r2}

# TODO - tune hyperparameters later
class XGBoost():
    def __init__(self, X_train, y_train, X_test, y_test):
        self.params = {
            "n_estimators": 500,
            "max_depth": 4,
            "min_samples_split": 5,
            "learning_rate": 0.01,
            "loss": "squared_error",
        } 
        self.xgboost = ensemble.GradientBoostingRegressor(**self.params)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_pred = []
    def run_model(self):
        self.xgboost.fit(self.X_train, self.y_train)
        self.y_pred = self.xgboost.predict(self.X_test)
        return self.xgboost
    def eval(self):
        mse = mean_squared_error(self.y_test, self.y_pred)
        r2 = r2_score(self.y_test, self.y_pred)
        return {'mse': mse, 'r2': r2}
    


# for later 
    

def main():
    data = Data('../data/pubchem_properties.csv', "MolecularWeight")
    X_trainval, X_test, y_trainval, y_test = data.split_data(0.2, 42)
    # Drop NaN values
    mask_trainval = ~(X_trainval.isna().any(axis=1) | y_trainval.isna())
    mask_test = ~(X_test.isna().any(axis=1) | y_test.isna())

    # Apply the masks
    X_trainval_clean = X_trainval[mask_trainval]
    y_trainval_clean = y_trainval[mask_trainval]
    X_test_clean = X_test[mask_test]
    y_test_clean = y_test[mask_test]
    svm = SVM(X_trainval_clean, y_trainval_clean, X_test_clean, y_test_clean)
    svm.run_model()
    svm_results = svm.eval()
    random_forest = RandomForest(X_trainval_clean, y_trainval_clean, X_test_clean, y_test_clean)
    random_forest.run_model()
    random_forest_results = random_forest.eval()
    xgboost = XGBoost(X_trainval_clean, y_trainval_clean, X_test_clean, y_test_clean)
    xgboost.run_model()
    xgboost_results = xgboost.eval()

    print(f"SVM Results: {svm_results}, Random Forest Results: {random_forest_results}, XGBoost Results: {xgboost_results}")

if __name__ == "__main__":
    main()


