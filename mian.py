import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import VarianceThreshold
import Classificator_MLP as clfMlp
import Classificator_SVM as clfsvm
import Classificator_DT as clfdt
import Classificator_NB as clfnb
import Classificator_NN as clfnn
import Classificator_LR as clflr

def main ():
    df = pd.read_csv("spam.dat",sep=',')
    X = df.drop(labels=['target'], axis=1)
    y = df['target']
    print(X.shape)
    X_new = VarianceThreshold(threshold=0.9*(1-0.9)).fit_transform(X, y)
    print(X_new.shape)    
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.20)
    print("MLP")
    clfMlp.callMLP(X_train, y_train, X_test, y_test)
    print("SVM")
    clfsvm.SVM(X_train, y_train, X_test, y_test)
    print("DT")
    clfdt.DecTree(X_train, y_train, X_test, y_test)
    print("NB")
    clfnb.NaiveBayes(X_train, y_train, X_test, y_test)
    print("KNN")
    clfnn.KNN(X_train, y_train, X_test, y_test)
    print("LR")
    clflr.LogReg(X_train, y_train, X_test, y_test)
if __name__ == "__main__":
    main()