from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

def LogReg(X,Y,X_test,Y_test):
    lr = LogisticRegression(solver='lbfgs')
    lr.fit(X,Y)
    print(confusion_matrix(Y_test,lr.predict(X_test),labels=['yes','no']))
