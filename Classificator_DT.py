from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

def DecTree(X,Y,X_test,Y_test):
    dt = DecisionTreeClassifier()
    dt.fit(X,Y)
    print(confusion_matrix(Y_test,dt.predict(X_test),labels=['yes','no']))