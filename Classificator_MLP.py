from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

def callMLP(X,Y,X_test,Y_test):
    mlp = MLPClassifier()
    mlp.fit(X,Y)
    print(confusion_matrix(Y_test,mlp.predict(X_test),labels=['yes','no']))