from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

def KNN(X,Y,X_test,Y_test):
    nn = KNeighborsClassifier()
    nn.fit(X,Y)
    print(confusion_matrix(Y_test,nn.predict(X_test),labels=['yes','no']))