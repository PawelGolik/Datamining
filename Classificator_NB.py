from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix

def NaiveBayes(X,Y,X_test,Y_test):
    nb = MultinomialNB()
    nb.fit(X,Y)
    print(confusion_matrix(Y_test,nb.predict(X_test),labels=['yes','no']))