from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import numpy
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_validate

def plot_validation_curve(estimator, param, param_name, title, X, y, ylim=None, cv=None,n_jobs=None,):
    train_scores, test_scores = validation_curve(estimator, X, y, param_name=param_name, param_range=param, cv=cv, scoring="accuracy", n_jobs=n_jobs)
    train_scores_mean = numpy.mean(train_scores, axis=1)
    train_scores_std = numpy.std(train_scores, axis=1)
    test_scores_mean = numpy.mean(test_scores, axis=1)
    test_scores_std = numpy.std(test_scores, axis=1)
    plt.figure()
    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel("Score")
    plt.ylim(0.5, 1.1)
    lw = 2
    plt.xticks(range(len(param)), param)
    plt.plot(range(len(param)), train_scores_mean, label="Training score",
                color="darkorange", lw=lw)
    plt.fill_between(range(len(param)), train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.2,
                    color="darkorange", lw=lw)
    plt.plot(range(len(param)), test_scores_mean, label="Cross-validation score",
                color="navy", lw=lw)
    plt.fill_between(range(len(param)), test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.2,
                    color="navy", lw=lw)
    plt.legend(loc="best")
    plt.savefig(title)

def SVM(X,Y,X_test,Y_test):
    parameter = {
    # [0.000001,0.0001,0.01,1,10,1000,100000]
    'gamma': [0.01],
    # [0.000001,0.0001,0.01,1,10,1000,100000,10000000,10000000]
    'C': [10],
    'kernel':['rbf','poly','sigmoid']
    }
    svm = GridSearchCV(SVC(),parameter,n_jobs=4,cv=5)
    plot_validation_curve(SVC(C=10,gamma=0.01),parameter['kernel'],'kernel',"Validation curve for SVM kernel",X,Y)
    svm.fit(X,Y)
    print(confusion_matrix(Y_test,svm.predict(X_test),labels=['yes','no']))