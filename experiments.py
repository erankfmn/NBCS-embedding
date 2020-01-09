import time

from sklearn import preprocessing, datasets
from sklearn.kernel_approximation import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import LinearSVC, SVC
from sklearn.datasets import load_svmlight_files
from NBCS import NBCS

cv = 5
# from sklearn import datasets
#
dataset = datasets.load_iris()
#dataset = datasets.load_wine()
# dataset = datasets.load_breast_cancer()
# #
X = dataset.data
y = dataset.target
X=preprocessing.MinMaxScaler().fit_transform(X)

#for loading from UCI

#X, y=load_svmlight_files(["assets/letter.scale"])
#X=preprocessing.MinMaxScaler().fit_transform(X.toarray())
#X=(X.toarray()+1)/2
#X=(X+np.ones(X.shape))/2

# #for skin
# with open('assets/Skin_NonSkin.csv') as f:
#     temp = [[int(x) for x in line.split(sep=',')] for line in f]
# temp=np.array(temp)
# y=temp[:,3]
# X=temp[:,:3]/255


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3)

#for polynomial

param_grid = {'C': [0.1,0.5,1,5,10,50,100],
             'degree': [1,2,3,4,5,6,7,8,9] }

clf = GridSearchCV(SVC(kernel='poly'), param_grid,cv=cv)
clf = clf.fit(X_train, y_train)

print("Best estimator found for degree %s by grid search:",clf.best_estimator_.support_vectors_.shape)
print(clf.best_estimator_)

print("score for SVC poly:",clf.score(X_test,y_test))

param_grid = {'C': [0.1,0.5,1,5,10,50,100],
              'gamma': [0.001, 0.005, 0.01, 0.1,0.5,1],  }

clf = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=cv)
clf = clf.fit(X_train, y_train)

print("Best estimator found for RBF:", clf.best_estimator_.support_vectors_.shape)
print(clf.best_estimator_)

print("score for SVC RBF:", clf.score(X_test, y_test))



#NSBC

pipe = Pipeline([
    ('embed', NBCS()),
    ('clf', LinearSVC()),
])

params={'embed__k':[2,3,4,5],
        'clf__C': [0.1,0.5,1,5,10,50,100]}

clf = GridSearchCV(pipe, params,cv=cv)
clf.fit(X_train, y_train)
print("score for NBCS:",clf.score(X_test,y_test), clf)

#polynomial
pipe = Pipeline([
    ('embed', PolynomialFeatures()),
    ('clf', LinearSVC()),
])

params={'embed__degree':[1,2,3,4,5,6,7,8,9],
        'clf__C': [0.1,0.5,1,5,10,50,100]}

clf = GridSearchCV(pipe, params,cv=cv)
clf.fit(X_train, y_train)
print("score for poly:",clf.score(X_test,y_test), clf)


#Rahimi, A. and Recht, B
pipe = Pipeline([
    ('embed', RBFSampler()),
    ('clf', LinearSVC()),
])

params={'embed__gamma':[0.001, 0.005, 0.01, 0.1,0.5,1],
        'clf__C': [0.1,0.5,1,5,10,50,100]}

clf = GridSearchCV(pipe, params,cv=cv)
clf.fit(X_train, y_train)
print("score for rbfsampler:",clf.score(X_test,y_test), clf)


#Nystroem

pipe = Pipeline([
    ('embed', Nystroem()),
    ('clf', LinearSVC()),
])

params={'embed__gamma':[0.001, 0.005, 0.01, 0.1,0.5,1],
        'clf__C': [0.1,0.5,1,5,10,50,100]}

clf = GridSearchCV(pipe, params,cv=cv)
clf.fit(X_train, y_train)
print("score for Nystroem:",clf.score(X_test,y_test), clf)



#Additive Chi Squared Kernel
pipe = Pipeline([
    ('embed', AdditiveChi2Sampler()),
    ('clf', LinearSVC()),
])

params={'embed__sample_steps':[1,2,3],
        'clf__C': [0.1,0.5,1,5,10,50,100]}

clf = GridSearchCV(pipe, params,cv=cv)
clf.fit(X_train, y_train)
print("score for AdditiveChi2Sampler:",clf.score(X_test,y_test), clf)


#SkewedChi2Sampler
chi2_feature = SkewedChi2Sampler(skewedness=.01,
                                 n_components=10,
                                 random_state=0)
X_features = chi2_feature.fit_transform(X_train, y_train)
test_feature=chi2_feature.transform(X_test)

clf.fit(X_features, y_train)

print("score for SkewedChi2Sampler:",clf.score(test_feature,y_test), SkewedChi2Sampler)