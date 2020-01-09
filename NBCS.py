from numpy.linalg import LinAlgError
from scipy.sparse import csr_matrix, lil_matrix
import itertools
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import SVR, SVC, LinearSVC
import numpy as np
import scipy as sp

class NBCS(BaseEstimator, TransformerMixin): #(BaseEstimator,RegressorMixin):

    def __init__(self,C=1,k=1):
        self.C=C
        self.k =k

    def add_point(self,point):

        index, r = self.find_point(point)
        self.list=np.concatenate((self.list ,[point]))
        new=np.zeros(point.shape[0]+1,dtype=int)
        combinations = np.array(list(itertools.combinations(self.indicies[index], point.shape[0])))
        for simplex in combinations:
            simplex=np.concatenate((simplex,[self.list.shape[0]-1]))
            new=np.vstack((new,simplex))

        self.indicies=np.vstack((self.indicies,new[1:,:]))
        self.indicies=np.delete(self.indicies,index,0)


    def emmbed(self,X):
        d = lil_matrix((X.shape[0], self.list.shape[0]))
        self.category=np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            try:
                j,r = self.find_point(X[i, :])
                d[i, self.indicies[j]] = r
                self.category[i]=j
            except TypeError:
                print("type error:",i,X[i, :])
            except IndexError:
                print("index error:", i,X[i, :])
        return d

    def find_point(self, pts):
        "returns which simplex you belong to and with  coefficients"
        for j in range(self.indicies.shape[0]):
            h = np.vstack([self.list[self.indicies[j]].T, np.ones(len(pts) + 1)])
            s = np.append(pts, 1)
            try:
                r = np.linalg.solve(h, s)
            except LinAlgError:
                print(h,s)
            flag = 1
            for alpha in r:
                if alpha < -1e-3:
                    flag = 0
                    break
            if (flag == 1):
             return j , r



    def find_point2(self, pts):
        "returns which simplex you belong to and with what coefficients"
        for j in range(self.indicies.shape[0]):
            h = sp.vstack([self.list[self.indicies[j]].T, np.ones(pts.shape[0] + 1)])
            s = sp.vstack([pts, 1])
            r = sp.linalg.solve(h, s)
            flag = 1
            for alpha in r:
                if alpha < -1e-3:
                    flag = 0
                    break
            if (flag == 1):
                return j, r.T



    def fit_barry(self, X, y=None ):
        self.list = np.eye(X.shape[1]) * X.shape[1]
        self.list = np.vstack([np.zeros(X.shape[1]), self.list])
        # self.list = np.array([[0, 0], [0, 2], [2, 0]])
        self.indicies = np.array([np.arange(X.shape[1] + 1)])

        for itr in range(self.k):
            indicies=self.indicies[:]
            d = self.emmbed(X)

            for simplex in range(indicies.shape[0]):
                 q = (self.category == simplex)
                 if not any(q) or X[q].shape[0] < 3:
                     continue
                 self.add_point(np.mean(self.list[indicies[simplex]],axis=0))
        return self

    def fit_adapt(self, X, y=None):
        svc = LinearSVC()
        self.list = np.eye(X.shape[1]) * X.shape[1]
        self.list = np.vstack([np.zeros(X.shape[1]), self.list])
        # self.list = np.array([[0, 0], [0, 2], [2, 0]])
        self.indicies = np.array([np.arange(X.shape[1] + 1)])
        for iter in range(self.k):
            indicies = self.indicies[:]
            d = self.emmbed(X)
            svc.fit(d, y)
            for simplex in range(indicies.shape[0]):
                q = (self.category == simplex)
                if not any(q):
                    continue
                X1 = X[q]
                y1 = y[q]
                d1 = d[q]
                p = (svc.predict(d1) != y1)
                if not any(p) or X1[p].shape[0] < 10:
                    continue
                self.add_point(np.mean(X1[p],axis=0))
        return self


    def transform(self,X):
        return self.emmbed(X)

    fit = fit_barry
