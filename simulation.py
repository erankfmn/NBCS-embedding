
from mesh import *
from NBCS import *


N=1000
C=1000
gamma=0.01
D=2 #dimension
t=6 #polytope order

X = np.random.normal(0,1,(N,D))
ws= np.random.normal(0,1,(t,D))
ws = ws / (np.linalg.norm(ws,axis=1).reshape(-1,1))

u=np.random.uniform(0,1,(N,1))
X=X/(np.linalg.norm(X,axis=1).reshape(-1,1))*(u**(1/D))

y=np.ones(N)

for index in range(N):
    z = 1
    for w in ws :
        if((X[index].dot(w)- 0.5 - gamma) > 0):
            z = -1
        else :
            if ((X[index].dot(w) - 0.4 -gamma) > 0):
                z =0
    y[index]=z

X=(X[y!= 0]+ [1,1])/2
y=y[y!= 0]

fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

xx, yy = make_meshgrid(X[:, 0], X[:, 1])
for k, title, ax in zip(range(1,5), range(1,5), sub.flatten()):

    model = NBCS(C,k=k)
    embedder = model.fit(X, y)
    points=embedder.transform(X)
    clf=SVC(C, kernel='linear').fit(points,y)
    # plot the decision function
    xy = embedder.transform(np.vstack([xx.ravel(), yy.ravel()]).T)

    plot_contours(ax, clf, xx, yy,xy,
                 cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(0, 1)
    ax.set_ylim(0 ,1)
    ax.set_title(title)

    # # plot decision boundary and margins
    Z = clf.decision_function(xy).reshape(xx.shape)
    ax.contour(xx,yy, Z, colors='k', levels=[-1,0,1], alpha=0.8,
                linestyles=['--', '-', '--'])

    x1 = np.concatenate((model.list[:, 0], [0]))
    x2 = np.concatenate((model.list[:, 1], [0]))

    for i in model.indicies:
        i=np.concatenate((i,[i[0]]))
        ax.plot(x1[i], x2[i], 'ro-')
plt.show()

