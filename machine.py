import numpy as np
import matplotlib.pyplot as pl
from multiprocessing import Process
from sklearn.naive_bayes import GaussianNB

N=1000
x0=np.random.normal(1,1,N).tolist()
x0=[[xx] for xx in x0]
y0=[0 for i in range(N)]
x1=np.random.normal(7,2,N).tolist()
x1=[[xx] for xx in x1]
y1=[1 for i in range(N)]
X=x0+x1
Y=y0+y1

Xpred=np.random.normal(5,5,N).tolist()
Xpred=[[xx] for xx in Xpred]

pl.scatter(range(len(X)),X,c=Y)
clf=GaussianNB()
clf.fit(X,Y)
Ypred=clf.predict(Xpred)
print len(Ypred)
pl.scatter(range(len(Xpred)*2,len(Xpred)*3),Xpred,c=Ypred)

pl.show()

