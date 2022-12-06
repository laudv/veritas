import veritas
from veritas import Domain
import numpy as np
import matplotlib.pyplot as plt
#import lightgbm as lgb

from matplotlib.patches import Rectangle
import matplotlib as mpl

def sqdist1(a, b, a0, b0):
    return np.sqrt((a-a0)**2 + (b-b0)**2)
def mesh(a, b):
    return np.fliplr(np.meshgrid(a, b, indexing="xy"))
def label(a, b, c):
    label = c.copy()
    label += a + b
    label[((c > 3) & (a < -2)) | ((c > 7) & (b > -2))] = 10
    return label


a0 = -3
b0 = 2
a = np.linspace(-5, 5, 200)
b = np.linspace(-5, 5, 200)
extent = [min(a), max(a), min(b), max(b)]

agrid, bgrid = mesh(a, b)

c = sqdist1(a, b, a0, b0)
cgrid = sqdist1(agrid, bgrid, a0, b0)

y = label(a, b, c)

X = np.vstack([agrid.ravel(), bgrid.ravel(), cgrid.ravel()]).T
Y = label(agrid, bgrid, cgrid).ravel()

#dtrain = lgb.Dataset(X, Y)
#params = {"objective": "l2", "boosting": "gbdt", "learning_rate":1.0, "max_depth":2}
#bst = lgb.train(params, dtrain, num_boost_round=1)
#at = veritas.addtree_from_lgb_model(bst)
#
#Ypred = bst.predict(X)
#Ypred2 = at.eval(X)

#if np.max(Ypred-Ypred2) > 1e-5:
#    print("WARNING: AddTree not the same")
#
#with open("examples/sqdistmodel.at", "w") as f:
#    f.write(at.to_json())

with open("examples/sqdistmodel.at") as f:
    at = veritas.AddTree.from_json(f.read())

print(at[0])


s = veritas.Search.max_output(at)
#s.prune([Domain(), Domain(), Domain(0,0)])
s.add_sqdist1_constraint(0, 1, 2, a0, b0)
s.steps(1)
