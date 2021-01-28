import datasets, sys
from veritas import Optimizer
from veritas import RobustnessSearch, VeritasRobustnessSearch
from veritas.kantchelian import KantchelianAttack, KantchelianTargetedAttack
import numpy as np
import matplotlib.pyplot as plt

ds = datasets.Calhouse()
ds.load_dataset()
ds.load_model(10, 4)

num_attr = ds.X.shape[1]
at = ds.at

def get_leaf_ids(at, example):
    return [at[i].predict_leaf(example) for i in range(len(at))]

def get_all_domains(at, X, num_attr):
    doms = []
    for example_i in range(len(X)):
        example = ds.X.iloc[example_i,:]
        leaf_ids = get_leaf_ids(at, example)
        prediction = sum(at[i].get_leaf_value(n) for i, n in enumerate(leaf_ids))
        ds = at.get_domains(leaf_ids)
        doms.append((ds, prediction))
    return doms

doms = get_all_domains(at, ds.X.iloc[0:100,:], num_attr)
#print(doms)

doms_per_attr = {}
for dom, pred in doms:
    for k, v in dom.items():
        l = doms_per_attr.get(k, [])
        l.append(v)
        doms_per_attr[k] = l

print("num_attr", num_attr)
for k, v in doms_per_attr.items():
    print(k, len(v))

#example = ds.X.iloc[1,:]
#leaf_ids = get_leaf_ids(at, example)
#
#print(leaf_ids)
#print(at.get_domains(leaf_ids))

#plt.hist(mnist.y, bins=100)
#plt.show()
#plt.hist(np.log(mnist.y), bins=100)
#plt.show()
#sys.exit()
#example_i = 1
#example = list(mnist.X.iloc[example_i,:])
#example_label = int(mnist.y[example_i])
#
#at0 = mnist.at[example_label]
#at1 = mnist.at[(example_label+1)%10]
#
#actual_prediction0 = at0.predict_single(example)
#actual_prediction1 = at1.predict_single(example)
#
#opt = Optimizer(minimize=mnist.at[1], maximize=mnist.at[0],
#        matches=set(), match_is_reuse=False)
#opt.prune_example(example, delta=20)
#
#data = opt.merge(max_merge_depth=2)
#print(data)
