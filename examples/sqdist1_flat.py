import veritas
from veritas import Domain
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle
import matplotlib as mpl

def sqdist1(a, b, a0, b0):
    return np.sqrt((a-a0)**2 + (b-b0)**2)
def mesh(a, b):
    return np.fliplr(np.meshgrid(a, b, indexing="xy"))
def label(a, b, c):
    label = c.copy()
    label += a + b
    label[((c > 3) & (a < 2)) | ((c > 7) & (b > -2))] = 5
    return label


a0 = -3
b0 = 2
a0 = 0
b0 = 0
a = np.linspace(-5, 5, 200)
b = np.linspace(-5, 5, 200)
extent = [min(a), max(a), min(b), max(b)]

agrid, bgrid = mesh(a, b)

c = sqdist1(a, b, a0, b0)
cgrid = sqdist1(agrid, bgrid, a0, b0)

y = label(a, b, c)

X = np.vstack([agrid.ravel(), bgrid.ravel(), cgrid.ravel()]).T
Y = label(agrid, bgrid, cgrid).ravel()

import lightgbm as lgb
dtrain = lgb.Dataset(X, Y)
params = {"objective": "l2", "boosting": "gbdt", "learning_rate":1.0, "max_depth":3}
bst = lgb.train(params, dtrain, num_boost_round=1)
at = veritas.addtree_from_lgb_model(bst)

Ypred = bst.predict(X)

#if np.max(Ypred-Ypred2) > 1e-5:
#    print("WARNING: AddTree not the same")

with open("examples/sqdistmodel.at", "w") as f:
    f.write(at.to_json())

with open("examples/sqdistmodel.at") as f:
    at = veritas.AddTree.from_json(f.read())

for t in at:
    print(t)
Ypred2 = at.eval(X)


s = veritas.Search.max_output(at)
#s.prune([Domain(), Domain(), Domain(0,3)])
s.add_sqdist1_constraint(0, 1, 2, a0, b0)
print(s.steps(1000))


# plots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,5))

im=ax1.imshow(Y.reshape(cgrid.shape), extent=extent, interpolation="none", vmin=min(Y), vmax=max(Y))
ax1.set_xlabel("a")
ax1.set_ylabel("b")
ax1.set_title("true value")
fig.colorbar(im, ax=ax1)

im=ax2.imshow(Ypred2.reshape(cgrid.shape), extent=extent, interpolation="none", vmin=min(Y), vmax=max(Y))
ax2.set_xlabel("a")
ax2.set_ylabel("b")
ax2.set_title("predicted value")
fig.colorbar(im, ax=ax2)

cmap = mpl.cm.cool
norm = mpl.colors.Normalize(vmin=min(Y), vmax=max(Y))

for i in range(s.num_solutions()):
    sol = s.get_solution(i)
    print(i, sol.box(), np.float32(sol.output))

true_unique = sorted(np.unique(Ypred2))
sols = np.array([s.get_solution(i).output for i in range(s.num_solutions())], dtype=np.float32)
sols_unique = sorted(np.unique(sols))
print("num_sol", sols, s.num_solutions())
print("true_unique", true_unique, len(true_unique))
print("sols_unique", sols_unique, len(sols_unique))

i, j = 0, 0
while i < len(true_unique) or j < len(sols_unique):
    if i < len(true_unique) and j < len(sols_unique):
        x = true_unique[i]
        y = sols_unique[j]
        if x < y:
            print("T", x, i)
            i += 1
        else:
            print("S", y, j)
            j += 1
    elif i < len(true_unique):
        print("T", true_unique[i], i)
        i += 1
    else:
        print("S", sols_unique[j], j)
        j += 1

import z3

def getfloat(x):
    if z3.is_algebraic_value(x):
        x = x.approx(20)
    return float(x.numerator_as_long()) / float(x.denominator_as_long())

#for i in range(1, 3): # overlap!
#for i in range(3, 7): # no overlap
#for i in range(7, 11): # overlap!
for i in range(s.num_solutions()):
    sol = s.get_solution(i)
    box = sol.box()
    xlo = max(extent[0], box[0].lo if 0 in box else -np.inf)
    xhi = min(extent[1], box[0].hi if 0 in box else np.inf)
    ylo = max(extent[2], box[1].lo if 1 in box else -np.inf)
    yhi = min(extent[3], box[1].hi if 1 in box else np.inf)
    dlo = max(extent[2], box[2].lo if 2 in box else -np.inf)
    dhi = min(extent[3], box[2].hi if 2 in box else np.inf)
    
    #r = Rectangle((xlo, ylo), xhi-xlo, yhi-ylo, fill=True, facecolor=cmap(norm((sol.output))), alpha=1.0)
    r = Rectangle((xlo, ylo), xhi-xlo, yhi-ylo, fill=True, facecolor=(1,0,0,.2))
    #print(i, box, sol.output, r)
    ax3.add_patch(r)
    #r = Rectangle((xlo, ylo), xhi-xlo, yhi-ylo, ls="-", lw=1, fill=False)
    ax3.add_patch(r)

    #x, y, d = z3.Reals("x y d")
    #solver = z3.Solver()
    #solver.add(x**2 + y**2 - d**2 == 0)
    #solver.add(x <= xhi)
    #solver.add(y <= yhi)
    #solver.add(d <= dhi)
    #solver.add(x >= xlo)
    #solver.add(y >= ylo)
    #solver.add(d >= dlo)
    #try:
    #    solver.check()
    #    model = solver.model()
    #    #print(model, box)
    #    example = [getfloat(model[x]), getfloat(model[y]), getfloat(model[d])]
    #    print(example, box, at.eval(example), sol.output)
    #except Exception as e:
    #    print("no model", e, box, sol.output)



ax3.set_xlim(extent[0:2])
ax3.set_ylim(extent[2:4])
ax3.plot([a0], [b0], ".", c="red")
plt.show()
