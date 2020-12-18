import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import z3

import veritas
import veritas.xgb


# Generate a random dataset
np.random.seed(14)
N = 10000
x = np.random.rand(N, 1)
y = np.random.rand(N, 1)
dist = np.sqrt(x**2 + y**2)
s = x + y
target = ((dist < 0.5) & (s > 0.2)) | ((x+2*y) > 2.0)

plt.plot(x[target], y[target], '.')
plt.plot(x[~target], y[~target], 'x')
plt.show()

X = np.concatenate((x, y, dist, s), axis=1)

xtrain = xgb.DMatrix(X, label=target, missing=None)
params = {
    "learning_rate": 0.3,
    "max_depth": 4,
    "objective": "binary:hinge",
    "eval_metric": "error",
    "tree_method": "hist",
    "seed": 1,
    "nthread": 1,
}
bst = xgb.train(params, xtrain, 10, [(xtrain, "train")])

features = ["x", "y", "dist"]
feat2id = {f : i for i, f in enumerate(features)}
at = veritas.xgb.addtree_from_xgb_model(bst)
at.base_score = 0.5

pred_raw_at = np.array(at.predict(X))
pred_raw = bst.predict(xtrain, output_margin=True)
print("max error", max(pred_raw_at - pred_raw))

Xv = np.zeros((100*100, 4))
for i, xv in enumerate(np.linspace(0, 1, 100)):
    for j, yv in enumerate(np.linspace(0, 1, 100)):
        Xv[i*100+j, 0:2] = [xv, yv]
        
Xv[:,2] = np.sqrt(Xv[:,0]**2+Xv[:,1]**2)
Xv[:,3] = Xv[:,0]+Xv[:,1]

vs = bst.predict(xgb.DMatrix(Xv), output_margin=True)

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 6))
pred = (pred_raw.reshape((N,1)) > 0.0)
ax0.plot(x[pred&target], y[pred&target], '.', color="darkblue", alpha=0.5, label="true pos")
ax0.plot(x[~pred&~target], y[~pred&~target], '.', color="darkred", alpha=0.5, label="true neg")
ax0.plot(x[pred&~target], y[pred&~target], 'x', color="blue", label="false pos")
ax0.plot(x[~pred&target], y[~pred&target], 'x', color="red", label="false neg")
im = ax1.imshow(vs.reshape(100,100).T, origin="lower", cmap="Spectral")
fig.colorbar(im, ax=ax1)
try:
    ax1.plot(points[:,0]*100,points[:,1]*100, ".", color="purple")
    #ax1.plot([20,100],[0, 80])
except: pass
plt.show()

opt = veritas.Optimizer(maximize=at)

opt.adjuster.add_norm(opt.feat_info.get_id(1, 0), 0.0, # X2 = sqrt(X0^2 + X1^2)
                      opt.feat_info.get_id(1, 1), 0.0,
                      opt.feat_info.get_id(1, 2))
opt.adjuster.add_sum(1.0, opt.feat_info.get_id(1, 0),  # X3 = X0 + X1
                     1.0, opt.feat_info.get_id(1, 1),
                      opt.feat_info.get_id(1, 3))
opt.adjuster.add_less_than(opt.feat_info.get_id(1, 1),  # X1 < X0 - 0.2
                      opt.feat_info.get_id(1, 0), -0.2)
box = [
    veritas.RealDomain(0, 1),
    veritas.RealDomain(0, 1),
    veritas.RealDomain(0.0, np.sqrt(2)),
    veritas.RealDomain(0.0, 2.0)
]
print("num reachable leafs before prune", opt.g1.num_vertices())
opt.prune_box(box, 1)
print("num reachable leafs after prune", opt.g1.num_vertices())

opt.steps(2000)

print((opt.num_solutions(), opt.num_rejected(), opt.num_candidate_cliques(), opt.num_steps()[1]))

points = []
for sol in opt.solutions():
    intervals = opt.solution_to_intervals(sol, 4)[1]

    # Use z3 to find specific values in the ranges that conform
    xv, yv, nv, sv = z3.Reals("xv yv nv sv")
    solver = z3.Solver()
    solver.add(xv >= np.nan_to_num(intervals[0][0]))
    solver.add(xv < np.nan_to_num(intervals[0][1]))
    solver.add(yv >= np.nan_to_num(intervals[1][0]))
    solver.add(yv < np.nan_to_num(intervals[1][1]))
    solver.add(nv >= np.nan_to_num(intervals[2][0]**2))
    solver.add(nv < np.nan_to_num(intervals[2][1]**2))
    solver.add(sv >= np.nan_to_num(intervals[3][0]))
    solver.add(sv < np.nan_to_num(intervals[3][1]))
    solver.add(nv == xv*xv + yv*yv)
    solver.add(sv == xv + yv)
    if solver.check() != z3.sat:
        print(intervals)
        print(" ==> UNSAT", sol.output1)
        continue
    model = solver.model()
    
    xv = float(model[xv].as_fraction())
    yv = float(model[yv].as_fraction())
    nv = np.sqrt(xv**2+yv**2)
    sv = xv + yv
                
    print("+", np.round([xv, yv, nv, sv],4), "->", sol.output1)
    if intervals[2][0] > nv or intervals[2][1] < nv:
        print("ERR norm", nv, intervals[2])
    if intervals[3][0] > sv or intervals[3][1] < sv:
        print("ERR sum", sv, intervals[3])
        
    points.append([xv, yv, nv, sv])

points = np.array(points)
print(points)
bst.predict(xgb.DMatrix(points), output_margin=True)

