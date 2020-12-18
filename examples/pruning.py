import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt

import veritas
import veritas.xgb

# Generate a random dataset
np.random.seed(14)
N = 2000
x = np.random.randint(0, 100, size=(N, 1)).astype(float)
y = np.random.randint(0, 100, size=(N, 1)).astype(float)
dist = np.sqrt(x**2 + y**2)
s = x + y
target = ((dist < 50) & (s > 20)) | ((x+2*y) > 200)

# Plot the dataset
#plt.plot(x[target], y[target], '.', color="blue")
#plt.plot(x[~target], y[~target], '.', color="red")
#plt.show()

X = np.concatenate((x, y), axis=1)

# Train a model using XGBoost
xtrain = xgb.DMatrix(X, label=target, missing=None)
params = {
    "learning_rate": 0.5,
    "max_depth": 4,
    "objective": "binary:hinge",
    "eval_metric": "error",
    "tree_method": "hist",
    "seed": 1,
    "nthread": 1,
}
bst = xgb.train(params, xtrain, 10, [(xtrain, "train")])

features = ["x", "y"]
feat2id = {f : i for i, f in enumerate(features)}
at = veritas.xgb.addtree_from_xgb_model(bst)
at.base_score = 0.5

# Check whether our "AddTree"'s predictions and XGBoost's match
pred_raw_at = np.array(at.predict(X))
pred_raw = bst.predict(xtrain, output_margin=True)
print("max error", max(pred_raw_at - pred_raw), "(should be no more than float32 rounding error)")

# Look in a 100×100 grid at the values produced by XGBoost
Xv = np.zeros((100*100, 2))
for i, xv in enumerate(range(100)):
    for j, yv in enumerate(range(100)):
        Xv[i*100+j, 0:2] = [xv, yv]
        
vs = bst.predict(xgb.DMatrix(Xv), output_margin=True)

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 6))
pred = (pred_raw.reshape((N,1)) > 0.0)
ax0.plot(x[pred&target], y[pred&target], '.', color="darkblue", alpha=0.5, label="true pos")
ax0.plot(x[~pred&~target], y[~pred&~target], '.', color="darkred", alpha=0.5, label="true neg")
ax0.plot(x[pred&~target], y[pred&~target], 'x', color="blue", label="false pos")
ax0.plot(x[~pred&target], y[~pred&target], 'x', color="red", label="false neg")
im = ax1.imshow(vs.reshape(100,100).T, origin="lower", cmap="Spectral")
fig.colorbar(im, ax=ax1)
plt.show()


# Use VERITAS to find the ordered possible outputs descendingly in box x: [25, 75], y: [50, 80]
opt = veritas.Optimizer(maximize=at)
box = [
    veritas.RealDomain(25, 75),
    veritas.RealDomain(50, 80),
]
print("num reachable leafs before prune", opt.g1.num_vertices())
opt.prune_box(box, 1)
print("num reachable leafs after prune", opt.g1.num_vertices())

opt.steps(2000)

print((opt.num_solutions(), opt.num_rejected(), opt.num_candidate_cliques(), opt.num_steps()[1]))

points = []
for sol in opt.solutions():
    # convert Solution object to list of intervals indexes by feature id
    intervals = opt.solution_to_intervals(sol, 4)[1]
    xv = sum(intervals[0])/2 # middle of first feature interval
    yv = sum(intervals[1])/2 # middle of second feature interval
    points.append([xv, yv, sol.output1])

points = np.array(points)
print(points)
#print(bst.predict(xgb.DMatrix(points), output_margin=True))

fig, ax = plt.subplots()
m, M = abs(min(points[:,2])), max(points[:,2])
colors = [[1 - (v+m)/(m+M), 0.0, (v+m)/(m+M)] for v in points[:,2]]
print(colors)
im = ax.imshow(vs.reshape(100,100).T, origin="lower", cmap="Spectral")
ax.scatter(points[:,0], points[:,1], c=colors, marker=".")
fig.colorbar(im, ax=ax)
plt.show()
