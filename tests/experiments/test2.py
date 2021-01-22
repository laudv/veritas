import datasets
from veritas import Optimizer
from veritas import RobustnessSearch, VeritasRobustnessSearch, MergeRobustnessSearch
from treeck_robust import TreeckRobustnessSearch
from veritas.kantchelian import KantchelianAttack, KantchelianTargetedAttack
import numpy as np
import matplotlib.pyplot as plt

#calhouse = datasets.Calhouse()
#calhouse.load_model(10, 4)
#
#print(calhouse.at)

mnist = datasets.Mnist()
mnist.load_model(20, 4)
mnist.load_dataset()
example_i = 5
example = list(mnist.X.iloc[example_i,:])
example_label = int(mnist.y[example_i])

#opt = Optimizer(minimize=mnist.at[1], maximize=mnist.at[0],
#        matches=set(), match_is_reuse=False)
#dur, oom = opt.arastar_multiple_solutions(100, eps_start=1.0)
#
#for sol in opt.solutions():
#    print(sol)
#
#print("num_solutions", opt.num_solutions())
#print("bounds", opt.bounds)
#print("times", opt.times)



#for at in mnist.ats:
#    print("====================")
#    print(at)

at0 = mnist.at[example_label]
at1 = mnist.at[(example_label+1)%10]

actual_prediction0 = at0.predict_single(example)
actual_prediction1 = at1.predict_single(example)

ver = VeritasRobustnessSearch(at0, at1, example, start_delta=20,
        stop_condition=RobustnessSearch.INT_STOP_COND)
ver_norm, ver_lo, ver_hi = ver.search()
ver_example = ver.generated_examples[-1]

print("=================================================")

mer = MergeRobustnessSearch(at0, at1, example, 2, start_delta=20,
        stop_condition=RobustnessSearch.INT_STOP_COND)
mer_norm, mer_lo, mer_hi = mer.search()

print("=================================================")

tck = TreeckRobustnessSearch(at0, at1, example, start_delta=20,
        stop_condition=RobustnessSearch.INT_STOP_COND)
tck_norm, tck_lo, tck_hi = tck.search()
tck_example = tck.generated_examples[-1]

print("=================================================")

m = KantchelianTargetedAttack(at0, at1, example=example)
m.optimize()
adv_example, adv_prediction0, adv_prediction1, norm = m.solution()

print("=================================================")

print("Original prediction            ", actual_prediction0, actual_prediction1)
print("Veritas adv prediction check   ",
        at0.predict_single(ver_example),
        at1.predict_single(ver_example))
print("Veritas norm check             ", max(abs(x-y) for x, y in zip(ver_example, example)), ver_lo, ver_norm, ver_hi)
print("Merge norm                     ", mer_lo, mer_norm, mer_hi)
print("Treeck norm check              ", max(abs(x-y) for x, y in zip(tck_example, example)), tck_lo, tck_norm, tck_hi)
print("Treeck adv prediction check    ",
        at0.predict_single(tck_example),
        at1.predict_single(tck_example))
print("Treeck adv prediction check    ",
        tck.source_at.predict_single(tck_example),
        tck.target_at.predict_single(tck_example))

print("KantchelianAttack adv prediction      ", adv_prediction0, adv_prediction1)
print("KantchelianAttack adv prediction check",
        at0.predict_single(adv_example),
        at1.predict_single(adv_example))
print("KantchelianAttack reported norm", norm)
print("KantchelianAttack norm check   ", max(abs(x-y) for x, y in zip(adv_example, example)))

fig, ax = plt.subplots(2, 4)
ax[0][0].set_title("original")
ax[0][0].imshow(np.array(example).reshape((28,28)))
ax[1][0].set_title("Veritas vs MILP")
ax[1][0].imshow((np.array(ver_example)-np.array(adv_example)).reshape((28,28)))
ax[0][1].set_title("Veritas")
ax[0][1].imshow(np.array(ver_example).reshape((28,28)))
ax[1][1].imshow((np.array(example)-np.array(ver_example)).reshape((28,28)))
#ax2.imshow(np.array(exp.example)-np.array(adv_example).reshape((28,28)))
#ax[0][2].imshow(np.array(tck_example).reshape((28,28)))
ax[0][2].set_title("Treeck")
ax[0][2].imshow(np.array(tck_example).reshape((28,28)))
ax[1][2].imshow((np.array(example)-np.array(tck_example)).reshape((28,28)))
ax[0][3].set_title("MILP")
ax[0][3].imshow(np.array(adv_example).reshape((28,28)))
ax[1][3].imshow((np.array(example)-np.array(adv_example)).reshape((28,28)))
plt.show()
