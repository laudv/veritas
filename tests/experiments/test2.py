import datasets
from veritas import Optimizer
from robust import RobustnessSearch, VeritasRobustnessSearch, MergeRobustnessSearch
from veritas.kantchelian import KantchelianAttack, KantchelianTargetedAttack
import numpy as np
import matplotlib.pyplot as plt

#calhouse = datasets.Calhouse()
#calhouse.load_model(10, 4)
#
#print(calhouse.at)

mnist = datasets.Mnist()
mnist.load_model(10, 4)
mnist.load_dataset()
example_i = 1
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

rob = VeritasRobustnessSearch(at0, at1, example, start_delta=20+1e-4,
        stop_condition=RobustnessSearch.INT_STOP_COND)
rob_norm, rob_lo, rob_hi = rob.search()
rob_example = rob.generated_examples[-1]
print("=================================================")

mer = MergeRobustnessSearch(at0, at1, example, 2, start_delta=20+1e-4,
        stop_condition=RobustnessSearch.INT_STOP_COND)
mer_norm, mer_lo, mer_hi = mer.search()
print("=================================================")

m = KantchelianTargetedAttack(at0, at1, example=example)
m.optimize()
adv_example, adv_prediction0, adv_prediction1, norm = m.solution()

print("Robust adv prediction check           ",
        at0.predict_single(rob_example),
        at1.predict_single(rob_example))
print("Robust norm check              ", max(abs(x-y) for x, y in zip(rob_example, example)), rob_norm)
print("Merge norm                     ", mer_norm)

print("KantchelianAttack act prediction      ", actual_prediction0, actual_prediction1)
print("KantchelianAttack adv prediction      ", adv_prediction0, adv_prediction1)
print("KantchelianAttack adv prediction check",
        at0.predict_single(adv_example),
        at1.predict_single(adv_example))
print("KantchelianAttack reported norm", norm)
print("KantchelianAttack norm check   ", max(abs(x-y) for x, y in zip(adv_example, example)))

#fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
#ax0.imshow(np.array(example).reshape((28,28)))
#ax1.imshow(np.array(adv_example).reshape((28,28)))
##ax2.imshow(np.array(exp.example)-np.array(adv_example).reshape((28,28)))
#ax2.imshow(np.array(rob_example).reshape((28,28)))
#plt.show()
