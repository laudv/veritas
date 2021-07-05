import datasets
import veritas

import matplotlib.pyplot as plt
import numpy as np

# Laad de dataset (je kan kijken naar tests/experiments/datasets.py)
num_classes = 10
mnist = datasets.Mnist()
mnist.load_dataset()
mnist.load_model(num_trees=100, tree_depth=5)
X, y = mnist.X, mnist.y # X is een pandas DataFrame

# kies een willekeurige example uit de dataset waarvoor je een adversarial
# example wil vinden
example_i = 120
example = list(X.iloc[example_i, :])
example_label = int(y[example_i])

at = mnist.at[example_label] # ensemble voor label van example
max_memory = 4*1024*1024*1024 # 4 GiB
opt = veritas.Optimizer(minimize=at, max_memory=max_memory) # laat Veritas at minimalizeren

# nu moeten we nog zorgen dat Veritas enkel zoekt rond 'example'
delta = 15 # maximale l-inf afstand tot 'example'
print("number of vertices before pruning:", opt.g0.num_vertices()) # g0 is de search graph voor geminimaliseerde at
opt.prune_example(example, delta)
print("number of vertices after pruning:", opt.g0.num_vertices())

# laat Veritas zoeken met een gerelaxeerde search
max_time = 10.0 # seconden
opt.arastar(max_time, eps_start=0.01, eps_incr=0.1, max_num_steps=100)

if opt.num_solutions() > 0:
    print(opt.num_solutions(), "solutions!")

    adv_example = opt.get_closest_example(opt.solutions()[-1], example)

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
    ax0.set_title(f"the original example\nwith label {example_label} and output {at.predict_single(example):.3g}")
    ax0.imshow(np.array(example).reshape((28, 28)), cmap="binary")

    ax1.set_title(f"the adversarial example\nwith output {at.predict_single(adv_example):.3g}")
    ax1.imshow(np.array(adv_example).reshape((28, 28)), cmap="binary")

    ax2.set_title(f"difference")
    im = ax2.imshow((np.array(example)-np.array(adv_example)).reshape((28, 28)), cmap="binary")
    fig.colorbar(im, ax=ax2)

    plt.show()

else:
    print("no adversarial examples found")
