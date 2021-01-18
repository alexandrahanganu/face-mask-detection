import matplotlib.pyplot as plt
import numpy as np

data = [3297, 1158, 1230]
labels = ['MASK', 'NO MASK', 'WRONG MASK']

fig, ax = plt.subplots(1)
bar = ax.bar(labels, data)
bar[0].set_color('g')
bar[1].set_color('r')
bar[2].set_color('y')
plt.savefig('')
plt.show()
