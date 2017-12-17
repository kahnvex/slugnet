import numpy as np
import matplotlib.pyplot as plt

z = np.arange(-4, 4, .01)
gz = 1 / (1 + np.exp(-z))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(z, gz)
ax.set_ylim([0.0, 1.0])
ax.set_xlim([-4.0, 4.0])
ax.grid(True)
ax.set_xlabel('z')
ax.set_ylabel('g(z)')
ax.set_title('Sigmoid')

plt.show()