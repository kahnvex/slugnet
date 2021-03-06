import numpy as np
import matplotlib.pyplot as plt

z = np.arange(-2, 2, .01)
phi_z = np.tanh(z)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(z, phi_z)
ax.set_ylim([-1.0, 1.0])
ax.set_xlim([-2.0, 2.0])
ax.grid(True)
ax.set_xlabel('z')
ax.set_ylabel('phi(z)')
ax.set_title('Hyperbolic Tangent')

plt.show()