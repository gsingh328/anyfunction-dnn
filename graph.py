import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-1, 1, 2**8)
w = [0.2330, 0.0582, 0.0000, 0.0105]
y = np.piecewise(x, [
    np.logical_and(x >= -1, x < -0.5),
    np.logical_and(x >= -.5, x < 0),
    np.logical_and(x >= 0, x < 0.5),
    x >= 0.5],
    [
    lambda x: x * w[0],
    lambda x: x * w[1],
    lambda x: x * w[2],
    lambda x: x * w[3]
    ])

plt.plot(x, y)
plt.savefig("temp.png", dpi=400)
