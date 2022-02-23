import numpy as np

def foo(x):
    x_float = x / 2**nbits
    y_float = (1 + np.tanh(4 * (2 * x_float - 1))) / 2
    y_int = y_float * 2**nbits
    return np.clip(np.round(y_int), 0, 2**nbits - 1)

nbits = 4

x = np.linspace(0,2**nbits - 1, num=2**nbits)

y = foo(x)
print(np.stack((x,y), axis=1))

# lut_foo = np.tile(x, (2**nbits, 1))
# print(lut_foo)
