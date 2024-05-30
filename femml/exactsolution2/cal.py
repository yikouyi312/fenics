import numpy as np
beta = 11000000000#6600000
dt = 0.05
epsilon = 1e-3
test = beta * np.power(dt, 7.0 / 4.0) * np.power(epsilon, 5.0 / 4.0)
print(test)
mu = 22
test1 = mu * dt
print(test1)
test2 = beta * np.power(dt, 7.0 / 4.0) / np.power(epsilon, 3.0 / 4.0)
print(test2)