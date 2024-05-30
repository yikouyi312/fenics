import numpy as np
beta = 550000
dt = 0.025
epsilon = 1e-3
test = beta * np.power(dt, 7.0 / 4.0) * np.power(epsilon, 3.0 / 4.0)
print(test)
mu = 10.0
test1 = mu * dt
print(test1)