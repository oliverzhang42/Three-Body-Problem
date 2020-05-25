from copy import deepcopy
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

TIMESTEP = 0.0001 # Time in seconds per timestep.
G = 0.1 # Approximation to see how it works.
SECONDS = 10
DISPLAY_SIZE = 2

def step(state, masses, timestep):
    m1 = masses[0]
    m2 = masses[1]

    x = state[0]
    y = state[1]
    vx = state[2]
    vy = state[3]

    a = -(G * m1 + G * m2) / (x**2 + y**2)

    ax = a * x / np.sqrt(x**2 + y**2)
    ay = a * y / np.sqrt(x**2 + y**2)

    vx += ax * timestep
    vy += ay * timestep

    x += vx * timestep
    y += vy * timestep

    new_state = [x, y, vx, vy]

    return new_state

def init():
    pass

'''
Update updates the visuals after a single step.
'''

if __name__ == '__main__':
    plt.xlim(-DISPLAY_SIZE, DISPLAY_SIZE)
    plt.ylim(-DISPLAY_SIZE, DISPLAY_SIZE)
    masses = [1, 1]

    state = np.array([1, 0.2, 0.3, -0.3]).astype('float64')

    history = []

    for i in range(int(SECONDS / TIMESTEP)):
        state = step(state, masses, TIMESTEP)
        history.append(state)

        if state[1] < 0.001 and state[1] > -0.001:
            print(state[0])

    history = np.array(history)

    print(history)

    plt.plot(history[:, 0], history[:, 1])
    plt.show()
