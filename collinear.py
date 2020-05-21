from copy import deepcopy
import numpy as np
from numpy.linalg import norm
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

TIMESTEP = 0.1 # Time in seconds per timestep.
G = 1 # Approximation to see how it works.
SECONDS = 10
DIAMETER = 0.4 # Diameter of each point
DISPLAY_SIZE = 10

'''
Just steps through a single timestep.

state: (numpy array) a 2x3 array containing the 2D position and velocity of
    each person. In particular, state[0] is all the positions and state[1] is
    all the velocities. Similarly, state[0][0] is the position of the 0th object
    and state[1][0] is the velocity of the smallest object.
masses: (list) a list of length 3 which contains the masses of each object.
timestep: (float) how long each timestep is in seconds.
'''
def step(state, masses, timestep, length):
    new_state = deepcopy(state)

    pos = state[0]
    vel = state[1]

    r01 = pos[1] - pos[0]
    r12 = pos[2] - pos[1]
    r20 = pos[0] - pos[2]

    # Adding in collisions
    if abs(r01) < length:
        tmp = pos[0]
        pos[0] = pos[1]
        pos[1] = tmp

    if abs(r20) < length:
        tmp = pos[0]
        pos[0] = pos[2]
        pos[2] = tmp

    if abs(r12) < length:
        tmp = pos[1]
        pos[1] = pos[2]
        pos[2] = tmp

    a0 = masses[1] * r01 / (abs(r01)**3) + masses[2] * (-r20) / (abs(r20)**3)
    a0 *= G
    a1 = masses[0] * (-r01) / (abs(r01)**3) + masses[2] * r12 / (abs(r12)**3)
    a1 *= G
    a2 = masses[0] * r20 / (abs(r20)**3) + masses[1] * (-r12) / (abs(r12)**3)
    a2 *= G

    accel = np.array([a0, a1, a2])

    new_state[0, :] += vel * timestep
    new_state[1, :] += accel * timestep

    return new_state

def init():
    pass

'''
Update updates the visuals after a single step.
'''
def update(state):
    plt.clf()
    plt.xlim(-DISPLAY_SIZE, DISPLAY_SIZE)
    plt.ylim(-DISPLAY_SIZE, DISPLAY_SIZE)
    f = plt.scatter(state[0, 0], [0], c='green') 
    f = plt.scatter(state[0, 1], [0], c='blue')
    f = plt.scatter(state[0, 2], [0], c='red')
    
    return f

if __name__ == '__main__':
    import pudb; pudb.set_trace()

    masses = [1, 1, 1]

    pos = [0, 2, -1]
    vel = [0, 0, 0]
    state = np.array([pos, vel]).astype('float64')

    history = []

    for i in range(int(SECONDS / TIMESTEP)):
        state = step(state, masses, TIMESTEP, DIAMETER)
        history.append(state)

    history = np.array(history)
    print(history[:, 1, 2])

    fig, ax = plt.subplots()

    ani = FuncAnimation(fig, update, frames=history, blit=False, interval=50)
    plt.show()
