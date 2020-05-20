from copy import deepcopy
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

TIMESTEP = 0.01 # Time in seconds per timestep.
G = 1 # Approximation to see how it works.
SECONDS = 10

'''
Just steps through a single timestep.

state: (numpy array) a 2x3x2 array containing the 2D position and velocity of
    each person. In particular, state[0] is all the positions and state[1] is
    all the velocities. Similarly, state[0][0] is the position of the 0th object
    and state[1][0] is the velocity of the smallest object.
masses: (list) a list of length 3 which contains the masses of each object.
timestep: (float) how long each timestep is in seconds.
'''
def step(state, masses, timestep):
    new_state = deepcopy(state)

    pos = state[0]
    vel = state[1]

    r01 = pos[1] - pos[0]
    r12 = pos[2] - pos[1]
    r20 = pos[0] - pos[2]

    a0 = masses[1] * r01 / (norm(r01)**3) + masses[2] * (-r20) / (norm(r20)**3)
    a0 *= G
    a1 = masses[0] * (-r01) / (norm(r01)**3) + masses[2] * r12 / (norm(r12)**3)
    a1 *= G
    a2 = masses[0] * r20 / (norm(r20)**3) + masses[1] * (-r12) / (norm(r12)**3)
    a2 *= G

    accel = np.array([a0, a1, a2])

    new_state[0, :] += vel * timestep
    new_state[1, :] += accel * timestep

    return new_state

def init():
    plt.xlim(2, 2)
    plt.ylim(-2, 2)

'''
Update updates the visuals after a single step.
'''
def update(state):
    plt.clf()
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    f = plt.scatter(state[0, :, 0], state[0, :, 1], c='green')
    return f

if __name__ == '__main__':

    masses = [1, 2, 4]

    pos = [[0, 0], [1, 0], [1, 1]]
    vel = [[1, 0], [1, 1], [0, 0]]
    state = np.array([pos, vel]).astype('float64')

    history = []

    for i in range(int(SECONDS / TIMESTEP)):
        state = step(state, masses, TIMESTEP)
        history.append(state)

    fig, ax = plt.subplots()
    ln = plt.scatter([], [])

    ani = FuncAnimation(fig, update, frames=history, blit=False, interval=50)
    plt.show()
