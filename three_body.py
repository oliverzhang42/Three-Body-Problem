from copy import deepcopy
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

TIMESTEP = 0.00001 # Time in seconds per timestep.
G = 0.1 # Approximation to see how it works.
SECONDS = 30

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

    masses = [1, 1, 1]

    pos = np.random.random((3,2))
    vel = np.random.random((3,2))
    state = np.array([pos, vel]).astype('float64')

    history = []

    start = time.time()

    for i in range(int(SECONDS / TIMESTEP)):
        state = step(state, masses, TIMESTEP)
        history.append(state)

    history = np.array(history)

    print(time.time() - start)

    fig, ax = plt.subplots()

    plt.plot(history[:,0,0,0], history[:,0,0,1])
    plt.plot(history[:,0,1,0], history[:,0,1,1])
    plt.plot(history[:,0,2,0], history[:,0,2,1])
    #ani = FuncAnimation(fig, update, frames=history, blit=False, interval=50)
    plt.show()
