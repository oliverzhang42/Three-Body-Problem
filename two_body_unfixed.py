from copy import deepcopy
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

TIMESTEP = 0.0001 # Time in seconds per timestep.
G = 0.1 # Approximation to see how it works.
SECONDS = 30

def step(state, masses, timestep):
    new_state = deepcopy(state)

    pos = state[0]
    vel = state[1]

    r01 = pos[1] - pos[0]

    a0 = masses[1] * r01 / (norm(r01)**3)
    a0 *= G
    a1 = masses[0] * (-r01) / (norm(r01)**3)
    a1 *= G

    accel = np.array([a0, a1])

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
    
    masses = [1, 1]

    pos = np.array([[0, 0], [1, 0.2]]) #np.random.random((2,2))
    vel = np.array([[0, 0.5], [0.3, 0.2]]) #np.random.random((2,2))
    state = np.array([pos, vel]).astype('float64')

    history = []

    start = time.time()

    for i in range(int(SECONDS / TIMESTEP)):
        state = step(state, masses, TIMESTEP)
        history.append(state)

    history = np.array(history)

    print(time.time() - start)

    fig, ax = plt.subplots()
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)

    # plt.plot(history[:,0,0,0], history[:,0,0,1])
    # plt.plot(history[:,0,1,0], history[:,0,1,1])

    rel_x = history[:,0,1,0] - history[:,0,0,0]
    rel_y = history[:,0,1,1] - history[:,0,0,1]

    plt.plot(rel_x, rel_y)

    '''
    center_mass = (history[:,0,0,:] + history[:,0,1,:])/2
    rel_1 = history[:,0,0,:] - center_mass
    rel_2 = history[:,0,1,:] - center_mass

    plt.plot(rel_1[:,0], rel_1[:,1])
    plt.plot(rel_2[:,0], rel_2[:,1])
    '''

    plt.show()
