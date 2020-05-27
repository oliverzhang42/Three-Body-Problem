from copy import deepcopy
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

TIMESTEP = 0.0001 # Time in seconds per timestep.
G = 0.1 # Approximation to see how it works.
SECONDS = 20
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


def f(state, masses, timestep, converge=15):
    assert abs(state[1]) < 0.001

    while(state[1] >= 0):
        if abs(state[0]) > converge or abs(state[1]) > converge:
            return np.array([0,0,0,0])
        
        state = step(state, masses, timestep) 

    return state

def init():
    pass

'''
Update updates the visuals after a single step.
'''

if __name__ == '__main__':
    # Regular Stuff

    '''
    plt.xlim(-DISPLAY_SIZE, DISPLAY_SIZE)
    plt.ylim(-DISPLAY_SIZE, DISPLAY_SIZE)
    masses = [1, 1]

    state = np.array([1, 0, 0.3, -0.3]).astype('float64')
    s1 = deepcopy(state)

    history = []

    for i in range(int(SECONDS / TIMESTEP)):
        state = step(state, masses, TIMESTEP)
        history.append(state)

    history = np.array(history)

    plt.plot(history[:, 0], history[:, 1])
    plt.show()
    '''

    # Lyapunov calculations
  
    import pudb; pudb.set_trace() 
    timestep = TIMESTEP
    masses = [1, 1, 1] 
    test_runs = 100
    runs = 0
    sum_lyapunov = [0, 0, 0] #x, vx, vy

    for i in range(test_runs):
        print(i)
        state = np.random.random((4,))
        state[1] = 0

        try1 = f(state, masses, timestep)
        if try1[0] == 0:
            continue

        runs += 1
        h = 0.01

        perturbed1 = state + [h, 0, 0, 0]
        perturbed2 = state + [0, 0, h, 0]
        perturbed3 = state + [0, 0, 0, h]
        
        newstate0 = f(f(state, masses, timestep), masses, timestep)
        newstate1 = f(f(perturbed1, masses, timestep), masses, timestep)
        newstate2 = f(f(perturbed2, masses, timestep), masses, timestep)
        newstate3 = f(f(perturbed3, masses, timestep), masses, timestep)

        lyapunov1 = (newstate1[0] - newstate0[0])/h
        lyapunov2 = (newstate2[2] - newstate0[2])/h
        lyapunov3 = (newstate3[3] - newstate0[3])/h

        sum_lyapunov[0] += lyapunov1
        sum_lyapunov[1] += lyapunov2
        sum_lyapunov[2] += lyapunov3

    avg = np.array(sum_lyapunov) / runs
    print(avg)
    print(runs)
