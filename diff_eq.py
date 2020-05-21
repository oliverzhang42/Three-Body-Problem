import numpy as np
import matplotlib.pyplot as plt

TIMESTEP = 0.01
TOTAL = 100
k = 1

time = [0]
disp = [5]
vel = [0]
accel = [0]

for i in range(int(TOTAL/TIMESTEP)):
    time.append(time[-1] + TIMESTEP)
    a = -k / (disp[-1]**2)

    accel.append(a)
    vel.append(vel[-1] + accel[-1] * TIMESTEP)
    disp.append(disp[-1] + vel[-1] * TIMESTEP)

    if vel[-1] < -5:
        break

print(vel)

plt.plot(time, disp)
plt.show()
