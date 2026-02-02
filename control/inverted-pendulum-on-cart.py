# Python script

import sympy as sp
import numpy as np
import control as ct
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import PillowWriter

# block 1 - define constants for equation of motion
t, m, M, g, L, d, b = sp.symbols(r't m M g L \delta b')
x, the, u = sp.symbols(r'x \theta u', cls=sp.Function)

the = the(t)
the_d = sp.diff(the, t)
the_dd = sp.diff(the_d, t)

x = x(t)
x_d = sp.diff(x, t)
x_dd = sp.diff(x_d, t)

u = u(t)

# block 2 - define variables to calculate the coordinates of the system

# Pend up x_pend is - L * sp.sin(the) but in pend down it is + L * sp.sin(the)
x_pend = + L * sp.sin(the) + x
y_pend = - L * sp.cos(the)
x_pend_d = sp.diff(x_pend, t)
y_pend_d = sp.diff(y_pend, t)

# block 3 - generate code so that we can numerically calculate the coordinates
xpendf = sp.lambdify((the, x, L), x_pend)
ypendf = sp.lambdify((the, x, L), y_pend)

def get_coords(x, the, R):
    return x, np.zeros(x.shape), xpendf(the, x, R), ypendf(the, x, R)

# block 4 - calculate energy of the system T and V
T = 1/2 * M * x_d ** 2 + 1/2 * m * (x_pend_d ** 2 + y_pend_d**2)

y_ground = L + y_pend

V = m * g * y_ground

# calculate the lagrange equation
Le = T - V

# solve the langrange equation to calculate the equation of motion
W_x = sp.diff(sp.diff(LE, x_d), t) - sp.diff(LE, x)
W_the = sp.diff(sp.diff(LE, the_d), t) - sp.diff(LE, the)

W_x2 = W_x + d * x_d - u
W_the2 = W_the

sols = sp.solve([W_x2, W_the2], [x_dd, the_dd])

# linearise the dynamics around the pendulum up position
S = sp.Matrix([
    x,
    x_d,
    the,
    the_d,
])

f = sp.Matrix([
    x_d,
    sols[x_dd],
    the_d,
    sols[the_dd]
])
f.simplify()

A = f.jacobian(S).subs({the: sp.pi})
A.simplify()

# Define B and calculate control matrix K 
B = sp.Matrix([
    0,
    1 / M,
    0,
    1 / (M * L)
])

values = {M: 5, m: 1, L: 2, g: 10, d: 1}

A_up = A.subs(values)
B_up = B.subs(values)

A_up2 = np.array(A_up, dtype=np.float64)
B_up2 = np.array(B_up, dtype=np.float64)

p = [-0.3, -0.4, -0.5, -0.6]
p = [-1, -1.1, -1.2, -1.3]  # working but a bit slow getting there
p = [-3, -3.1, -3.2, -3.3]  # to aggresive and blows up
K = ct.place(A_up2, B_up2, p)

# from equation generate numerically functions we can simulate 
args = [*list(S), u]
f_lambda = sp.lambdify(args, f.subs(values))

def dSdt(S, t, uf):    
    u = uf(S)

    return f_lambda(*S, u).T[0]

t = np.linspace(0, 10, 1000)

y0 = [-1, 0, np.pi * 0.9, 0]

wr = np.array([0, 0, np.pi, 0])
uf = lambda S: np.dot(-K, (S - wr))[0]

ans = odeint(dSdt, y0=y0, t=t, args=(uf,))
cart_x, cart_y, pend_x, pend_y = get_coords(ans.T[0], ans.T[2], 2)

# Plot

def animate(i):
    ln1.set_data([cart_x[i], pend_x[i]], [cart_y[i], pend_y[i]])

fig, ax = plt.subplots(1,1, figsize=(8,8))
ax.grid()
ln1, = plt.plot([], [], 'ro--', lw=3, markersize=8)
ax.set_ylim(-2, 2)
ax.set_xlim(-2,2)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ani = animation.FuncAnimation(fig, animate, frames=1000, interval=50)
plt.show()
