'''
  LM2M2 interaction for the 4He system
'''

import numpy as np
from mu2 import Mesh, NonlocalCounterterm, Interaction, System

kB = 1.0
c = 1.0 
factor = 12.11928 #  K.Å^2
MASS = 1.0/factor #  K^{-1}.Å^{-2}
a = 100.2 #  Å
epsilon_over_k = 10.97 #  K
epsilon = epsilon_over_k * kB #  K
rm = 2.9695 #  Å
Astar = 1.89635353e5
alphastar = 10.70203539
betastar = -1.90740649
c6 = 1.34687065
c8 = 0.41308398
c10 = 0.17060159
D = 1.4088
Aa = 0.0026000000
x1 = 1.0035359490
x2 = 1.4547903690
B = 2.0*np.pi/(x2-x1)
r_min = 1.5
r_max = 100.0
C6 = epsilon * rm**6 * c6 # K•Å^6
BETA6 = (MASS*C6)**0.25
RMESH = Mesh(0, 20*BETA6, 3000)

def F(x):
    return 1.0  if x >= D else np.exp(-(D/x-1.0)*(D/x-1.0))


def Vastar(x):
    return Aa*(np.sin(B*(x-x1)-np.pi/2.0)+1.0) if (x1 <= x and x <= x2) else 0.0


def Vbstar(x):
    return Astar*np.exp(-alphastar*x + betastar*x**2) - (c6/x**6 + c8/x**8 +\
        c10/x**10) * F(x)


def V(r):
    x = r/rm
    return c * epsilon * (Vastar(x) + Vbstar(x))


h = 0.01
V_prime_r_min = (-V(r_min+2*h) + 8*V(r_min+h) - 8*V(r_min-h) + V(r_min-2*h)) / (12*h)
V_r_min = V(r_min)
kappa = V_prime_r_min / V_r_min

def V_hard_core_reg(r, R):
    # R is ignored
    if (r_min-r > 0):
        return V(r_min)*(2.0-np.exp(kappa*(r_min-r)))
    else:
        return V(r)


class LM2M2System(System):
    def __init__(self, R, ell, nq=200):
        qmesh = Mesh(0, 10*2/R, nq)

        xterm = NonlocalCounterterm(
            lambda p, k, L: 0,
            lambda p, k, L: 0,
            lambda p, k, L: 1,
            qmesh, R, ell
        )

        interaction = Interaction(
            V_hard_core_reg,
            xterm,
            RMESH
        )

        super().__init__(interaction, MASS/2, ell)
