'''
  LM2M2-like vdW interaction for the 4He system
  * first plus => C_6 in increased such that r_0 is the LM2M2 r_0
  * second plus => Other regulator powers (hardness) and schemes (semi-local).
'''

import numpy as np
from mu2 import Mesh, NonlocalCounterterm, Interaction, System

FACTOR = 12.11928 #  K.Å^2
MASS = 1.0/FACTOR # 1/(K•Å^2)

# From the LM2M2 potential...
BETA6 = 5.54125
C6 = BETA6**4/MASS
RMESH = Mesh(0, 20*BETA6, 3000)

B2 = 1.31e-3 # 2-body binding energy, K
A0 = 100.0 # scattering length, Å
R0 = 7.33 # effective range, Å

def local_reg(r, R, n1, n2):
    return (1 - np.exp(-(r/R)**n1))**n2


def fr(r):
    return -C6/r**6


def long_range_potential(r, R, n1, n2):
    return local_reg(r, R, n1, n2) * fr(r)


def nonlocal_regulator(q, L, n, l):
    return (q/L)**l*np.exp(-(q/L)**n)


def nonlocal_lo_term(p, k, L, n, l):
    return nonlocal_regulator(p, L, n, l) * nonlocal_regulator(k, L, n, l)


def nonlocal_nlo_term(p, k, L, n, l):
    return ((p/L)**2 + (k/L)**2)/2 * nonlocal_lo_term(p, k, L, n, l)


class SemiLocalHelium4System(System):
    def __init__(self, R, ell, n1, n2, nct, nq=200):
        qmesh = Mesh(0, 10*2/R, nq)

        xterm = NonlocalCounterterm(
            lambda p, k, L: nonlocal_lo_term(p, k, L, nct, ell),
            lambda p, k, L: nonlocal_nlo_term(p, k, L, nct, ell),
            lambda p, k, L: nonlocal_regulator(p, L, nct, 0) * nonlocal_regulator(k,
                L, nct, 0),
            qmesh, R, ell
        )

        interaction = Interaction(
            lambda r, R: long_range_potential(r, R, n1, n2),
            xterm,
            RMESH,
            scheme='semilocal'
        )

        super().__init__(interaction, MASS/2, ell)


class NonlocalHelium4System(System):
    def __init__(self, R, ell, n1, n2, nct, nq=200):
        qmesh = Mesh(0, 10*2/R, nq)

        xterm = NonlocalCounterterm(
            lambda p, k, L: nonlocal_lo_term(p, k, L, nct, ell),
            lambda p, k, L: nonlocal_nlo_term(p, k, L, nct, ell),
            lambda p, k, L: nonlocal_regulator(p, L, nct, 0) * nonlocal_regulator(k,
                L, nct, 0),
            qmesh, R, ell
        )

        interaction = Interaction(
            lambda r, R: long_range_potential(r, R, n1, n2),
            xterm,
            RMESH,
            scheme='nonlocal'
        )

        super().__init__(interaction, MASS/2, ell)
