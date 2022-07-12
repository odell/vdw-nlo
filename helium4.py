'''
  LM2M2-like vdW interaction for the 4He system
'''

import numpy as np
from mu2 import Mesh, NonlocalCounterterm, Interaction, System

FACTOR = 12.11928 #  K.Å^2
MASS = 1.0/FACTOR # 1/(K•Å^2)

# From the LM2M2 potential...
HBAR = 1.0
KB = 1.0
c6 = 1.34687065
RM = 2.9695 #  Å
EPS_OVER_KB = 10.97/FACTOR # 1/Å^2
EPS = EPS_OVER_KB * KB #  1/Å^2
C6 = EPS * RM**6 * c6 * FACTOR # K•Å^6
BETA6 = (MASS*C6)**0.25 # Å
RMESH = Mesh(0, 20*BETA6, 3000)

B2 = 1.31e-3 # 2-body binding energy, K
A0 = 100.0 # scattering length, Å
R0 = 7.33 # effective range, Å

N1 = 2
N2 = 8
NCT = 6

def local_reg(r, R):
    return (1 - np.exp(-(r/R)**N1))**N2


def fr(r):
    return -C6/r**6


def long_range_potential(r, R):
    return local_reg(r, R) * fr(r)


def nonlocal_regulator(q, L, n, l):
    return (q/L)**l*np.exp(-(q/L)**n)


def nonlocal_lo_term(p, k, L, n, l):
    return nonlocal_regulator(p, L, n, l) * nonlocal_regulator(k, L, n, l)


def nonlocal_nlo_term(p, k, L, n, l):
    return ((p/L)**2 + (k/L)**2)/2 * nonlocal_lo_term(p, k, L, n, l)


def construct_helium4_system(R, ell, nq=200):
    qmesh = Mesh(0, 10*2/R, nq)

    lo_xterm = NonlocalCounterterm(
        lambda p, k, L: nonlocal_lo_term(p, k, L, NCT, ell),
        lambda p, k, L: nonlocal_nlo_term(p, k, L, NCT, ell),
        lambda p, k, L: nonlocal_regulator(p, L, NCT, ell) * nonlocal_regulator(k,
            L, NCT, ell),
        qmesh, R, ell
    )

    interaction = Interaction(
        long_range_potential,
        lo_xterm,
        RMESH
    )

    return System(interaction, MASS/2, ell)
