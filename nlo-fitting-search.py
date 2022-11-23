#!/usr/bin/env python
import numpy as np
from scipy import optimize
import helium4plus as he4plus


# Read in the LO flow
Rs_plus, lo_plus_gs = np.loadtxt('datfiles/he4plusplus_LO_semilocal_2_6_6_rg_flow.txt', unpack = True)
nRplus = Rs_plus.size

g_plus_los = np.empty(nR)
g_plus_nlos = np.empty(nR)

# binding energies
b2s_plus = np.empty(nRplus)

# momentum grid
momenta = np.linspace(0.01/he4.BETA6, 0.3/he4.BETA6, 30)

# scale to render g dimensionless
X = he4.FACTOR * he4.BETA6 # K•Å^3

# kcotd delta returned from a and r
KCD = -1/he4plus.A0 + he4plus.R0/2*momenta**2

# First, we will tune k(cot(delta)). Then we'll go back and refine a_0 and r_0.
def kcd_diff(gpair, sys):
    kcd = sys.kcotd_gen_fast(momenta, *gpair)
    return np.sum(((KCD - kcd)/KCD)**2)


def inverse_a0_diff(gi, gj, sys):
    a0, _ = sys.a0_and_r0(gi, gj, momenta, use_c=True)
    return (1/he4.A0 - 1/a0)*he4.A0

def a0_diff(gi, gj, sys):
    a0, _ = sys.a0_and_r0(gi, gj, momenta, use_c=True)
    return (he4.A0 - a0)/he4.A0

# Let us assume we want to work through the whole lo rg file

for i in range(Rs_plus):
    R = Rs_plus[i]
    s = he4.Helium4System(R, 0)
    xhi = -0.1
    xlo = -200.
    g_nlo_test = np.linspace(xhi, xlo, 1400)*X
    test_results = []
    glo_refit = []
    guess = lo_plus_gs[NR]
    for gj in g_nlo_test:
        result = optimize.fsolve(a0_diff, guess, args=(gj, s),xtol=1e-14)
        atemp, rtemp = s.a0_and_r0(result[0], gj, momenta, use_c=True)
        test_results.append(s.a0_and_r0(result[0], gj, momenta, use_c=True))
        glo_refit.append(result[0])
        guess = result[0]
        print(guess)
