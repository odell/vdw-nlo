import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import newton, brentq
from gao_sols import *
import matplotlib.pyplot as plt
import sys

e = float(sys.argv[1])
c6 = float(sys.argv[2])
d = Delta(e, c6)

data = np.loadtxt('datfiles/chi.dat')
deltas = data[:, 0]
chis = data[:, 1]
chi = interp1d(deltas, chis)

K = chi(d)
print('K = ', K)
chi_shift = lambda i: chi(i) - K
brackets = np.array([])
chis_shifted = np.array([chi_shift(i) for i in deltas])
for i in range(1, len(chis_shifted)):
	x1 = chis_shifted[i]
	x0 = chis_shifted[i-1]
	if np.sign(x1) != np.sign(x0):
		if np.abs(x1-x0) < 10:
			brackets = np.append(brackets, (deltas[i], deltas[i-1]))
n = len(brackets)/2
brackets.resize((int(n), 2))
for pair in brackets:
	D = brentq(lambda i: chi(i)-K, pair[0], pair[1])
	print(energy(D, c6))

# energies = np.array([energy(d, c6) for d in deltas])
# plt.plot(energies, chis_shifted)
# plt.show()
