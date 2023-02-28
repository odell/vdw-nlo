import sys
from gao_sols import *
# import matplotlib.pyplot as plt

argc = len(sys.argv)
l = int(sys.argv[1]) if argc > 1 else 0

HBAR = 1.0
KB = 1.0
c6 = 1.34687065
RM = 2.9695 #  Å
EPS_OVER_KB = 10.97/FACTOR # 1/Å^2
EPS = EPS_OVER_KB * KB #  1/Å^2
C6 = EPS * RM**6 * c6 * FACTOR # K•Å^6

data = np.loadtxt('datfiles/roots%d.dat' % (l), dtype=complex)
Deltas = np.real(data[:, 0])
roots = data[:, 1]

chivals = np.array([])
for (D, nu) in zip(Deltas, roots):
    x = chi(-D, l, nu, C6)
    print(-D, x)
    chivals = np.append(chivals, x)

file = open(f'datfiles/chi{l:d}.dat', 'w')
for (d, x) in zip(Deltas, chivals):
    file.write('%.8e\t%.8e\n' % (-d, np.real(x)))
file.close()

# plt.plot(-Deltas, np.real(chivals), label='real')
# plt.ylim([-10, 10])
# plt.plot(Deltas, np.imag(chivals), label='imag')
# plt.plot(Deltas, np.abs(chivals), label='abs')
# plt.legend()
# plt.show()
