import sys
from gao_sols import *
# import matplotlib.pyplot as plt

argc = len(sys.argv)
l = int(sys.argv[1]) if argc > 1 else 0

deltac = [9.654418e-2, 1.473792e-1, 4.306921e-1, 1.580826, 2.073296]
Dmin = 0.01
Dmax = 400

x = np.concatenate((
    np.logspace(np.log10(0.0001), np.log10(1), 50),
    np.linspace(1.1, 10, 100),
    np.linspace(10.1, 400, 300)
))

xi = np.array([x[0]])
y = np.array([])
y = np.append(y, newton(lambda i: charFunc([i, 0], l, x[0]), nu0(l), maxiter=1000))
print(x[0], y[0])
for i in range(1, len(x)):
    Del = x[i]
    if Del < deltac[l]:
        root = findRoot(l, Del, y[-1].real)
        xi = np.append(xi, Del)
        y = np.append(y, root)
        print(Del, root)
    else:
        try:
            root = findRoot(l, Del, y[-1].imag+0.05)
            print(Del, root)
            xi = np.append(xi, Del)
            y = np.append(y, root.real + np.abs(root.imag)*1j)
        except:
            print('Didn\'t converge.')
            pass

file = open('datfiles/roots%d.dat' % (l), 'w')
for (d, n) in zip(xi, y):
	file.write('%.8e\t%.8e+%.8ej\n' % (d, np.real(n), np.abs(np.imag(n))))
file.close()
#xin = np.array(xi)
#yn = np.array(y)
#data = np.zeros((np.size(xin), 2))+np.zeros((np.size(xin), 2))*1j
#data[:, 0] = xin
#data[:, 1] = yn
#np.savetxt('roots%d.dat' % (l), data)
# plt.plot(xi, np.imag(y))
# plt.show()
