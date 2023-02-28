from gao_sols import *
import matplotlib.pyplot as plt

l = 0
C6 = 1
deltac = [9.654418e-2, 1.473792e-1, 4.306921e-1, 1.580826, 2.073296]

roots = np.loadtxt('datfiles/roots0_smallDelta.dat', dtype=complex)
Deltas = np.real(roots[:, 0])
nus = roots[:, 1]

#index = 300
#d = Deltas[index]
#n = nus[index]
#val = X(energy(d, C6), C6, l, n)
#valtest = [Xtest(energy(d, C6), C6, l, n, i) - val for i in range(1, 20)]
#list(map(lambda i: print(i[0], i[1]), enumerate(valtest)))

# plotting Chi
chis = [chi(-d, l, n, C6) for (d, n) in zip(Deltas, nus)]
plt.plot(-Deltas, np.real(chis))
#plt.plot(Deltas, np.imag(chis))
plt.ylim([-10, 10])
plt.show()

# testing C(nu) = lim(j->inf) cj(nu)
# index = 50
# Delta = Deltas[index]
# nu = nus[index]
# print(Delta, nu)
# c = np.array([1])
# for i in range(1, 1000):
#     c = np.append(c, c[i-1]*QConv(nu+i-1, l, Delta))
# plt.plot(np.real(c))
# plt.plot(np.imag(c))
# plt.show()
# print(np.abs((c[-1]-C(nu, l, Delta))/C(nu, l, Delta)))
# 
#  finding the roots
# x = np.array([])
# nu = np.array([])
# x = np.append(x, ks[0])
# nu = np.append(nu, root(l, ks[0], nu0(l)))
# 
# for i in range(1, len(ks)):
# 	k = ks[i]
# 	if k < 4*deltac[l]:
# 		r = root(l, k, nu[-1].real)
# 		print(k/4, r)
# 	else:
# 		try:
# 			r = root(l, k, nu[-1].imag+0.1)
# 			print(k, r)
# 			nu = np.append(nu, r.real + np.abs(root.imag)*1j)	
# 			x = np.append(x, k)
# 		except:
# 			pass
# 
# plt.plot(d, nu)
# plt.show()
