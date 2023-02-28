import numpy as np
import matplotlib.pyplot as plt
import cmath
from scipy.optimize import root, fsolve, minimize
from scipy.special import gamma, jv, yv

mu = 1
hbar = 1

nu0 = lambda l: (2 * l + 1) / 4
beta6 = lambda c6: (2 * mu * C6 / hbar ** 2) ** 0.25
Delta = lambda epsilon, C6: (1 / 16) * epsilon * beta6(C6)**2 * 2*mu/hbar**2

def Qbar(nu, l, delta, n):
	if n <= 0: return 1 / ((nu+1) * ((nu+1)**2 - nu0(l)**2))
	y = 1 / ((nu+1) * ((nu+1)**2 - nu0(l)**2) - delta**2 * Qbar(nu+1, l, delta, n-1))
	return y

def QbarConverged(nu, l, delta):
	max = 100
	n = 10
	val0 = Qbar(nu, l, delta, n)
	val1 = Qbar(nu, l, delta, n+1)
	diff = abs((val1-val0)) / val0
	eps = 1e-12
	while diff > eps:
		val0 = val1
		n += 1
		val1 = Qbar(nu, l, delta, n+1)
		diff = abs(val1 - val0) / val0
		if n > max:
			print("Max iterations reached.")
			break
	return val1

def Q(nu, l, delta, n):
	if n <= 0: return 1
	x = 1 / ((nu+1) * ((nu+1)**2 - nu0(l)**2) * (nu+2) * ((nu+2)**2 - nu0(l)**2))
	return 1 / (1 - delta**2 * x * Q(nu+1, l, delta, n-1))

def QConverged(nu, l, delta):
	max = 100
	n = 10
	val0 = Q(nu, l, delta, n)
	val1 = Q(nu, l, delta, n+1)
	diff = abs(val1 - val0) / val0
	eps = 1e-12
	while diff > eps:
		val0 = val1
		n += 1
		val1 = Q(nu, l, delta, n+1)
		diff = (val1-val0) / val0
		if n > max:
			print("Max iterations reached.")
			break
	return val1

def charFunc(nu_array, l, delta):
	nu = nu_array[0] + nu_array[1] * 1j
	y = (nu**2 - nu0(l)**2) - (delta**2 / nu) * (QbarConverged(nu, l, delta) - QbarConverged(-nu, l, delta))
	result = [y.real, y.imag]
	return result

def findRoot(l, delta):
	deltac = [9.654418e-2, 1.473792e-1, 4.306921e-1, 1.580826, 2.073296]
	if delta <= deltac[l]:
		y = fsolve(charFunc, [0.5, 0], args = (l, delta), xtol = 1e-10)
		return y[0]
	else:
		y = fsolve(charFunc, [0, 0.5], args = (l, delta), xtol = 1e-10)
		return abs(y[1])

def c(nu, j, l, delta):
	b_0 = 1
	nu_prime = nu
	while nu_prime <= nu + j - 1:
		b_0 *= QConverged(nu_prime, l, delta)
		nu_prime += 1
	return b_0

def b(j, l, delta, nu):
	numerator = (gamma(nu) * gamma(nu-nu0(l)+1) * gamma(nu+nu0(l)+1))
	denominator = gamma(nu+j) * gamma(nu-nu0(l)+j+1) * gamma(nu+nu0(l)+j+1)
	return (-delta)**j * (numerator / denominator) * c(nu, j, l, delta)

def bminus(j, l, delta, nu):
	numerator = (gamma(nu-j+1) * gamma(nu-nu0(l)-j) * gamma(nu+nu0(l)-j))
	denominator = gamma(nu+1) * gamma(nu-nu0(l)) * gamma(nu+nu0(l))
	return (-delta)**j * (numerator / denominator) * c(-nu, j, l, delta)

def X(epsilon, C6, l, nu):
	min = -10
	max = 10
	sum = 0
	delta = Delta(epsilon, C6)
	for m in range(min, max+1):
		if 2*m < 0:
			sum += (-1)**m * bminus(abs(2*m), l, delta, nu)
		else:
			sum += (-1)**m * b(2*m, l, delta, nu)
	return sum

def Y(epsilon, C6, l, nu):
	min = -10
	max = 10
	sum = 0
	delta = Delta(epsilon, C6)
	for m in range(min, max+1):
		if 2*m+1 < 0:
			sum += (-1)**m * bminus(abs(2*m+1), l, delta, nu)
		else:
			sum += (-1)**m * b(2*m+1, l, delta, nu)
	return sum

def alpha(epsilon, C6, l, nu):
	return np.cos(np.pi * (nu-nu0(l)) / 2) * X(epsilon, C6, l, nu) - np.sin(np.pi * (nu-nu0(l)) / 2) * Y(epsilon, C6, l, nu)

def beta(epsilon, C6, l, nu):
	return np.sin(np.pi * (nu-nu0(l)) / 2) * X(epsilon, C6, l, nu) + np.cos(np.pi * (nu-nu0(l)) / 2) * Y(epsilon, C6, l, nu)

def fbar(r, epsilon, C6, l, nu):
	sum = 0
	min = -10
	max = 10
	lvdw = beta6(C6)
	r_prime = r / lvdw
	delta = Delta(epsilon, C6)
	for m in range(min, 1):
		sum += bminus(-m, l, delta, nu) * np.sqrt(r) * jv(nu+m, 0.5 / r_prime**2)
	for m in range(0, max+1):
		sum += b(m, l, delta, nu) * np.sqrt(r) * jv(nu+m, 0.5 / r_prime**2)
	return sum

def gbar(r, epsilon, C6, l, nu):
	sum = 0
	min = -10
	max = 10
	lvdw = beta6(C6)
	r_prime = r / lvdw
	delta = Delta(epsilon, C6)
	for m in range(min, 1):
		sum += bminus(-m, l, delta, nu) * np.sqrt(r) * yv(nu+m, 0.5 / r_prime**2)
	for m in range(0, max+1):
		sum += b(m, l, delta, nu) * np.sqrt(r) * yv(nu+m, 0.5 / r_prime**2)
	return sum

def f0(r, epsilon, C6, l, nu):
	a = alpha(epsilon, C6, l, nu)
	b = beta(epsilon, C6, l, nu)
	return (1 / (a**2 + b**2)) * (a*fbar(r, epsilon, C6, l, nu) - b*gbar(r, epsilon, C6, l, nu))

def g0(r, epsilon, C6, l, nu):
	a = alpha(epsilon, C6, l, nu)
	b = beta(epsilon, C6, l, nu)
	return (1 / (a**2 + b**2)) * (b*fbar(r, epsilon, C6, l, nu) + a*gbar(r, epsilon, C6, l, nu))

l = 0
C6 = 33
#epsilon = -0.023452
epsilon = -0.02
nu = findRoot(l, Delta(epsilon, C6))

print(QbarConverged(0.2, 0, Delta(0.0001, 33)))
'''
def asymptotic_value(e):
	return f0(100, e[0], C6, l, findRoot(l, Delta(e[0], C6)))
sol = minimize(asymptotic_value, x0 = [-1]).x
print(sol)
'''
r = np.linspace(0.001, 10.0, 100)
psi = [f0(i, epsilon, C6, l, nu) for i in r]
plt.plot(r, psi)
plt.show()
