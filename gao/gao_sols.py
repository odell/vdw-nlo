import numpy as np
from scipy.optimize import fsolve, newton
from scipy.special import gamma

FACTOR = 12.11928 #  K.Å^2
MASS = 1.0 / FACTOR # 1/(K•Å^2)
mu = MASS / 2.0
hbar = 1
nu0 = lambda l: (2 * l + 1) / 4
beta6 = lambda C6: (2*mu*C6 / hbar**2) ** 0.25
Delta = lambda epsilon, C6: (1 / 16) * epsilon * beta6(C6)**2 * 2*mu/hbar**2 
energy = lambda Del, C6: 16*Del*hbar**2 / (2*mu*beta6(C6)**2)

def p(nu, Del, l):
    return -Del**2 / ((nu+1) * ((nu+1)**2 - nu0(l)**2) * (nu+2) * ((nu+2)**2 - nu0(l)**2))

def QConv(nu, l, Del):
    tiny = 1e-30
    Nmax = 1000
    b0 = 0
    f0 = b0 if b0 != 0 else tiny 
    b = 1
    C0 = f0
    D0 = 0
    factor = 2
    j = 1
    while np.abs(factor - 1) > 1e-16:
        a = p(nu+j-2, Del, l) if j > 1 else 1
        D = b + a*D0
        if D == 0:
            D = tiny 
        D = 1/D
        C = b + a/C0
        if C == 0:
            C = tiny 
        factor = C*D
        f = f0*factor
        D0 = D
        C0 = C
        f0 = f
        j += 1
        if j > Nmax:
            break
    return f

def QbarConv(nu, l, Del):
    tiny = 1e-30
    Nmax = 1000
    b0 = 0
    f0 = b0 if b0 != 0 else tiny
    C0 = f0
    D0 = 0
    factor = 2
    j = 1
    while abs(factor-1) > 1e-16:
        a = -Del**2 if j > 1 else 1
        b = (nu+j)*((nu+j)**2 - nu0(l)**2) 
        D = b + a*D0
        D = 1/D if D != 0 else 1/tiny
        C = b + a/C0
        if C == 0:
            C = tiny
        factor = C*D
        f = f0*factor
        D0 = D
        C0 = C
        f0 = f
        j += 1
        if j > Nmax:
            print('QbarConv didn\'t converge.')
            break
    return f

def Q(nu, l, k):
  tiny = 1e-30
  Nmax = 10000
  b0 = 0
  f0 = b0 if b0 != 0 else tiny
  C0 = f0
  D0 = 0
  factor = 2
  j = 1
  while abs(factor-1) > 1e-15:
    aj = -k**2 / (16 * (nu+1) * ((nu+1)**2 - nu0(l)**2) * (nu+2) * ((nu+2)**2 - nu0(l)**2)) if j > 1 else 1
    bj = 1
    Dj = bj + aj*D0
    if Dj == 0:
      Dj = tiny
    Cj = bj + aj/C0
    if Cj == 0:
      Cj = tiny
    Dj = 1/Dj
    factor = Cj*Dj
    f = f0*factor 
    D0 = Dj
    C0 = Cj
    f0 = f
    j += 1
    if j > Nmax:
      print('Q did not converge for nu = ', nu, 'and Delta = ', k/4, '.')
      break
  return f

def A10(nu_array, l, k):
	nu = nu_array[0] + nu_array[1]*1j
	return (nu**2 - nu0(l)**2) - Q(-nu, l, k)/(256*nu*(nu-1)*((nu-1)**2 - nu0(l)**2)) * k**4 - Q(nu, l, k)/(256*nu*(nu+1)*((nu+1)**2 - nu0(l)**2)) * k**4 

def root(l, k, guess):
	deltac = [9.654418e-2, 1.473792e-1, 4.306921e-1, 1.580826, 2.073296]
	rootReal = l/2 if l%2==0 else (l-1)/2 + 1
	m = 10000
	if k < 4*deltac[l]:
		z = newton(lambda i: A10([i, 0], l, k), guess, maxiter=m)
		return z + 0j
	else:
		z = newton(lambda i: A10([rootReal, i], l, k), guess, maxiter=m)
		return rootReal + z*1j

def charFunc(nu_array, l, delta):
	nu = nu_array[0] + nu_array[1] * 1j
	y = (nu**2 - nu0(l)**2) - (delta**2 / nu) * (QbarConv(nu, l, delta) - QbarConv(-nu, l, delta))
	return np.abs(y) 

def findRoot(l, delta, start):
    deltac = [9.654418e-2, 1.473792e-1, 4.306921e-1, 1.580826, 2.073296]
    rootReal = l/2 if l%2==0 else (l-1)/2 + 1
    m = 10000
    if delta < deltac[l]:
        z = newton(lambda i: charFunc([i, 0], l, delta), start, maxiter=m)
        return z + 0j
    else:
        z = newton(lambda i: charFunc([rootReal, i], l, delta), start, maxiter=m, tol=1e-6)
        #z = fsolve(lambda i: charFunc([rootReal, i[0]], l, delta), [start], factor=1)[0]
        return rootReal + z.real*1j

def cj(nu, j, l, delta):
    b_0 = 1
    for i in range(0, j):
        b_0 *= QConv(nu+i, l, delta)
    return b_0

def bj(j, l, delta, nu):
    if j > 0:
        numerator = (gamma(nu) * gamma(nu-nu0(l)+1) * gamma(nu+nu0(l)+1))
        denominator = gamma(nu+j) * gamma(nu-nu0(l)+j+1) * gamma(nu+nu0(l)+j+1)
        return (-delta)**j * (numerator / denominator) * cj(nu, j, l, delta)
    else:
        numerator = (gamma(nu+j+1) * gamma(nu-nu0(l)+j) * gamma(nu+nu0(l)+j))
        denominator = gamma(nu+1) * gamma(nu-nu0(l)) * gamma(nu+nu0(l))
        return (-delta)**(-j) * (numerator / denominator) * cj(-nu, -j, l, delta)

def X(epsilon, C6, l, nu):
    delta = Delta(epsilon, C6)
    diff = 1
    last = 1e-30
    sum = 1
    i = 1
    while (diff > 1e-16):
        last2 = last 
        last = sum
        sum += (-1)**i*bj(2*i, l, delta, nu) + (-1)**(-i)*bj(-2*i, l, delta, nu)
        diff = max(np.abs((sum-last)/last), np.abs((last-last2)/last2))
        i += 1
    return sum

def Xtest(epsilon, C6, l, nu, n):
    delta = Delta(epsilon, C6)
    sum = 1
    for i in range(1, n):
        sum += (-1)**i*bj(2*i, l, delta, nu) + (-1)**(-i)*bj(-2*i, l, delta, nu)
    return sum

def Y(epsilon, C6, l, nu):
    delta = Delta(epsilon, C6)
    diff = 1
    last = 1e-30 
    sum = bj(1, l, delta, nu) 
    i = 1
    while (diff > 1e-16):
        last2 = last
        last = sum
        sum += (-1)**i*bj(2*i+1, l, delta, nu) + (-1)**(-i)*bj(-2*i+1, l, delta, nu)
        diff = max(np.abs((sum-last)/last), np.abs((last-last2)/last2))
        i += 1
    return sum

def alpha(epsilon, C6, l, nu):
    return np.cos(np.pi * (nu-nu0(l)) / 2) * X(epsilon, C6, l, nu) - np.sin(np.pi * (nu-nu0(l)) / 2) * Y(epsilon, C6, l, nu)

def beta(epsilon, C6, l, nu):
	return np.sin(np.pi * (nu-nu0(l)) / 2) * X(epsilon, C6, l, nu) + np.cos(np.pi * (nu-nu0(l)) / 2) * Y(epsilon, C6, l, nu)

def besselJ(v, x):
    C0 = (x/2)**v / gamma(v + 1)
    C = 0
    k = 1
    while (abs((C-C0)/C0) > 1e-16):
        C = C0
        C0 += (x/2)**v * (-1)**k / (gamma(k+1) * gamma(v+k+1)) * (x/2)**(2*k)
        k += 1
    return C0

def besselY(v, x):
    return (np.cos(v*np.pi)*besselJ(v, x) - besselJ(-v, x)) / np.sin(v*np.pi)

def fbar(r, epsilon, C6, l, nu):
    lvdw = beta6(C6)
    r_prime = r / lvdw
    delta = Delta(epsilon, C6)
    last = 1e-30
    tot = bj(0, l, delta, nu)
    j = 1
    while (np.abs((tot-last)/last) > 1e-16):
        last = tot
        tot += bj(j, l, delta, nu)*np.sqrt(r)*besselJ(nu+j, 0.5/r_prime**2) + bj(-j, l, delta, nu)*np.sqrt(r)*besselJ(nu-j, 0.5/r_prime**2)
        j += 1	
    return tot

def gbar(r, epsilon, C6, l, nu):
    lvdw = beta6(C6)
    r_prime = r / lvdw
    delta = Delta(epsilon, C6)
    last = 1e-30
    tot = bj(0, l, delta, nu)
    j = 1
    while (np.abs((tot-last)/last) > 1e-16):
        last = tot
        tot += bj(j, l, delta, nu)*np.sqrt(r)*besselY(nu+j, 0.5/r_prime**2) + bj(-j, l, delta, nu)*np.sqrt(r)*besselY(nu-j, 0.5/r_prime**2)
        j += 1
    return tot

def f0(r, epsilon, C6, l, nu):
    a = alpha(epsilon, C6, l, nu)
    b = beta(epsilon, C6, l, nu)
    return (1 / (a**2 + b**2)) * (a*fbar(r, epsilon, C6, l, nu) - b*gbar(r, epsilon, C6, l, nu))

def g0(r, epsilon, C6, l, nu):
    a = alpha(epsilon, C6, l, nu)
    b = beta(epsilon, C6, l, nu)
    return (1 / (a**2 + b**2)) * (b*fbar(r, epsilon, C6, l, nu) + a*gbar(r, epsilon, C6, l, nu))

def C(nu, l, Del):
    reldiff = 1
    count = 0
    last = 1e-30
    c = 1
    j = 0
    while (reldiff > 1e-16):
        last2 = last
        last = c
        c *= QConv(nu+j, l, Del) 
        reldiff = max(np.abs((c-last)/last), np.abs((last-last2)/last2))
        j += 1
    return c

def G(nu, l, Del):
    return np.abs(Del)**(-nu) * gamma(1+nu0(l)+nu)*gamma(1-nu0(l)+nu)/gamma(1-nu) * C(nu, l, Del)

def chi(Del, l, nu, c6):
    epsilon = energy(Del, c6)
    a = alpha(epsilon, c6, l, nu)
    b = beta(epsilon, c6, l, nu)
    Gplus = G(nu, l, Del)
    Gminus = G(-nu, l, Del)
    return ((a*np.sin(np.pi*nu) - b*np.cos(np.pi*nu))*Gminus + b*Gplus) / ((b*np.sin(np.pi*nu) + a*np.cos(np.pi*nu))*Gminus - a*Gplus)

def Zff(eps, l, nu, c6):
    D = Delta(eps, c6)
    a = alpha(eps, c6, l, nu)
    b = beta(eps, c6, l, nu)
    Gm = G(-nu, l, D)
    Gp = G(nu, l, D)
    x = X(eps, c6, l, nu)
    y = Y(eps, c6, l, nu)
    return ((x**2 + y**2)*np.sin(np.pi*nu))**(-1) * (-(-1)**l * \
        (a*np.sin(np.pi*nu) - b*np.cos(np.pi*nu))*Gm*np.sin(np.pi*nu - l*np.pi/2 - \
        np.pi/4) + b*Gp*np.cos(np.pi*nu - l*np.pi/2 - np.pi/4))

def Zfg(eps, l, nu, c6):
    D = Delta(eps, c6)
    a = alpha(eps, c6, l, nu)
    b = beta(eps, c6, l, nu)
    Gm = G(-nu, l, D)
    Gp = G(nu, l, D)
    x = X(eps, c6, l, nu)
    y = Y(eps, c6, l, nu)
    return ((x**2 + y**2)*np.sin(np.pi*nu))**(-1) * (-(-1)**l * \
        (a*np.sin(np.pi*nu) - b*np.cos(np.pi*nu))*Gm*np.cos(np.pi*nu - l*np.pi/2 - \
        np.pi/4) + b*Gp*np.sin(np.pi*nu - l*np.pi/2 - np.pi/4))

def Zgf(eps, l, nu, c6):
    D = Delta(eps, c6)
    a = alpha(eps, c6, l, nu)
    b = beta(eps, c6, l, nu)
    Gm = G(-nu, l, D)
    Gp = G(nu, l, D)
    x = X(eps, c6, l, nu)
    y = Y(eps, c6, l, nu)
    return ((x**2 + y**2)*np.sin(np.pi*nu))**(-1) * (-(-1)**l * (b*np.sin(np.pi*nu) + a*np.cos(np.pi*nu))*Gm*np.sin(np.pi*nu - l*np.pi/2 - np.pi/4) - a*Gp*np.cos(np.pi*nu - l*np.pi/2 - np.pi/4))

def Zgg(eps, l, nu, c6): 
    D = Delta(eps, c6)
    a = alpha(eps, c6, l, nu)
    b = beta(eps, c6, l, nu)
    Gm = G(-nu, l, D)
    Gp = G(nu, l, D)
    x = X(eps, c6, l, nu)
    y = Y(eps, c6, l, nu)
    return ((x**2 + y**2)*np.sin(np.pi*nu))**(-1) * (-(-1)**l * (b*np.sin(np.pi*nu) + a*np.cos(np.pi*nu))*Gm*np.cos(np.pi*nu - l*np.pi/2 - np.pi/4) - a*Gp*np.sin(np.pi*nu - l*np.pi/2 - np.pi/4))
