'''
Functions defined in Gao PRA 58 1728
'''

import numpy as np
from scipy.optimize import fsolve, newton
from scipy.special import gamma

def nu_0(ell):
    '''
    Defined above Eq. (13).
    '''
    return (2*ell+1) / 4


def beta_6(C6, mu):
    '''
    Defined below Eq. (1).
    Assumes hbar is 1.
    '''
    return (2*mu*C6)**0.25


def Delta(eps, C6, mu):
    '''
    Eq. (12)
    '''
    return 2*mu*eps/16 * beta_6(C6, mu)**2 


def energy(D, C6, mu):
    '''
    Defined implicitly in Eq. (1).
    '''
    return 16*D / (2*mu*beta_6(C6, mu)**2)


def p(nu, D, ell):
    '''
    I don't know what this is.
    '''
    return -D**2 / ((nu+1) * ((nu+1)**2 - nu_0(ell)**2) * (nu+2) * ((nu+2)**2 - nu_0(ell)**2))


def QConv(nu, ell, D):
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
        a = p(nu+j-2, D, ell) if j > 1 else 1
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


def QbarConv(nu, ell, D):
    tiny = 1e-30
    Nmax = 1000
    b0 = 0
    f0 = b0 if b0 != 0 else tiny
    C0 = f0
    D0 = 0
    factor = 2
    j = 1
    while abs(factor-1) > 1e-16:
        a = -D**2 if j > 1 else 1
        b = (nu+j)*((nu+j)**2 - nu_0(ell)**2) 
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


def Eq15(nu_array, ell, delta):
    '''
    Eq. (15)
    '''
    nu = nu_array[0] + nu_array[1] * 1j
    y = (nu**2 - nu_0(ell)**2) - (delta**2 / nu) * (QbarConv(nu, ell, delta) - QbarConv(-nu, ell, delta))
    return np.abs(y) 


DELTA_C = np.array([
    9.654418e-2,
    1.473792e-1,
    4.306921e-1,
    1.580826,
    2.073296
])

def find_nu(ell, delta, start):
    root_real = ell/2 if ell%2 == 0 else (ell-1)/2 + 1
    m = 10000
    if delta < DELTA_C[ell]:
        z = newton(lambda i: Eq15([i, 0], ell, delta), start, maxiter=m)
        return z + 0j
    else:
        z = newton(lambda i: Eq15([root_real, i], ell, delta), start, maxiter=m, tol=1e-6)
        #z = fsolve(lambda i: Eq15([rootReal, i[0]], ell, delta), [start], factor=1)[0]
        return root_real + z.real*1j


def cj(nu, j, ell, delta):
    b_0 = 1
    for i in range(0, j):
        b_0 *= QConv(nu+i, ell, delta)
    return b_0


def bj(j, ell, delta, nu):
    if j > 0:
        numerator = (gamma(nu) * gamma(nu-nu_0(ell)+1) * gamma(nu+nu_0(ell)+1))
        denominator = gamma(nu+j) * gamma(nu-nu_0(ell)+j+1) * gamma(nu+nu_0(ell)+j+1)
        return (-delta)**j * (numerator / denominator) * cj(nu, j, ell, delta)
    else:
        numerator = (gamma(nu+j+1) * gamma(nu-nu_0(ell)+j) * gamma(nu+nu_0(ell)+j))
        denominator = gamma(nu+1) * gamma(nu-nu_0(ell)) * gamma(nu+nu_0(ell))
        return (-delta)**(-j) * (numerator / denominator) * cj(-nu, -j, ell, delta)


def X(epsilon, ell, nu, C6, mu):
    delta = Delta(epsilon, C6, mu)
    diff = 1
    last = 1e-30
    sum = 1
    i = 1
    while (diff > 1e-16):
        last2 = last 
        last = sum
        sum += (-1)**i*bj(2*i, ell, delta, nu) + (-1)**(-i)*bj(-2*i, ell, delta, nu)
        diff = max(np.abs((sum-last)/last), np.abs((last-last2)/last2))
        i += 1
    return sum


def Xtest(epsilon, ell, nu, n, C6, mu):
    delta = Delta(epsilon, C6, mu)
    sum = 1
    for i in range(1, n):
        sum += (-1)**i*bj(2*i, ell, delta, nu) + (-1)**(-i)*bj(-2*i, ell, delta, nu)
    return sum


def Y(epsilon, ell, nu, C6, mu):
    delta = Delta(epsilon, C6, mu)
    diff = 1
    last = 1e-30 
    sum = bj(1, ell, delta, nu) 
    i = 1
    while (diff > 1e-16):
        last2 = last
        last = sum
        sum += (-1)**i*bj(2*i+1, ell, delta, nu) + (-1)**(-i)*bj(-2*i+1, ell, delta, nu)
        diff = max(np.abs((sum-last)/last), np.abs((last-last2)/last2))
        i += 1
    return sum


def alpha(epsilon, C6, ell, nu):
    return np.cos(np.pi * (nu-nu_0(ell)) / 2) * X(epsilon, C6, ell, nu) - np.sin(np.pi * (nu-nu_0(ell)) / 2) * Y(epsilon, C6, ell, nu)


def beta(epsilon, C6, ell, nu):
	return np.sin(np.pi * (nu-nu_0(ell)) / 2) * X(epsilon, C6, ell, nu) + np.cos(np.pi * (nu-nu_0(ell)) / 2) * Y(epsilon, C6, ell, nu)


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


def fbar(r, epsilon, C6, ell, nu):
    lvdw = beta_6(C6, mu)
    r_prime = r / lvdw
    delta = Delta(epsilon, C6)
    last = 1e-30
    tot = bj(0, ell, delta, nu)
    j = 1
    while (np.abs((tot-last)/last) > 1e-16):
        last = tot
        tot += bj(j, ell, delta, nu)*np.sqrt(r)*besselJ(nu+j, 0.5/r_prime**2) + \
            bj(-j, ell, delta, nu)*np.sqrt(r)*besselJ(nu-j, 0.5/r_prime**2)
        j += 1	
    return tot


def gbar(r, epsilon, C6, ell, nu):
    lvdw = beta_6(C6, mu)
    r_prime = r / lvdw
    delta = Delta(epsilon, C6)
    last = 1e-30
    tot = bj(0, ell, delta, nu)
    j = 1
    while (np.abs((tot-last)/last) > 1e-16):
        last = tot
        tot += bj(j, ell, delta, nu)*np.sqrt(r)*besselY(nu+j, 0.5/r_prime**2) + \
            bj(-j, ell, delta, nu)*np.sqrt(r)*besselY(nu-j, 0.5/r_prime**2)
        j += 1
    return tot


def f0(r, epsilon, C6, ell, nu):
    a = alpha(epsilon, C6, ell, nu)
    b = beta(epsilon, C6, ell, nu)
    return (1 / (a**2 + b**2)) * (a*fbar(r, epsilon, C6, ell, nu) - b*gbar(r, epsilon, C6, ell, nu))


def g0(r, epsilon, C6, ell, nu):
    a = alpha(epsilon, C6, ell, nu)
    b = beta(epsilon, C6, ell, nu)
    return (1 / (a**2 + b**2)) * (b*fbar(r, epsilon, C6, ell, nu) + a*gbar(r, epsilon, C6, ell, nu))


def C(nu, ell, D):
    reldiff = 1
    count = 0
    last = 1e-30
    c = 1
    j = 0
    while (reldiff > 1e-16):
        last2 = last
        last = c
        c *= QConv(nu+j, ell, D) 
        reldiff = max(np.abs((c-last)/last), np.abs((last-last2)/last2))
        j += 1
    return c


def G(nu, ell, D):
    return np.abs(D)**(-nu) * \
        gamma(1+nu_0(ell)+nu)*gamma(1-nu_0(ell)+nu)/gamma(1-nu) * \
        C(nu, ell, D)


def chi(D, ell, nu, C6, mu):
    epsilon = energy(D, C6, mu)
    a = alpha(epsilon, C6, ell, nu)
    b = beta(epsilon, C6, ell, nu)
    Gplus = G(nu, ell, D)
    Gminus = G(-nu, ell, D)
    return ((a*np.sin(np.pi*nu) - b*np.cos(np.pi*nu))*Gminus + b*Gplus) / \
        ((b*np.sin(np.pi*nu) + a*np.cos(np.pi*nu))*Gminus - a*Gplus)


def Zff(eps, ell, nu, C6, mu):
    D = Delta(eps, C6, mu)
    a = alpha(eps, C6, ell, nu)
    b = beta(eps, C6, ell, nu)
    Gm = G(-nu, ell, D)
    Gp = G(nu, ell, D)
    x = X(eps, C6, ell, nu)
    y = Y(eps, C6, ell, nu)
    return ((x**2 + y**2)*np.sin(np.pi*nu))**(-1) * (-(-1)**ell * \
        (a*np.sin(np.pi*nu) - b*np.cos(np.pi*nu))*Gm*np.sin(np.pi*nu - ell*np.pi/2 - \
        np.pi/4) + b*Gp*np.cos(np.pi*nu - ell*np.pi/2 - np.pi/4))


def Zfg(eps, ell, nu, C6, mu):
    D = Delta(eps, C6, mu)
    a = alpha(eps, C6, ell, nu)
    b = beta(eps, C6, ell, nu)
    Gm = G(-nu, ell, D)
    Gp = G(nu, ell, D)
    x = X(eps, C6, ell, nu)
    y = Y(eps, C6, ell, nu)
    return ((x**2 + y**2)*np.sin(np.pi*nu))**(-1) * (-(-1)**ell * \
        (a*np.sin(np.pi*nu) - b*np.cos(np.pi*nu))*Gm**np.cos(np.pi*nu - ell*np.pi/2 \
        - np.pi/4) + b*Gp*np.sin(np.pi*nu - ell*np.pi/2 - np.pi/4))


def Zgf(eps, ell, nu, C6, mu):
    D = Delta(eps, C6, mu)
    a = alpha(eps, C6, ell, nu)
    b = beta(eps, C6, ell, nu)
    Gm = G(-nu, ell, D)
    Gp = G(nu, ell, D)
    x = X(eps, C6, ell, nu)
    y = Y(eps, C6, ell, nu)
    return ((x**2 + y**2)*np.sin(np.pi*nu))**(-1) * (-(-1)**ell * \
        (b*np.sin(np.pi*nu) + a*np.cos(np.pi*nu))*Gm*np.sin(np.pi*nu - ell*np.pi/2 - \
        np.pi/4) - a*Gp*np.cos(np.pi*nu - ell*np.pi/2 - np.pi/4))


def Zgg(eps, ell, nu, C6, mu): 
    D = Delta(eps, C6, mu)
    a = alpha(eps, C6, ell, nu)
    b = beta(eps, C6, ell, nu)
    Gm = G(-nu, ell, D)
    Gp = G(nu, ell, D)
    x = X(eps, C6, ell, nu)
    y = Y(eps, C6, ell, nu)
    return ((x**2 + y**2)*np.sin(np.pi*nu))**(-1) * (-(-1)**ell * \
        (b*np.sin(np.pi*nu) + a*np.cos(np.pi*nu))*Gm*np.cos(np.pi*nu - ell*np.pi/2 - \
        np.pi/4) - a*Gp*np.sin(np.pi*nu - ell*np.pi/2 - np.pi/4))
