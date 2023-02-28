import numpy as np
from scipy.special import gamma, hyp0f1

def J(v, x):
    C = (x/2)**v / gamma(v + 1)
    last = 1e-6 
    k = 1
    while (abs((C-last)/last) > 1e-16):
        last = C
        C += (x/2)**v * (-1)**k / (gamma(k+1) * gamma(v+k+1)) * (x/2)**(2*k)
        k += 1
    return C

def Y(v, x):
    return (np.cos(v*np.pi)*J(v, x) - J(-v, x)) / np.sin(v*np.pi)

nu = 0.25+0.25j

# check J
x = np.linspace(1e-6, 10, 100)
jv = np.array([J(nu, i) for i in x])

data = np.zeros((100, 3))
data[:, 0] = x
data[:, 1] = np.real(jv)
data[:, 2] = np.imag(jv)
np.savetxt('Jv.dat', data)

yv = np.array([Y(nu, i) for i in x])

# check Y
data = np.zeros((100, 3))
data[:, 0] = x
data[:, 1] = np.real(yv)
data[:, 2] = np.imag(yv)
np.savetxt('Yv.dat', data)
