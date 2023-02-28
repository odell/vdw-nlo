eps = 1e-16
tiny = 1e-30
nu = 1
E = -20
l = 0
Nmax = 1000

print('nu = ', nu, '\tE = ', E)

def nu0(l):
    return (2*l+1) / 4

def p(nu, Del, l):
    return -Del**2 / ((nu+1) * ((nu+1)**2 - nu0(l)**2) * (nu+2) * ((nu+2)**2 - nu0(l)**2))

def q(nu, l):
    return (nu+1) * ((nu+1)**2 - nu0(l)**2)

'''
Q 
'''
b0 = 0
f0 = b0 if b0 != 0 else tiny 
b = 1
C0 = f0
D0 = 0
Delta = 2
j = 1
while abs(Delta - 1) > eps:
    a = p(nu+j-2, E, l) if j > 1 else 1
    D = b + a*D0
    if D == 0:
        D = tiny 
    D = 1/D
    C = b + a/C0
    if C == 0:
        C = tiny 
    Delta = C*D
    f = f0*Delta
    D0 = D
    C0 = C
    f0 = f
    j += 1
    if j > Nmax:
        break
    print('Q = ', f)

'''
Qbar 
'''
b0 = 0
f0 = b0 if b0 != 0 else tiny
C0 = f0
D0 = 0
Delta = 2
j = 1
while abs(Delta - 1) > eps:
    a = -E**2 if j > 1 else 1
    b = q(nu+j-1, l)
    D = b + a*D0
    D = 1/D
    C = b + a/C0
    Delta = C*D
    f = f0*Delta
    D0 = D
    C0 = C
    f0 = f
    j += 1
    if j > Nmax:
        break
    print('Qbar = ', f)
