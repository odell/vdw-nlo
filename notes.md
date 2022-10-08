# ${}^4{\rm He}$ NLO

## Interaction

The coordinate-space potential is
$$
V(r) = \left[1 - e^{-(10r/R)^2}\right]^8\left(-\frac{C_6}{r^6}\right)
$$
where
$$
C_6 = 10130.537638625547 ~{\rm K}~{\rm \AA}^6~.
$$
The nonlocal, momentum-space potential is then
$$
\tilde{V}_{\ell,\ell^\prime}(p,k)= e^{-(p/\Lambda)^6}~e^{-(k/\Lambda)^6}~\frac{2}{\pi}\int_0^\infty dr~r^2~j_\ell(pr)V(r)j_{\ell^\prime}(kr)
$$
with $\Lambda\equiv 2/R$. 

The leading-order (LO), nonlocal, momemtum-space counterterm is
$$
\tilde{V}_{\ell,\ell^\prime}(p,k)= g_{\rm LO}~\delta_{\ell,\ell^\prime}\left(\frac{p}{\Lambda}\right)^\ell \left(\frac{k}{\Lambda}\right)^{\ell^\prime} e^{-(p/\Lambda)^6} e^{-(k/\Lambda)^6}
$$
The next-to-leading order counterterm is
$$
\tilde{V}_{\ell,\ell^\prime}(p,k)= g_{\rm NLO}~\delta_{\ell,\ell^\prime} \left(\frac{(p/\Lambda)^2+(k/\Lambda)^2}{2}\right) \left(\frac{p}{\Lambda}\right)^\ell \left(\frac{k}{\Lambda}\right)^{\ell^\prime} e^{-(p/\Lambda)^6} e^{-(k/\Lambda)^6}.
$$

## Tuning

I tuned $g_{\rm LO}$ such that $a_0 = 100~\AA$ at LO. At NLO, I'm fixing $r_0=7.33~\AA$ with $g_{\rm NLO}$ in addition to $a_0$