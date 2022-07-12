# vdw-nlo
Performs LO and NLO predictions for 4He and 6Li systems.

The 4He-4He interaction is defined in `helium4.py`. There is a function defined
there, `construct_helium4_system`, that uses default parameters for the
regulators and integration meshes to return a "System". 

The LO tuning for 4He is completed in the corresponding notebook. The scattering
length is chosen as the observable of interest (rather than the binding energy
of the dimer) to maintain consistency with the 6Li work (that has been done but
not yet brought over into this repository).

## `mu2`

The calculations in this repository rely on the package `mu2`. This package is
available on the Python Package Index (PyPI) via:

```
pip install mu2
```
