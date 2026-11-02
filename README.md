# Athena-Aniso

**Braginskii anisotropic transport for [Athena++](https://github.com/PrincetonUniversity/athena).**

Athena-Aniso extends the Athena++ MHD code with collisional Braginskii transport (anisotropic thermal conduction, anisotropic and gyro-viscosity, the Hall effect and thermoelectric effects) plus bulk viscosity, together with the time-step limits, super-time-stepping split and Riemann-solver signal speeds needed to run them stably. In partially ionised gas, neutrals diffuse isotropically and ions anisotropically, weighted by the local ionisation fraction.

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)
<!-- Once a release is archived on Zenodo:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX) -->

This is the code for Albery A.F and Pittard J.M., `von Neumann Stability Analysis of Braginskii MHD: Thermal Conduction, Viscosity, Magnetic Diffusion, and Thermoelectric Effects; and an Implementation in Athena++`, In Prep. It is built on Athena++ (upstream commit `823614c90b594472747a0ac2a699e4a454f300d2`); for everything inherited from Athena++ (AMR, coordinate systems, the ideal MHD solvers, I/O) see the [Athena++ documentation](https://github.com/PrincetonUniversity/athena/wiki).

## What Athena-Aniso adds

**Transport.** Every new coefficient defaults to zero; a term is switched on by giving its coefficient a positive value in the input file.

| Process | Components | Effects |
|---|---|---|
| Thermal conduction | parallel, perpendicular, cross | anisotropic conduction, Righi–Leduc |
| Shear viscosity | five Braginskii terms | parallel, perpendicular and gyroviscous stress |
| Bulk viscosity | isotropic | molecular gas |
| Thermoelectric heat flux | parallel, perpendicular, cross | Peltier, Ettingshausen |
| Thermoelectric force (EMF) | parallel, perpendicular, cross | Seebeck, Nernst |
| Hall effect | $\eta_{\text{H}}\vec{J}\times\hat{b}$ EMF | whistler waves |
| Partial ionisation | conduction, viscosity | isotropic neutrals, anisotropic ions |

**Numerics.**

- Analytic time-step limits for every Braginskii term, with separate parabolic and dispersive time steps.
- A split compatible with super-time-stepping: parabolic parts can be integrated with Athena++'s RKL integrators, while dispersive terms (Hall, Righi–Leduc, gyroviscosity, and Peltier/Seebeck) stay with the main integrator.
- HLL-Mod: HLLE with per-variable signal speeds that include the dispersive waves (whistler, thermoelectric, gyroviscous), extending the Hall-HLL approach of Lesur et al. (2014).

The equations, sign conventions, units and problem-generator interface for the transport terms are in **[diffusion.md](diffusion.md)**.

## Quick start

Build requirements are the same as for Athena++.

```bash
git clone https://github.com/AlexandraAlbery/Athena-Aniso.git
cd Athena-Aniso
python configure.py --prob=<problem> -b   # -b (MHD): the anisotropic, Hall and thermoelectric terms need B
make clean && make -j
./bin/athena -i <path/to/athinput>
```

- **Super-time-stepping:** configure with `-sts` and set `sts_integrator = rkl1` or `rkl2` in the `<time>` block, as in Athena++.
- **HLL-Mod:** configure with `--flux = hlle_mod`

Switch transport terms on in the `<problem>` block of the input file:

```
<problem>
kappa_iso                = 1.0   # Isotropic Thermal Conduction (Stock Athena++)
nu_iso                   = 1.0   # Isotropic Viscoisty (Stock Athena++)
eta_ohm                  = 1.0   # Ohmic Resistivity (Stock Athena++)
eta_ad                   = 1.0   # Ambipolar Diffusion (Stock Athena++)
kappa_aniso_neutral      = 1.0   # Neutral Thermal Conduction
kappa_aniso_para         = 1.0   # Parallel Thermal Conduction
kappa_aniso_perp         = 1.0   # Perpendicular Thermal Conduction
kappa_aniso_wedge        = 1.0   # Wedge Thermal Conduction (Righi-Leduc Effect)
nu_aniso_neutral         = 1.0   # Neutral viscosity
nu_aniso_para            = 1.0   # Parallel (Pi0) Viscosity
nu_aniso_perp1           = 1.0   # Perpendicular (Pi1) Viscosity
nu_aniso_perp2           = 1.0   # Perpendicular (Pi2) Viscosity
nu_aniso_wedge1          = 1.0   # Wedge (Pi3) Gyro-Viscosity
nu_aniso_wedge2          = 1.0   # Wedge (Pi4) Gyro-Viscosity
zeta_bulk                = 1.0   # Bulk Viscosity
eta_hall                 = 1.0   # Hall Effect
kappa_aniso_TE_para      = 1.0   # Parallel Thermoelectric Effects (Peltier + Seebeck Effects)
kappa_aniso_TE_perp      = 1.0   # Perpendicular Thermoelectric Effects (Peltier + Seebeck Effects)
kappa_aniso_TE_wedge     = 1.0   # Wedge Thermoelectric Effects (Nernst + Ettingshausen Effects)
ion_frac_const           = 1.0   # Constant Ionisation Fraction
debug_therm_conduct_only = false # Only Thermoelectric Conduction (Peltier + Ettingshausen Effects) debug only!
debug_therm_force_only   = false # Only Thermoelectric Force (Seebeck + Nernst Effects) debug only!
```

Coefficients that vary in space or time are set cell by cell in the functions enrolled with `EnrollConductionCoefficient` for thermal conduction, `EnrollViscosityCoefficient` for viscoisty, and `EnrollFieldDiffusivity` for the Hall and thermoelectric effects. See [diffusion.md](diffusion.md).

## Verification

The test problems from the paper run in 3D Cartesian geometry with a single mode along x. The problem generator is `src/pgen/aniso-tests.cpp`, with input files in `inputs/mhd/athinput.aniso-tets`.

| Case | Test |
|---|---|
| 0–6 | convergence to analytical solutions, covering every transport term |
| 7 | HLL-Mod suppressing grid-scale instability at RK3, for the Righi–Leduc, gyroviscous, Hall and Peltier–Seebeck terms, singly and combined |
| 8 | time-step sweep |

## Citing Athena-Aniso

If you use Athena-Aniso, please cite the method paper and Athena++:

```bibtex
@unpublished{AthenaAniso,
  author = {Alexandra F. Albery and Julian M. Pittard},
  title  = {Von Neumann Stability Analysis of Braginskii MHD: Thermal Conduction, Viscosity, Magnetic Diffusion, and Thermoelectric Effects - An Implementation in Athena++},
  year   = {2026},
  note   = {in prep.}
}

@article{Stone2020,
	doi = {10.3847/1538-4365/ab929b},
	url = {https://doi.org/10.3847%2F1538-4365%2Fab929b},
	year = 2020,
	month = jun,
	publisher = {American Astronomical Society},
	volume = {249},
	number = {1},
	pages = {4},
	author = {James M. Stone and Kengo Tomida and Christopher J. White and Kyle G. Felker},
	title = {The Athena$\mathplus$$\mathplus$ Adaptive Mesh Refinement Framework: Design and Magnetohydrodynamic Solvers},
	journal = {The Astrophysical Journal Supplement Series},
}
```

<!-- Once archived: to cite the exact version used, also cite its Zenodo DOI (10.5281/zenodo.XXXXXXX). -->

## Relationship to Athena++

Athena-Aniso is a modified version of Athena++ and is not an official Athena++ release. Please report problems with it on this repository's [issue tracker](https://github.com/AlexandraAlbery/Athena-Aniso/issues) rather than to the Athena++ developers.

## License

BSD 3-Clause, the same as Athena++. The original Athena++ copyright notice is kept in [LICENSE](LICENSE).
