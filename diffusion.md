# Braginskii transport in Athena-Aniso

Athena-Aniso extends Athena++'s `HydroDiffusion` and `FieldDiffusion` modules with collisional Braginskii transport:

- anisotropic thermal conduction: parallel, perpendicular and wedge (Righi–Leduc) components;
- anisotropic shear viscosity, including gyroviscosity;
- bulk viscosity;
- thermoelectric effects: Peltier, Seebeck, Ettingshausen and Nernst.

In partially ionised gas the neutral component diffuses isotropically and the ionised component anisotropically, weighted by the ionisation fraction (see [Partially ionised gas](#partially-ionised-gas)).

Every coefficient defaults to zero. A term is switched on by giving its coefficient a value greater than zero in the `<problem>` block of the input file. Coefficients that vary in space or time are set cell by cell in the functions enrolled with `EnrollConductionCoefficient` and `EnrollViscosityCoefficient`, the same as Athena++'s isotropic coefficients; each section below lists the accessors.

**Contents:** [Notation](#notation) · [Thermal conduction](#thermal-conduction) · [Thermoelectric effects](#thermoelectric-effects) · [Shear viscosity](#shear-viscosity) · [Bulk viscosity](#bulk-viscosity) · [Partially ionised gas](#partially-ionised-gas) · [Units summary](#units-summary) · [References](#references)

## Notation

| Symbol | Meaning |
|---|---|
| $\hat{b} = \vec{B}/\lvert\vec{B}\rvert$ | unit vector along the magnetic field |
| $\boldsymbol{I}_{\perp} = \boldsymbol{I} - \hat{b}\otimes\hat{b}$ | projection perpendicular to $\hat{b}$ |
| $\boldsymbol{W} = \vec{\nabla}[\vec{v}] + \vec{\nabla}[\vec{v}]^{\top} - \frac{2}{3}(\nabla\cdot\vec{v})\boldsymbol{I}$ | rate-of-strain tensor |
| $\boldsymbol{W}_{\perp} = \boldsymbol{I}_{\perp}\cdot\boldsymbol{W}\cdot\boldsymbol{I}_{\perp}$ | perpendicular part of $\boldsymbol{W}$ |
| $\vec{J} = \nabla\times\vec{B}$ | current density (code units) |
| $[T]=P/\rho$ | temperature (code units) |
| $\text{HII}_{\text{frac}}$ | ionisation fraction, `HydroDiffusion::ion_frac(k,j,i)` |

Cross products with a tensor act on the adjacent index:

```math
(\hat{b}\times\boldsymbol{W})_{ij} = \epsilon_{ikl}\hat{b}_k W_{lj}, \qquad (\boldsymbol{W}\times\hat{b})_{ij} = \epsilon_{jkl}W_{ik}\hat{b}_l .
```

## Thermal conduction

The conductive heat flux combines an isotropic neutral flux with an anisotropic ionised flux:

```math
\vec{q}_{\text{aniso}} = (1-\text{HII}_{\text{frac}})\vec{q}_{\text{neu}} + \text{HII}_{\text{frac}}\vec{q}_{e} .
```

The neutral flux is isotropic,

```math
\vec{q}_{\text{neu}} = -\rho\kappa_{\text{neu}}\nabla[T],
```

and the ionised flux has parallel, perpendicular and wedge components:

```math
\begin{aligned}
\vec{q}_{e} &= \vec{q}_{\parallel} + \vec{q}_{\perp} + \vec{q}_{\wedge},\\
\vec{q}_{\parallel} &= -\rho\kappa_{\parallel}(\hat{b}\cdot\nabla[T])\hat{b},\\
\vec{q}_{\perp} &= -\rho\kappa_{\perp}\left(\nabla[T] - (\hat{b}\cdot\nabla[T])\hat{b}\right),\\
\vec{q}_{\wedge} &= -\rho\kappa_{\wedge}\hat{b}\times\nabla[T].
\end{aligned}
```

The wedge term is the Righi–Leduc heat flux. Its sign is that of Braginskii's electron heat flux; Braginskii's ion wedge term has the opposite sign.

| Input (`<problem>`) | Coefficient | Accessor |
|---|---|---|
| `kappa_aniso_neutral` | $\kappa_{\text{neu}}$ | `kappa(HydroDiffusion::DiffProcess::kappa_neutral,k,j,i)` |
| `kappa_aniso_para` | $\kappa_{\parallel}$ | `kappa(HydroDiffusion::DiffProcess::kappa_para,k,j,i)` |
| `kappa_aniso_perp` | $\kappa_{\perp}$ | `kappa(HydroDiffusion::DiffProcess::kappa_perp,k,j,i)` |
| `kappa_aniso_wedge` | $\kappa_{\wedge}$ | `kappa(HydroDiffusion::DiffProcess::kappa_wedge,k,j,i)` |

All four are thermal diffusivities with dimensions $L^{2} T^{-1}$, the same convention as Athena++'s isotropic `kappa_iso`. Setting `kappa_aniso_neutral` > 0 turns on ionisation weighting for conduction.

## Thermoelectric effects

Thermoelectric transport has two parts, both computed in `FieldDiffusion`: a current-driven heat flux $\vec{q}_{J}$ and a temperature-gradient-driven thermoelectric force $\vec{F}_{T}$. One set of coefficients, $\kappa_{J,\parallel}$, $\kappa_{J,\perp}$ and $\kappa_{J,\wedge}$, controls both, so switching on a component switches on its heat flux and its force together.

**Heat flux.** $\vec{q}_{J}$ is added directly to the energy flux at cell faces:

```math
\begin{aligned}
\vec{q}_{J} &= \text{HII}_{\text{frac}}(\vec{q}_{J,\parallel} + \vec{q}_{J,\perp} + \vec{q}_{J,\wedge}),\\
\vec{q}_{J,\parallel} &= -\kappa_{J,\parallel}(\hat{b}\cdot\vec{J})\hat{b},\\
\vec{q}_{J,\perp} &= -\kappa_{J,\perp}\left(\vec{J} - \hat{b}(\hat{b}\cdot\vec{J})\right),\\
\vec{q}_{J,\wedge} &= -\kappa_{J,\wedge}\hat{b}\times\vec{J}.
\end{aligned}
```

**Force.** $\vec{F}_{T}$ is added to the electric fields at cell edges for UCT:

```math
\begin{aligned}
\vec{F}_{T} &= \vec{F}_{T,\parallel} + \vec{F}_{T,\perp} + \vec{F}_{T,\wedge},\\
\vec{F}_{T,\parallel} &= -\frac{1}{T}\kappa_{J,\parallel}\hat{b}(\hat{b}\cdot\nabla[T]),\\
\vec{F}_{T,\perp} &= -\frac{1}{T}\kappa_{J,\perp}\left(\nabla[T] - \hat{b}(\hat{b}\cdot\nabla[T])\right),\\
\vec{F}_{T,\wedge} &= -\frac{1}{T}\kappa_{J,\wedge}\hat{b}\times\nabla[T].
\end{aligned}
```

The individual terms are the classical thermoelectric effects:

| Term | Driven by | Effect |
|---|---|---|
| $\vec{q}_{J,\parallel}$, $\vec{q}_{J,\perp}$ | current | Peltier |
| $\vec{q}_{J,\wedge}$ | current across $\vec{B}$ | Ettingshausen |
| $\vec{F}_{T,\parallel}$, $\vec{F}_{T,\perp}$ | temperature gradient | Seebeck |
| $\vec{F}_{T,\wedge}$ | temperature gradient across $\vec{B}$ | Nernst |

Because one coefficient multiplies $\vec{J}$ in $\vec{q}_{J}$ and $\nabla[T]/T$ in $\vec{F}_{T}$, the Peltier coefficient is $[T]$ times the Seebeck coefficient: the Onsager reciprocity relation is built in.

**Well-posedness.** Use the thermoelectric terms together with neutral and/or perpendicular thermal conduction and Ohmic and/or ambipolar diffusion. On its own, the wedge (Nernst–Ettingshausen) branch behaves like a backward heat equation; perpendicular conduction and perpendicular magnetic diffusion together are what stabilise it. The paper gives the criterion.

| Input (`<problem>`) | Coefficient | Accessor |
|---|---|---|
| `kappa_therm_para` | $\kappa_{J,\parallel}$ | `FieldDiffusion::kappa_therm_para` |
| `kappa_therm_perp` | $\kappa_{J,\perp}$ | `FieldDiffusion::kappa_therm_perp` |
| `kappa_therm_wedge` | $\kappa_{J,\wedge}$ | `FieldDiffusion::kappa_therm_wedge` |

These can also be set in the function enrolled with `EnrollFieldDiffusivity` alongside Ohmic, ambipolar, and Hall.

**Units.** $\kappa_{J}$ has the dimensions of an electrical current $L^{3/2} M^{1/2} T^{-2}$. This makes $\vec{q}_{J}$ an energy flux and $\vec{F}_{T}$ an EMF with the units of $\vec{v}\times\vec{B}$. Converting a Gaussian coefficient to this normalisation ($B_{\text{code}} = B_{\text{G}}/\sqrt{4\pi}$) brings in a factor $c/\sqrt{4\pi}$. Braginskii's parallel electron coefficient, for example, becomes

```math
\kappa_{J,\parallel} = 0.71\frac{k_{\mathrm{B}}c}{e\sqrt{4\pi}}T = 0.71\frac{\tilde{m}c}{\sqrt{4\pi}e}\frac{P}{\rho} \qquad (Z = 1).
```

## Shear viscosity

As for conduction, the viscous stress combines an isotropic neutral part with an anisotropic ionised part:

```math
\boldsymbol{\Pi}_{\text{aniso}} = (1-\text{HII}_{\text{frac}})\boldsymbol{\Pi}_{\text{neu}} + \text{HII}_{\text{frac}}\boldsymbol{\Pi}_{\text{ion}},
```

where $\boldsymbol{\Pi}_{\text{neu}} = \rho\nu_{\text{neu}}\boldsymbol{W}$ is the isotropic stress, the same as Athena++'s `nu_iso`. The ionised stress is the sum of the five Braginskii terms:

```math
\begin{aligned}
\boldsymbol{\Pi}_{\text{ion}} &= \boldsymbol{\Pi}^{(0)} + \boldsymbol{\Pi}^{(1)} + \boldsymbol{\Pi}^{(2)} + \boldsymbol{\Pi}^{(3)} + \boldsymbol{\Pi}^{(4)},\\
\boldsymbol{\Pi}^{(0)} &= 3\rho\nu_{\parallel}\left(\hat{b}\otimes\hat{b}-\frac{1}{3}\boldsymbol{I}\right)\left(\hat{b}\otimes\hat{b}-\frac{1}{3}\boldsymbol{I}\right):\vec{\nabla}[\vec{v}],\\
\boldsymbol{\Pi}^{(1)} &= \rho\nu_{\perp}\left[\boldsymbol{W}_{\perp} + \frac{1}{2}\boldsymbol{I}_{\perp}(\hat{b}\cdot\boldsymbol{W}\cdot\hat{b})\right],\\
\boldsymbol{\Pi}^{(2)} &= 4\rho\nu_{\perp}\left[\boldsymbol{I}_{\perp}\cdot\boldsymbol{W}\cdot(\hat{b}\otimes\hat{b}) + (\hat{b}\otimes\hat{b})\cdot\boldsymbol{W}\cdot\boldsymbol{I}_{\perp}\right],\\
\boldsymbol{\Pi}^{(3)} &= -\frac{1}{2}\rho\nu_{\wedge}\left[\hat{b}\times\boldsymbol{W}\cdot\boldsymbol{I}_{\perp} - \boldsymbol{I}_{\perp}\cdot\boldsymbol{W}\times\hat{b}\right],\\
\boldsymbol{\Pi}^{(4)} &= -2\rho\nu_{\wedge}\left[\hat{b}\times\boldsymbol{W}\cdot(\hat{b}\otimes\hat{b}) - (\hat{b}\otimes\hat{b})\cdot\boldsymbol{W}\times\hat{b}\right].
\end{aligned}
```

$\boldsymbol{\Pi}^{(0)}$ is the parallel viscosity, $\boldsymbol{\Pi}^{(1)}$ and $\boldsymbol{\Pi}^{(2)}$ are the perpendicular viscosity, and $\boldsymbol{\Pi}^{(3)}$ and $\boldsymbol{\Pi}^{(4)}$ are the gyroviscosity.

**Sign convention.** $\boldsymbol{\Pi}$ is the viscous stress, which enters the momentum flux with a minus sign, $\rho\vec{v}\otimes\vec{v} + (p + B^2/2)\boldsymbol{I} - \vec{B}\otimes\vec{B} - \boldsymbol{\Pi}$, the same as Athena++'s isotropic viscosity. Braginskii's $\boldsymbol{\pi}$ is $-\boldsymbol{\Pi}$; with that sign the five terms are Braginskii's ion stress with $\eta_0 = \rho\nu_{\parallel}$, $\eta_1 = \rho\nu_{\perp}$, $\eta_2 = 4\eta_1$, $\eta_3 = \rho\nu_{\wedge}$ and $\eta_4 = 2\eta_3$.

**Choosing coefficients.** The factors $\eta_2 = 4\eta_1$ and $\eta_4 = 2\eta_3$ are already built into $\boldsymbol{\Pi}^{(2)}$ and $\boldsymbol{\Pi}^{(4)}$. Compute the physical kinematic viscosities $\nu_{\perp} = \eta_1/\rho$ and $\nu_{\wedge} = \eta_3/\rho$, then pass the same value to `nu_aniso_perp1` and `nu_aniso_perp2`, and to `nu_aniso_wedge1` and `nu_aniso_wedge2`. Keeping each pair as separate inputs lets each term be switched on independently.

| Input (`<problem>`) | Coefficient | Term | Accessor |
|---|---|---|---|
| `nu_aniso_neutral` | $\nu_{\text{neu}}$ | $\boldsymbol{\Pi}_{\text{neu}}$ | `nu(HydroDiffusion::DiffProcess::nu_neutral,k,j,i)` |
| `nu_aniso_para` | $\nu_{\parallel}$ | $\boldsymbol{\Pi}^{(0)}$ | `nu(HydroDiffusion::DiffProcess::nu_para,k,j,i)` |
| `nu_aniso_perp1` | $\nu_{\perp}$ | $\boldsymbol{\Pi}^{(1)}$ | `nu(HydroDiffusion::DiffProcess::nu_perp1,k,j,i)` |
| `nu_aniso_perp2` | $\nu_{\perp}$ | $\boldsymbol{\Pi}^{(2)}$ | `nu(HydroDiffusion::DiffProcess::nu_perp2,k,j,i)` |
| `nu_aniso_wedge1` | $\nu_{\wedge}$ | $\boldsymbol{\Pi}^{(3)}$ | `nu(HydroDiffusion::DiffProcess::nu_wedge1,k,j,i)` |
| `nu_aniso_wedge2` | $\nu_{\wedge}$ | $\boldsymbol{\Pi}^{(4)}$ | `nu(HydroDiffusion::DiffProcess::nu_wedge2,k,j,i)` |

All six are kinematic viscosities with dimensions $L^{2} T^{-1}$, the same convention as Athena++'s isotropic `nu_iso`. Setting `nu_aniso_neutral` > 0 turns on ionisation weighting for viscosity.

## Bulk viscosity

```math
\boldsymbol{\Pi}_{\text{bulk}} = \zeta(\nabla\cdot\vec{v})\boldsymbol{I}
```

Switched on with `zeta_bulk` in `<problem>`; set `HydroDiffusion::zeta_bulk` in the function enrolled with `EnrollViscosityCoefficient`.

Bulk viscosity only acts in molecular gas: it comes from the relaxation of rotational and vibrational degrees of freedom. If the simulation tracks the thermodynamic state or chemistry and passes through partially atomic, fully atomic, partially ionised or fully ionised regimes, weight $\zeta$ by the molecular mass fraction.

$\zeta$ should be passed as a dynamic viscosity $M L^{-1} T^{-1}$ as it is not multiplied by $\rho$ in the solver

## Partially ionised gas

Ionisation weighting uses one ionisation-fraction array, `HydroDiffusion::ion_frac(k,j,i)`, shared by all processes. It is only switched on when a neutral coefficient is on:

- **Conduction:** if `kappa_aniso_neutral` > 0, the function enrolled with `EnrollConductionCoefficient` must fill `ion_frac(k,j,i)` (for example from a thermodynamic lookup table) as well as `kappa(HydroDiffusion::DiffProcess::kappa_neutral,k,j,i)`.
- **Viscosity:** if `nu_aniso_neutral` > 0, the function enrolled with `EnrollViscosityCoefficient` must fill `ion_frac(k,j,i)` as well as `nu(HydroDiffusion::DiffProcess::nu_neutral,k,j,i)`.

If the neutral coefficient is zero or not set in the input file, the gas is treated as fully ionised and the anisotropic terms are not weighted.

## Units summary

| Coefficients | Inputs | Dimensions |
|---|---|---|
| $\kappa_{\text{neu}}$, $\kappa_{\parallel}$, $\kappa_{\perp}$, $\kappa_{\wedge}$ | `kappa_aniso_*` | $L^{2} T^{-1}$ (thermal diffusivity) |
| $\nu_{\text{neu}}$, $\nu_{\parallel}$, $\nu_{\perp}$, $\nu_{\wedge}$ | `nu_aniso_*` | $L^{2} T^{-1}$ (kinematic viscosity) |
| $\kappa_{J,\parallel}$, $\kappa_{J,\perp}$, $\kappa_{J,\wedge}$ | `kappa_therm_*` | $L^{3/2} M^{1/2} T^{-2}$ (electrical current) |
| $\zeta$ | `zeta_bulk` | $M L^{-1} T^{-1}$ (dynamic viscosity) |

L, T and M denote length, time and mass.

## References

- Braginskii, S. I. 1965, *Reviews of Plasma Physics*, 1, 205
- Stone, J. M., Tomida, K., White, C. J. & Felker, K. G. 2020, ApJS, 249, 4
- Albery, A. F, Pittard, J. M (in prep.), *Von Neumann Stability Analysis of Braginskii MHD: Thermal Conduction, Viscosity, Magnetic Diffusion, and Thermoelectric Effects - An implementation in Athena++*
