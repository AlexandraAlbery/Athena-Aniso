This adds support for full Braginskii thermal conduction, viscosity, and thermoelectric effects (Peltier, Seebeck, Ettinghausen, Nernst).
Bulk viscosity is also added.

THERMAL CONDUCTION:
The thermal condcution is given by:
$\vec{q}_{\text{aniso}} = (1-\text{HII}_{\text{frac}})\vec{q}_{\text{neu}}+\text{HII}_{\text{frac}}\vec{q}_{\text{ion}}$.
This allows for neutral gas to behave isotropically and ionised gas to behave anisotropically in partially ionised regimes.

$\vec{q}_{\text{neu}}$ is isotropic thermal conductivity: $\vec{q}_{\text{neu}} = -\kappa_{\text{neu}}\nabla[T]$,

$\vec{q}_{\text{ion}}$ is split in to three parts

$\vec{q}_{\text{ion}} = \vec{q}_{\parallel} + \vec{q}_{\perp} + \vec{q}_{\wedge},$

$\vec{q}_{\parallel}=-\kappa_{\parallel}(\hat{b}\cdot\nabla [T])\hat{b},$

$\vec{q}_{\perp} = -\kappa_{\perp}(\nabla[T] - (\hat{b}\cdot\nabla[T])\hat{b}),$

$\vec{q}_{\wedge} = -\kappa_{\wedge}\hat{b}\times\nabla[T].

the ionisating fraction (for all processes) is accesed from
HydroDiffusion::ion_frac(k,j,i).
Ionisation weighting is only enabled if neutral diffusion is enabled.

The thermal conduction coefficients can be accessed from
kappa(HydroDiffusion::DiffProcess::kappa_neutral,k,j,i),
kappa(HydroDiffusion::DiffProcess::kappa_para,k,j,i),
kappa(HydroDiffusion::DiffProcess::kappa_perp,k,j,i),
kappa(HydroDiffusion::DiffProcess::kappa_wedge,k,j,i).

THERMOELECTRIC EFFECTS
Thermoelectric effects are split in to two parts: thermoelectric conduction $\vec{q}_{J}$ and thermoelectric force $\vec{F}_{T}$. Both are called in field diffusion

$\vec{q}_{J,\parallel} = -\kappa_{J,\parallel}(\hat{b}\cdot\vec{J})\hat{b}.$

$\vec{q}_{J,\perp} = -\kappa_{J,\perp}(\vv{J}-\hat{b}(\hat{b}\cdot\vec{J}))$

$\vec{q}_{J,\wedge} = -\kappa_{J,\wedge}\hat{b}\times\vec{J}$.

These are added directly to the energy flux at the cell faces.

\kappa_{\text{neu}}, \kappa_{\parallel}, \kappa_{\perp}, \kappa_{wedge} must be supplied as diffusivities with dimensions of L^{2} T^{-1}. The same as isotropic case.
kappa_{J,\parallel}, \kappa_{J,\perp}, \kappa_{J,wedge} must be supplied in dimensions of: This usually means a prefactor of k_{b}c/e.

These are activated by giving them a value greater than zero in the input file:
kappa_aniso_neutral = pin->GetOrAddReal("problem", "kappa_aniso_neutral", 0.0); // iso conduction for neutral gas
kappa_aniso_para = pin->GetOrAddReal("problem", "kappa_aniso_para", 0.0); // aniso parallel conduction
kappa_aniso_perp = pin->GetOrAddReal("problem", "kappa_aniso_perp", 0.0); // aniso perpendicular conduction
kappa_aniso_wedge = pin->GetOrAddReal("problem", "kappa_aniso_wedge", 0.0); // aniso cross conduction

If any of the thermoelectric terms are active their respective thermoelectric force terms are added to and substracted from the energy and magnetic flux respectively:

$vec{F}_{T} =  \vec{F}_{T,\parallel} + \vec{F}_{T,\perp} + \vec{F}_{T,\wedge}$, //
$\vec{F}_{T,\parallel} = -\frac{1}{T} \kappa_{J,\parallel}\hat{b}(\hat{b}\cdot\nabla[T])$, //
$\vec{F}_{T,\perp} = -\frac{1}{T} \kappa_{J,\perp}(\nabla[T] - \hat{b}(\hat{b}\cdot\nabla[T]))$, //
$\vec{F}_{T,\wedge} = -\frac{1}{T} \kappa_{J,\wedge}\hat{b}\times\nabla[T]$.


kappa_therm_para = pin->GetOrAddReal("problem", "kappa_therm_para", 0.0); // thermoelectric parallel conduction
kappa_therm_perp = pin->GetOrAddReal("problem", "kappa_therm_perp", 0.0); // thermoelectric perpendicular conduction
kappa_therm_wedge = pin->GetOrAddReal("problem", "kappa_therm_wedge", 0.0); // thermoelectric cross conduction

All coefficented can be passed in EnrollConductionCoefficient. They are accessed via:
HydroDiffusion::kappa_neutral, HydroDiffusion::kappa_para, HydroDiffusion::kappa_perp, kappa_wedge, FieldDiffusion::kappa_therm_para, FieldDiffusion::kappa_therm_perp, FieldDiffusion::kappa_therm_wedge.
If -\kappa_{\text{neu}} is defined in the input file, then HydroDiffusion::ion_frac must also be passed in the EnrollConductionCoefficient function as well as HydroDiffusion::kappa_neutral. For example from a thermodynamic lookup table.
If  -\kappa_{\text{neu}} is not activated, then full ionisation is assumed, and the activated anisotropic components will not be weighted by ionisation fraction.

SHEAR VISCOSITY
The anisotropic shear viscosity is given by:
\boldsymbol{\Pi}_{\text{aniso}}= (1- \text{HII}_{\text{frac}})\boldsymbol{\Pi}_{\text{neu}} + \text{HII}_{\text{frac}}\boldsymbol{\Pi}_{\text{ion}}.
his allows for neutral gas to behave isotropically and ionised gas to behave anisotropically in partially ionised regimes.
\boldsymbol{\Pi}_{\text{ion}} = \boldsymbol{\Pi}^{(0) + \boldsymbol{\Pi}^{(1) + \boldsymbol{\Pi}^{(2)} + \boldsymbol{\Pi}^{(3)} + \boldsymbol{\Pi}^{(4)}
\boldsymbol{\Pi}^{(0)} = 3\rho\nu_{\parallel}\left(\hat{b}\otimes\hat{b}-\frac{1}{3}\boldsymbol{I}\right)\left(\hat{b}\otimes\hat{b}-\frac{1}{3}\boldsymbol{I}\right):\vv{\nabla}[\vv{v}],
\boldsymbol{\Pi}^{(1)} =  \rho\nu_{\perp}\left[\boldsymbol{W}_{\perp} +\frac{1}{2}\boldsymbol{I}_{\perp}(\hat{b}\cdot\boldsymbol{W}\cdot\hat{b}) \right],
\boldsymbol{\Pi}^{(2)} =  4\rho\nu_{\perp}\left[\boldsymbol{I}_{\perp} \cdot \boldsymbol{W} \cdot \left(\hat{b}\otimes\hat{b}\right) +\left(\hat{b}\otimes\hat{b}\right) \cdot \boldsymbol W \cdot \boldsymbol{I}_{\perp}
\right],
\boldsymbol{\Pi}^{(3)} = -\frac{1}{2}\rho\nu_{\wedge}\left[\hat{b}\times\boldsymbol{W}\cdot\boldsymbol{I}_{\perp} -\boldsymbol{I}_{\perp}\cdot \boldsymbol{W} \times \hat{b} \right],
\boldsymbol{\Pi}^{(4)} = -2\rho\nu_{\wedge}\left[\hat{b}\times\boldsymbol{W}\cdot(\hat{b}\otimes\hat{b})-(\hat{b}\otimes\hat{b})\cdot\boldsymbol{W}\times\hat{b}
\right].
These are activated in the input file by:
nu_aniso_neutral{pin->GetOrAddReal("problem", "nu_aniso_neutral", 0.0)}, // neutral viscosity
nu_aniso_para{pin->GetOrAddReal("problem", "nu_aniso_para", 0.0)}, // parallel viscosity - Pi^(0)
nu_aniso_perp1{pin->GetOrAddReal("problem", "nu_aniso_perp1", 0.0)}, //Pi^(1)
nu_aniso_perp2{pin->GetOrAddReal("problem", "nu_aniso_perp2", 0.0)}, //Pi^(2)
nu_aniso_wedge1{pin->GetOrAddReal("problem", "nu_aniso_wedge1", 0.0)}, //Pi^(3)
nu_aniso_wedge2{pin->GetOrAddReal("problem", "nu_aniso_wedge2", 0.0)}, //Pi^(4)

These are access via: These are accessed via:
nu(HydroDiffusion::DiffProcess::nu_neutral,k,j,i),
nu(HydroDiffusion::DiffProcess::nu_para,k,j,i),
nu(HydroDiffusion::DiffProcess::nu_perp1,k,j,i),
nu(HydroDiffusion::DiffProcess::nu_perp2,k,j,i),
nu(HydroDiffusion::DiffProcess::nu_wedge1,k,j,i), 
nu(HydroDiffusion::DiffProcess::nu_wedge2,k,j,i).

Due to the constants out the front, it is recomended to calculate actual parallel and wedge kinemtic viscosity coefficents and pass the same value to perp1 and perp2, and wedge1 and wedge2.
nu_neutral, nu_para, nu_perp1, nu_perp2, nu_wedge1, and nu_wedge2 must be supplied as diffusivities with dimensions of L^{2} T^{-1}. The same as isotropic case.
As with thermal conduction (1- \text{HII}_{\text{frac}})\boldsymbol{\Pi}_{\text{neu}} is the isotropic viscosity weighted by ionisation fraction.
If neutral viscosity is defined in the input file then HydroDiffusion::ion_frac must also be passed in EnrollViscosityCoefficient.
If neutral viscosity is not activated then full ionisation is assumed and anisotropic parts will not be weighted by ionisation fraction.

BULK VISCOSITY
Bulk viscosity is implemented as: \boldsymbol{\Pi}_{\text{bulk}} = \zeta(\nabla\cdot\vv{v})\boldsymbol{I}. It can be activated via: zeta_bulk = pin->GetOrAddReal("problem", "zeta_bulk", 0.0).
It is passed to HydroDiffusion::zeta_bulk in the problem generator in the EnrollViscosityCoefficient function.
As bulk viscosity only effects molecules. It is recomended to weight by molecular mass fractions if the simulation tracks thermodynamics states or chemistry and enters partially atomic, fully atomic, partially ionised or fully ionised regimes.
Bulk viscoisty should be pssed in units of $L$
