//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file hydro_diffusion.cpp
//! \brief Class to implement diffusion processes in the hydro equations

// C headers

// C++ headers
#include <algorithm>  // min()
#include <cstring>    // strcmp()
#include <iostream>   // endl
#include <limits>
#include <sstream>    // sstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../../coordinates/coordinates.hpp"
#include "../../eos/eos.hpp"   // GetGamma()
#include "../../field/field.hpp"
#include "../../mesh/mesh.hpp"
#include "../../parameter_input.hpp"
#include "../hydro.hpp"
#include "hydro_diffusion.hpp"

//! HydroDiffusion constructor
HydroDiffusion::HydroDiffusion(Hydro *phyd, ParameterInput *pin) :
  hydro_diffusion_defined(false),
  nu_iso{pin->GetOrAddReal("problem", "nu_iso", 0.0)}, // isotropic
  nu_aniso_neutral{pin->GetOrAddReal("problem", "nu_aniso_neutral", 0.0)}, // neutral
  nu_aniso_para{pin->GetOrAddReal("problem", "nu_aniso_para", 0.0)}, // para Pi^(0)
  nu_aniso_perp1{pin->GetOrAddReal("problem", "nu_aniso_perp1", 0.0)}, // perp Pi^(1)
  nu_aniso_perp2{pin->GetOrAddReal("problem", "nu_aniso_perp2", 0.0)}, // perp Pi^(2)
  nu_aniso_wedge1{pin->GetOrAddReal("problem", "nu_aniso_wedge1", 0.0)}, //wedge Pi^(3)
  nu_aniso_wedge2{pin->GetOrAddReal("problem", "nu_aniso_wedge2", 0.0)}, // wedge Pi^(4)
  zeta_bulk{pin->GetOrAddReal("problem", "zeta_bulk", 0.0)}, // bulk

  pmy_hydro_(phyd), pmb_(pmy_hydro_->pmy_block), pco_(pmb_->pcoord) {
  int nc1 = pmb_->ncells1, nc2 = pmb_->ncells2, nc3 = pmb_->ncells3;


  if(nu_aniso_neutral  > 0.0 || nu_aniso_para  > 0.0 || nu_aniso_perp1  > 0.0
      || nu_aniso_perp2  > 0.0 || nu_aniso_wedge1  > 0.0 || nu_aniso_wedge2  > 0.0) {
      aniso_viscosity = true;
    }

  if(nu_iso > 0.0 && aniso_viscosity) {
    std::stringstream msg;
    msg << "### FATAL ERROR in HydroDiffusion" << std::endl
    << "Isotropic and ansiotropic viscosity cannot be activavted together"
    << std::endl;
    ATHENA_ERROR(msg);
  }

  // Check if viscous process are active
  if (nu_iso > 0.0 || aniso_viscosity || zeta_bulk  > 0.0) {
    hydro_diffusion_defined = true;
    viscosity_on = true;
    visc_aniso_para   = (nu_aniso_neutral > 0.0 || nu_aniso_para > 0.0
                         || nu_aniso_perp1 > 0.0 || nu_aniso_perp2 > 0.0);
    visc_disp_defined = (nu_aniso_wedge1 > 0.0 || nu_aniso_wedge2 > 0.0);
    visc_para_defined = (nu_iso > 0.0 || zeta_bulk > 0.0 || visc_aniso_para);
    // Allocate memory for fluxes
    visflx[X1DIR].NewAthenaArray(NHYDRO, nc3, nc2, nc1+1);
    visflx[X2DIR].NewAthenaArray(NHYDRO, nc3, nc2+1, nc1);
    visflx[X3DIR].NewAthenaArray(NHYDRO, nc3+1, nc2, nc1);
    if (nu_aniso_wedge1 > 0.0 || nu_aniso_wedge2 > 0.0) {
      visflx_disp[X1DIR].NewAthenaArray(NHYDRO, nc3, nc2, nc1+1);
      visflx_disp[X2DIR].NewAthenaArray(NHYDRO, nc3, nc2+1, nc1);
      visflx_disp[X3DIR].NewAthenaArray(NHYDRO, nc3+1, nc2, nc1);
    }
    x1area_.NewAthenaArray(nc1+1);
    x2area_.NewAthenaArray(nc1);
    x3area_.NewAthenaArray(nc1);
    x2area_p1_.NewAthenaArray(nc1);
    x3area_p1_.NewAthenaArray(nc1);
    vol_.NewAthenaArray(nc1);
    fx_.NewAthenaArray(nc1);
    fy_.NewAthenaArray(nc1);
    fz_.NewAthenaArray(nc1);
    div_vel_.NewAthenaArray(nc3, nc2, nc1);
    nu.NewAthenaArray(8, nc3, nc2, nc1);
    if (pmb_->pmy_mesh->ViscosityCoeff_ == nullptr) {
      CalcViscCoeff_ = ConstViscosity;
      ion_frac_const = pin->GetOrAddReal("problem", "ion_frac_const", 0.0);
  } else {
      CalcViscCoeff_ = pmb_->pmy_mesh->ViscosityCoeff_;
    }
  }

  // Check if thermal conduction is active
  if (NON_BAROTROPIC_EOS) {
    kappa_iso  = pin->GetOrAddReal("problem", "kappa_iso", 0.0);
    kappa_aniso_neutral = pin->GetOrAddReal("problem", "kappa_aniso_neutral", 0.0);
    kappa_aniso_para = pin->GetOrAddReal("problem", "kappa_aniso_para", 0.0);
    kappa_aniso_perp = pin->GetOrAddReal("problem", "kappa_aniso_perp", 0.0);
    kappa_aniso_wedge = pin->GetOrAddReal("problem", "kappa_aniso_wedge", 0.0);

    cnd_aniso_para   = (kappa_aniso_neutral > 0.0 || kappa_aniso_para > 0.0
                        || kappa_aniso_perp > 0.0);
    cnd_disp_defined = (kappa_aniso_wedge > 0.0);
    cnd_para_defined = (kappa_iso > 0.0 || cnd_aniso_para);

    if(kappa_aniso_neutral > 0.0 ||  kappa_aniso_para > 0.0 || kappa_aniso_perp > 0.0 ||
      kappa_aniso_wedge > 0.0) aniso_conduction = true;

    if(kappa_iso > 0.0 && aniso_conduction) {
        std::stringstream msg;
        msg << "### FATAL ERROR in HydroDiffusion" << std::endl
        << "Isotropic and ansiotropic thermal conduction cannot be activavted together"
        << std::endl;
        ATHENA_ERROR(msg);
    }

    if (kappa_iso > 0.0) {
      hydro_diffusion_defined = true;
      conduction_on = true;
      cndflx[X1DIR].NewAthenaArray(nc3, nc2, nc1+1);
      cndflx[X2DIR].NewAthenaArray(nc3, nc2+1, nc1);
      cndflx[X3DIR].NewAthenaArray(nc3+1, nc2, nc1);
      kappa.NewAthenaArray(1, nc3, nc2, nc1);
      if (pmb_->pmy_mesh->ConductionCoeff_ == nullptr)
        CalcCondCoeff_ = ConstConduction;
      else
        CalcCondCoeff_ = pmb_->pmy_mesh->ConductionCoeff_;
    }

    if (aniso_conduction) {
      hydro_diffusion_defined = true;
      conduction_on = true;
      cndflx[X1DIR].NewAthenaArray(nc3, nc2, nc1+1);
      cndflx[X2DIR].NewAthenaArray(nc3, nc2+1, nc1);
      cndflx[X3DIR].NewAthenaArray(nc3+1, nc2, nc1);
      if(kappa_aniso_wedge > 0.0) {
        cndflx_disp[X1DIR].NewAthenaArray(nc3, nc2, nc1+1);
        cndflx_disp[X2DIR].NewAthenaArray(nc3, nc2+1, nc1);
        cndflx_disp[X3DIR].NewAthenaArray(nc3+1, nc2, nc1);
      }
      kappa.NewAthenaArray(5, nc3, nc2, nc1);

      if (pmb_->pmy_mesh->ConductionCoeff_ == nullptr) {
        CalcCondCoeff_ = ConstConduction;
        ion_frac_const = pin->GetOrAddReal("problem", "ion_frac_const", 0.0);}
      else
        CalcCondCoeff_ = pmb_->pmy_mesh->ConductionCoeff_;
      }
  }

  if (hydro_diffusion_defined && RELATIVISTIC_DYNAMICS) {
    std::stringstream msg;
    msg << "### FATAL ERROR in HydroDiffusion" << std::endl
        << "Diffusion is incompatibile with relativistic dynamics" << std::endl;
    ATHENA_ERROR(msg);
  }

  if (hydro_diffusion_defined) {
    dx1_.NewAthenaArray(nc1);
    dx2_.NewAthenaArray(nc1);
    dx3_.NewAthenaArray(nc1);
    nu_tot_.NewAthenaArray(nc1);
    kappa_tot_.NewAthenaArray(nc1);
    nu_tot_disp_.NewAthenaArray(nc1);
    kappa_tot_disp_.NewAthenaArray(nc1);
    ion_frac.NewAthenaArray(nc3, nc2, nc1);
    inv_dt_cnd.NewAthenaArray(nc3,nc2,nc1);
    inv_dt_cnd_disp.NewAthenaArray(nc3,nc2,nc1);
    inv_dt_visc.NewAthenaArray(nc3,nc2,nc1);
    inv_dt_vis_disp.NewAthenaArray(nc3,nc2,nc1);
  }

  if ((kappa_aniso_wedge > 0.0 || nu_aniso_wedge1 > 0.0 || nu_aniso_wedge2 > 0.0)
        && (pin->GetString("time","integrator") != "rk3" &&
        pin->GetString("time","integrator") != "rk4" &&
        pin->GetString("time","integrator") !="ssprk5_4")
        && Globals::my_rank == 0 && pmb_->gid == 0) {
        std::cout
    << "### Warning in FieldDiffusion" << std::endl
    << "Field dispersion is enabled with less than RK3 time integration." << std::endl
    << "Numerical stability not guaranteed." << std::endl
    << "Current time integrator is " << pin->GetString("time","integrator") <<std::endl;
  }

  if((kappa_aniso_wedge > 0.0 || nu_aniso_wedge1 > 0.0 || nu_aniso_wedge2 > 0.0)
      && (Globals::my_rank == 0 && pmb_->gid ==0) && RIEMANN_SOLVER != "hlle_mod") {
    std::cout
    << "### Warning in HydroDiffusion" << std::endl
    << "Hydro dispersion is active without modified Riemann solver signal speeds"
    << std::endl
    << "Grid scale instabilies may occur"
    << std::endl;
  }

  // Check if any parabolic terms are on
  hydro_para_defined = visc_para_defined || cnd_para_defined;
  // Check if any dispersive terms are on
  hydro_disp_defined = visc_disp_defined || cnd_disp_defined;
}

//----------------------------------------------------------------------------------------
//! \fn void HydroDiffusion::CalcDiffusionFlux
//! \brief Wrapper function for calculating local diffusivity coefficients and then
//!  diffusion fluxes (of various flavors) for overall hydro flux.

void HydroDiffusion::CalcDiffusionFlux(const AthenaArray<Real> &prim,
                                       const AthenaArray<Real> &iprim,
                                       const AthenaArray<Real> &bcc,
                                       bool calc_para,
                                       bool calc_disp) {
  calc_para_ = calc_para;
  calc_disp_ = calc_disp;
  SetDiffusivity(prim, bcc);

  if (viscosity_on) {
    if (calc_para_) ClearFlux(visflx);
    if (calc_disp_ && visc_disp_defined) ClearFlux(visflx_disp);
    if (calc_para_ && nu_iso > 0.0) ViscousFluxIso(prim, iprim, visflx);
    if ((calc_para_ && visc_aniso_para) || (calc_disp_ && visc_disp_defined))
      ViscousFluxAniso(prim, iprim, visflx, visflx_disp);
    if (calc_para_ && zeta_bulk > 0.0) ViscousFluxBulk(prim, iprim, visflx);
  }
  if (conduction_on) {
    if (calc_para_) ClearFlux(cndflx);
    if (calc_disp_ && cnd_disp_defined) ClearFlux(cndflx_disp);
    if (calc_para_ && kappa_iso > 0.0) ThermalFluxIso(prim, cndflx);
    if ((calc_para_ && cnd_aniso_para) || (calc_disp_ && cnd_disp_defined))
      ThermalFluxAniso(prim, cndflx, cndflx_disp);
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void HydroDiffusion::AddDiffusionEnergyFlux
//! \brief Adds only diffusion energy flux to hydro flux
//
// TODO(felker): completely general operation for any 2x AthenaArray<Real>
// flux[3]. Replace calls with below "AddDiffusionFlux" on shallow-sliced inputs.

void HydroDiffusion::AddDiffusionEnergyFlux(AthenaArray<Real> *flux_src,
                                            AthenaArray<Real> *flux_des) {
  int is = pmb_->is; int js = pmb_->js; int ks = pmb_->ks;
  int ie = pmb_->ie; int je = pmb_->je; int ke = pmb_->ke;

  AthenaArray<Real> &x1flux = flux_des[X1DIR];
  AthenaArray<Real> &x2flux = flux_des[X2DIR];
  AthenaArray<Real> &x3flux = flux_des[X3DIR];
  AthenaArray<Real> &x1diflx = flux_src[X1DIR];
  AthenaArray<Real> &x2diflx = flux_src[X2DIR];
  AthenaArray<Real> &x3diflx = flux_src[X3DIR];

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
    #pragma omp simd
      for (int i=is; i<=ie; ++i) {
        x1flux(IEN,k,j,i) += x1diflx(k,j,i);
        if (i == ie) x1flux(IEN,k,j,i+1) += x1diflx(k,j,i+1);
        if (pmb_->block_size.nx2 > 1) {
          x2flux(IEN,k,j,i) += x2diflx(k,j,i);
          if (j == je) x2flux(IEN,k,j+1,i) += x2diflx(k,j+1,i);
        }
        if (pmb_->block_size.nx3 > 1) {
          x3flux(IEN,k,j,i) += x3diflx(k,j,i);
          if (k == ke) x3flux(IEN,k+1,j,i) += x3diflx(k+1,j,i);
        }
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void HydroDiffusion::AddDiffusionFlux
//!  \brief Adds all componenets of diffusion flux to hydro flux
//
// TODO(felker): completely general operation for any 2x AthenaArray<Real> flux[3]. Move
// from this class, and remove "Diffusion" from name.

void HydroDiffusion::AddDiffusionFlux(AthenaArray<Real> *flux_src,
                                      AthenaArray<Real> *flux_des) {
  int size1 = flux_des[X1DIR].GetSize();
#pragma omp simd
  for (int i=0; i<size1; ++i)
    flux_des[X1DIR](i) += flux_src[X1DIR](i);

  if (pmb_->block_size.nx2 > 1) {
    int size2 = flux_des[X2DIR].GetSize();
#pragma omp simd
    for (int i=0; i<size2; ++i)
      flux_des[X2DIR](i) += flux_src[X2DIR](i);
  }
  if (pmb_->block_size.nx3 > 1) {
    int size3 = flux_des[X3DIR].GetSize();
#pragma omp simd
    for (int i=0; i<size3; ++i)
      flux_des[X3DIR](i) += flux_src[X3DIR](i);
  }
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void HydroDiffusion::ClearFlux
//! \brief Reset diffusion flux back to zeros
//
// TODO(felker): completely general operation for any 1x AthenaArray<Real> flux[3]. Move
// from this class.

void HydroDiffusion::ClearFlux(AthenaArray<Real> *flux) {
  flux[X1DIR].ZeroClear();
  flux[X2DIR].ZeroClear();
  flux[X3DIR].ZeroClear();
  return;
}

//---------------------------------------------------------------------------------------
//! \fn void HydroDiffusion::SetDiffusivity(const Athena<Real> &w,
//!                                          const AthenaArray<Real> &bc)
//! \brief  Set local hydro diffusion coefficients based on the fluid and
//! magnetic field variables across the mesh.
//! Called within HydroDiffusion::CalcDiffusionFlux wrapper function.

void HydroDiffusion::SetDiffusivity(const AthenaArray<Real> &w,
                                    const AthenaArray<Real> &bc) {
  int il = pmb_->is - NGHOST; int jl = pmb_->js; int kl = pmb_->ks;
  int iu = pmb_->ie + NGHOST; int ju = pmb_->je; int ku = pmb_->ke;
  if (pmb_->block_size.nx2 > 1) {
    jl -= NGHOST; ju += NGHOST;
  }
  if (pmb_->block_size.nx3 > 1) {
    kl -= NGHOST; ku += NGHOST;
  }

  // set viscosity using func ptr
  if (viscosity_on)
    CalcViscCoeff_(this, pmb_, w, bc, il, iu, jl, ju, kl, ku);
  // set thermal conduction using func ptr
  if (conduction_on) {
    CalcCondCoeff_(this, pmb_, w, bc, il, iu, jl, ju, kl, ku);}

  return;
}

//---------------------------------------------------------------------------------------
//! \fn void HydroDiffusion::NewDiffusionDt(...)
//! \brief Get the hydro diffusion timestep for viscous and conduction processes

void HydroDiffusion::NewDiffusionDt(Real &dt_vis, Real &dt_cnd, Real &dt_vis_disp,
                                    Real &dt_cnd_disp, AthenaArray<Real> &inv_dt_cnd,
                                    AthenaArray<Real> &inv_dt_visc,
                                    AthenaArray<Real> &inv_dt_cnd_disp,
                                    AthenaArray<Real> &inv_dt_vis_disp) {
  Real real_max = std::numeric_limits<Real>::max();
  const bool f2 = pmb_->pmy_mesh->f2;
  const bool f3 = pmb_->pmy_mesh->f3;
  int il = pmb_->is - NGHOST; int jl = pmb_->js; int kl = pmb_->ks;
  int iu = pmb_->ie + NGHOST; int ju = pmb_->je; int ku = pmb_->ke;

  inv_dt_cnd.ZeroClear();
  inv_dt_visc.ZeroClear();
  inv_dt_cnd_disp.ZeroClear();
  inv_dt_vis_disp.ZeroClear();
  Real FOUR_THIRDS = 1.333333333333333;
  Real fac = 0; Real fac_RL = 0; Real fac_gyro = 0;
  Real len = 0; Real len_perp = 0;
  Real p1 = 0; Real p2 = 0; Real p3 = 0;
  Real zr = 0; Real HII = 0; Real HII_cnd = 0; Real nu_max = 0;

  // parabolic (RK1+):  1/(2 N_dim)
  if (f3)      fac = 1.0/6.0;
  else if (f2) fac = 0.25;
  else         fac = 0.5;

  // Righi-Leduc (wedge) conduction (RK3+) sqrt(3)/2d
  if (f3)      fac_RL = 0.2886751346;
  else if (f2) fac_RL = 0.4330127019;
  else         fac_RL = 0.8660254038;

  // gyro-viscosity (RK3+) sqrt(3)/4d
  if (f3)      fac_gyro = 0.1443375673;
  else if (f2) fac_gyro = 0.2165063509;
  else         fac_gyro = 0.4330127019;

  Real gamma = pmb_->peos->GetGamma();
  Real gm1   = gamma - 1.0;

  dt_vis = real_max;
  dt_cnd = real_max;

  // dispersive/hyperoblic dt for wedge elements
  dt_vis_disp = real_max;
  dt_cnd_disp = real_max;

  AthenaArray<Real> &nu_t = nu_tot_;
  AthenaArray<Real> &kappa_t = kappa_tot_;
  AthenaArray<Real> &nu_t_disp = nu_tot_disp_;
  AthenaArray<Real> &kappa_t_disp = kappa_tot_disp_;
  AthenaArray<Real> &dx1 = dx1_, &dx2 = dx2_, &dx3 = dx3_;

  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      pmb_->pcoord->CenterWidth1(k, j, il, iu, dx1);
      pmb_->pcoord->CenterWidth2(k, j, il, iu, dx2);
      pmb_->pcoord->CenterWidth3(k, j, il, iu, dx3);
      for (int i=il; i<=iu; ++i) {
        len = dx1(i);
        if (f2) len = std::min(len, dx2(i));
        if (f3) len = std::min(len, dx3(i));
        if (conduction_on) {
          kappa_t(i) = (kappa_iso > 0.0) ? kappa(DiffProcess::iso,k,j,i) : 0.0;
          kappa_t_disp(i) = 0.0;
          if (aniso_conduction) {
            HII_cnd = (kappa_aniso_neutral == 0.0 ? 1.0 : ion_frac(k,j,i));
            kappa_t(i) += (1.0-HII_cnd)*kappa(DiffProcess::kappa_neutral,k,j,i)
                       + HII_cnd*std::max(kappa(DiffProcess::kappa_para,k,j,i),
                                          kappa(DiffProcess::kappa_perp,k,j,i));
            if (kappa_aniso_wedge > 0.0) {
              kappa_t_disp(i) = HII_cnd*kappa(DiffProcess::kappa_wedge,k,j,i);
              kappa_t_disp(i) *= gm1;
              p1 = dx1(i)/(1.0/dx2(i) + 1.0/dx3(i));
              p2 = dx2(i)/(1.0/dx1(i) + 1.0/dx3(i));
              p3 = dx3(i)/(1.0/dx1(i) + 1.0/dx2(i));
              len_perp = std::min(p1, std::min(p2, p3));
              inv_dt_cnd_disp(k,j,i) += 1.0 / (static_cast<Real>(
                                               len_perp*fac_RL/(kappa_t_disp(i)
                                                + TINY_NUMBER)));
              inv_dt_cnd(k,j,i) += inv_dt_cnd_disp(k,j,i);
              dt_cnd_disp = std::min(dt_cnd_disp, 1.0 / inv_dt_cnd_disp(k,j,i));
              }
            }
        kappa_t(i) *= gm1;
        inv_dt_cnd(k,j,i) += 1.0 / (static_cast<Real>(
                                  SQR(len)*fac/(kappa_t(i) + TINY_NUMBER)));
        dt_cnd = std::min(dt_cnd, static_cast<Real>(
                                  SQR(len)*fac/(kappa_t(i) + TINY_NUMBER)));
        }

        if (viscosity_on) {
          zr = (zeta_bulk > 0.0)
                ? nu(DiffProcess::zeta,k,j,i)/pmb_->phydro->w(IDN,k,j,i) : 0.0;
        nu_max = (nu_iso > 0.0) ? FOUR_THIRDS*nu(DiffProcess::iso,k,j,i) : 0.0;
        if (aniso_viscosity) {
          HII = (nu_aniso_neutral == 0.0) ? 1.0 : ion_frac(k,j,i);
          nu_max += ViscAnisoEigenMax(HII, nu(DiffProcess::nu_neutral,k,j,i),
                                    nu(DiffProcess::nu_para,k,j,i),
                                    nu(DiffProcess::nu_perp1,k,j,i),
                                    nu(DiffProcess::nu_perp2,k,j,i), zr);
          if (nu_aniso_wedge1 > 0.0 || nu_aniso_wedge2 > 0.0) {
            nu_t_disp(i) = GyroViscEigenMax(HII, nu(DiffProcess::nu_wedge1,k,j,i),
                                                  nu(DiffProcess::nu_wedge2,k,j,i));
            inv_dt_vis_disp(k,j,i) += 1.0 / (static_cast<Real>(
                                      SQR(len)*fac_gyro/(nu_t_disp(i) + TINY_NUMBER)));
            inv_dt_visc(k,j,i) += inv_dt_vis_disp(k,j,i);
            dt_vis_disp = std::min(dt_vis_disp, 1.0 / inv_dt_vis_disp(k,j,i));
          }
        } else {
          nu_max += zr;
        }
        nu_t(i) = nu_max;
        inv_dt_visc(k,j,i) += 1.0 / (static_cast<Real>(
                                      SQR(len)*fac/(nu_t(i) + TINY_NUMBER)));
        dt_vis = std::min(dt_vis, (static_cast<Real>(
                                    SQR(len)*fac/(nu_t(i) + TINY_NUMBER))));
      }
    }
  }
}
return;
}

//----------------------------------------------------------------------------------------
//! \fn Real ViscAnisoEigen(...)
//! \brief Calculates the eigenvalue for anisotropic viscosity
inline Real HydroDiffusion::ViscAnisoEigen(const Real u, const Real HII,
                                           const Real nu_neu, const Real nu_par,
                                           const Real nu_p1, const Real nu_p2,
                                           const Real zr) {
  const Real um = 1.0 - u;
  const Real tr = zr + (7.0/3.0)*(1.0 - HII)*nu_neu
                + HII*(nu_par*(u + 1.0/3.0) + nu_p1*um + 4.0*nu_p2);
  const Real det =
        zr*(1.0 - HII)*nu_neu + (4.0/3.0)*SQR((1.0 - HII))*SQR(nu_neu)
      + HII*( zr*((3.0*nu_par + nu_p1)*u*um + 4.0*nu_p2*SQR(2.0*u - 1.0))
          + (1.0 - HII)*nu_neu*(3.0*nu_par*SQR(u - 1.0/3.0) + nu_p1*SQR(um)
                       + 16.0*nu_p2*u*um)
          + (4.0/3.0)*(1.0 - HII)*nu_neu*((3.0*nu_par + nu_p1)*u*um
                                 + 4.0*nu_p2*SQR(2.0*u - 1.0)) )
      + (4.0/3.0)*SQR(HII)*(nu_par*nu_p1*u*um + nu_par*nu_p2*SQR(u + 1.0)
                          + 3.0*nu_p1*nu_p2*SQR(um));
  Real disc = SQR(tr) - 4.0*det;
  return 0.5*(tr + std::sqrt(disc));
}

//----------------------------------------------------------------------------------------
//! \fn Real ViscAnisoEigenMax(...)
//! \brief Calculates the maximum eigenvalue for anisotropic viscoisty with u in [0,1]
Real HydroDiffusion::ViscAnisoEigenMax(const Real HII, const Real nu_neu,
                                       const Real nu_par, const Real nu_p1,
                                       const Real nu_p2, const Real zr) {
  Real lam = std::max(ViscAnisoEigen(0.0, HII, nu_neu, nu_par, nu_p1, nu_p2, zr),
                      ViscAnisoEigen(1.0, HII, nu_neu, nu_par, nu_p1, nu_p2, zr));
  const Real denom1 = 3.0*nu_par + nu_p1 - 16.0*nu_p2;
  if (std::fabs(denom1) > 0.0) { // u star 1
    const Real us1 = (nu_par + nu_p1 - 8.0*nu_p2)/denom1;
    if (us1 >= 0.0 && us1 <= 1.0)
      lam = std::max(lam, ViscAnisoEigen(us1, HII, nu_neu, nu_par, nu_p1, nu_p2, zr));
  }
  const Real denom2 = 4.0*(1.0 - HII)*nu_neu + (nu_par + 3.0*nu_p1)*HII + 12.0*zr;
  if (std::fabs(denom2) > 0.0) { // u star 2
    const Real us2 = ((2.0 - 2.0*HII)*nu_neu + (3.0*nu_p1 - nu_par)*HII + 6.0*zr)/denom2;
    if (us2 >= 0.0 && us2 <= 1.0)
      lam = std::max(lam, ViscAnisoEigen(us2, HII, nu_neu, nu_par, nu_p1, nu_p2, zr));
  }
  return lam;
}

//----------------------------------------------------------------------------------------
//! \fn Real GyroViscEigenMax(...)
//! \brief Calculate the maximum eigenvalue for gyro-vsicosity for u in [0,1]
Real HydroDiffusion::GyroViscEigenMax(const Real HII, const Real nu_w1,
                                      const Real nu_w2) {
  Real lam = std::max(HII*nu_w1, 2.0*HII*nu_w2);
  const Real denom = nu_w1*(4.0*nu_w2 - nu_w1);
  if (denom > 0.0) {
    lam = std::max(lam, 2.0*HII*std::sqrt(nu_w2*nu_w2*nu_w2*(2.0*nu_w1 + nu_w2)/denom));
  }
  return lam;
}
