//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file field_diffusion.cpp
//! \brief Class to implement diffusion processes in the induction equations

// C headers

// C++ headers
#include <algorithm>  // min()
#include <cmath>      // sqrt(), abs()
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
#include "../../eos/eos.hpp" // Gamma
#include "../../hydro/hydro.hpp"
#include "../../hydro/hydro_diffusion/hydro_diffusion.hpp" // Ionisation fraction
#include "../../mesh/mesh.hpp"
#include "../../parameter_input.hpp"
#include "../field.hpp"
#include "field_diffusion.hpp"

//! FieldDiffusion constructor

FieldDiffusion::FieldDiffusion(MeshBlock *pmb, ParameterInput *pin) :

    pmy_block(pmb), field_diffusion_defined(false) {
  int nc1 = pmb->ncells1, nc2 = pmb->ncells2, nc3 = pmb->ncells3;
  // Check if field diffusion
  eta_ohm = pin->GetOrAddReal("problem", "eta_ohm", 0.0);
  eta_hall = pin->GetOrAddReal("problem", "eta_hall", 0.0);
  eta_ad = pin->GetOrAddReal("problem", "eta_ad", 0.0);
  kappa_aniso_TE_para = pin->GetOrAddReal("problem", "kappa_aniso_TE_para", 0.0);
  kappa_aniso_TE_perp = pin->GetOrAddReal("problem", "kappa_aniso_TE_perp", 0.0);
  kappa_aniso_TE_wedge = pin->GetOrAddReal("problem", "kappa_aniso_TE_wedge", 0.0);
  debug_therm_conduct_only = pin->GetOrAddBoolean("problem",
                                                  "debug_therm_conduct_only", false);
  debug_therm_force_only = pin->GetOrAddBoolean("problem",
                                                "debug_therm_force_only", false);


  if ((eta_ohm > 0.0) || (eta_hall > 0.0) || (eta_ad > 0.0)
      || (kappa_aniso_TE_para > 0.0) || (kappa_aniso_TE_perp > 0.0)
      || (kappa_aniso_TE_wedge > 0.0)) {
    field_diffusion_defined = true;

    // Allocate memory for scratch vectors
    etaB.NewAthenaArray(3, nc3, nc2, nc1);
    thermKappa.NewAthenaArray(3,nc3,nc2,nc1);
    e_oa.x1e.NewAthenaArray(nc3+1, nc2+1, nc1);
    e_oa.x2e.NewAthenaArray(nc3+1, nc2, nc1+1);
    e_oa.x3e.NewAthenaArray(nc3, nc2+1, nc1+1);
    e_h.x1e.NewAthenaArray(nc3+1, nc2+1, nc1);
    e_h.x2e.NewAthenaArray(nc3+1, nc2, nc1+1);
    e_h.x3e.NewAthenaArray(nc3, nc2+1, nc1+1);
    pflux.x1f.NewAthenaArray(nc3, nc2, nc1+1);
    pflux.x2f.NewAthenaArray(nc3, nc2+1, nc1);
    pflux.x3f.NewAthenaArray(nc3+1, nc2, nc1);
    pflux_disp.x1f.NewAthenaArray(nc3, nc2, nc1+1);
    pflux_disp.x2f.NewAthenaArray(nc3, nc2+1, nc1);
    pflux_disp.x3f.NewAthenaArray(nc3+1, nc2, nc1);

    jfx.NewAthenaArray(3, nc3+1, nc2+1, nc1+1);
    jfy.NewAthenaArray(3, nc3+1, nc2+1, nc1+1);
    jfz.NewAthenaArray(3, nc3+1, nc2+1, nc1+1);
    jcc.NewAthenaArray(3, nc3+1, nc2+1, nc1+1);

    eta_tot_.NewAthenaArray(nc1);
    bmag_.NewAthenaArray(nc3, nc2, nc1);

    jedge_.x1e.NewAthenaArray(nc3+1, nc2+1, nc1);
    jedge_.x2e.NewAthenaArray(nc3+1, nc2, nc1+1);
    jedge_.x3e.NewAthenaArray(nc3, nc2+1, nc1+1);

    face_area_.NewAthenaArray(nc1);
    face_area_p1_.NewAthenaArray(nc1);
    edge_length_.NewAthenaArray(nc1);
    edge_length_m1_.NewAthenaArray(nc1);
    cell_volume_.NewAthenaArray(nc1);
    dx1_.NewAthenaArray(nc1);
    dx2_.NewAthenaArray(nc1);
    dx3_.NewAthenaArray(nc1);

    inv_dt_OAH.NewAthenaArray(nc3,nc2,nc1);
    inv_dt_TE.NewAthenaArray(nc3,nc2,nc1);
    inv_dt_TE_hyp.NewAthenaArray(nc3,nc2,nc1);

    // Thermoelectric effects
    use_TE_wedge = (kappa_aniso_TE_wedge > 0.0);
    use_TE_para  = (kappa_aniso_TE_para  > 0.0);
    use_TE_perp  = (kappa_aniso_TE_perp  > 0.0);

    field_oa_defined   = (eta_ohm > 0.0 || eta_ad > 0.0 || use_TE_wedge);
    field_disp_defined = (eta_hall > 0.0 || use_TE_para || use_TE_perp);

    if (use_TE_para || use_TE_perp || use_TE_wedge) {
      Tcc.NewAthenaArray(nc3, nc2, nc1);
      Te1.NewAthenaArray(nc3+1, nc2+1, nc1+1);
      Te2.NewAthenaArray(nc3+1, nc2+1, nc1+1);
      Te3.NewAthenaArray(nc3+1, nc2+1, nc1+1);

      teflx[X1DIR].NewAthenaArray(nc3, nc2, nc1+1);
      teflx[X2DIR].NewAthenaArray(nc3, nc2+1, nc1);
      teflx[X3DIR].NewAthenaArray(nc3+1, nc2, nc1);

      teflx_disp[X1DIR].NewAthenaArray(nc3, nc2, nc1+1);
      teflx_disp[X2DIR].NewAthenaArray(nc3, nc2+1, nc1);
      teflx_disp[X3DIR].NewAthenaArray(nc3+1, nc2, nc1);
    }

    if (pmb->pmy_mesh->FieldDiffusivity_ == nullptr)
      CalcMagDiffCoeff_ = ConstDiffusivity;
    else
      CalcMagDiffCoeff_ = pmb->pmy_mesh->FieldDiffusivity_;
  }

  if (field_diffusion_defined && RELATIVISTIC_DYNAMICS) {
    std::stringstream msg;
    msg << "### FATAL ERROR in FieldDiffusion" << std::endl
        << "Diffusion is incompatibile with relativistic dynamics" << std::endl;
    ATHENA_ERROR(msg);
  }

  // Both conduction and force must be on for Onsager symmetry
  if (debug_therm_conduct_only && debug_therm_force_only
      && Globals::my_rank == 0 && pmb->gid == 0) {
  std::stringstream msg;
  msg << "### FATAL ERROR in FieldDiffusion" << std::endl
      << "Both thermoelectric diffusion debug flags are true" << std::endl
      << "Set both flags to false (or remove them from the input file) "
      << "to activate both thermo-electric conduction and thermo-electric force."
      << std::endl;
  ATHENA_ERROR(msg);
  }

  if (debug_therm_conduct_only && !debug_therm_force_only
      && Globals::my_rank == 0 && pmb->gid == 0) {
  std::cout
    << "### Warning in FieldDiffusion" << std::endl
    << "Only thermoelectric conduction (without thermo-electric force) is enabled"
    << std::endl
    << "This breaks Onsager symmetry and should only be used for debugging/testing"
    << std::endl;
  }

  if (!debug_therm_conduct_only && debug_therm_force_only
      && Globals::my_rank == 0 && pmb->gid == 0) {
  std::cout
    << "### Warning in FieldDiffusion" << std::endl
    << "Only thermoelectric force (without thermo-electric conduction) is enabled"
    << std::endl
    << "This breaks Onsager symmetry and should only be used for debugging/testing"
    << std::endl;
  }

  if ((eta_hall > 0.0 || kappa_aniso_TE_para > 0.0 || kappa_aniso_TE_perp > 0.0)
                           && (pin->GetString("time","integrator") != "rk3" &&
                           pin->GetString("time","integrator") != "rk4" &&
                           pin->GetString("time","integrator") !="ssprk5_4")
      && Globals::my_rank == 0 && pmb->gid == 0) {
  std::cout
    << "### Warning in FieldDiffusion" << std::endl
    << "Field dispersion is enabled with less than RK3 time integration." << std::endl
    << "Numerical stability not guaranteed." << std::endl
    << "Current time integrator is " << pin->GetString("time","integrator") <<std::endl;
  }

  if (((kappa_aniso_TE_para > 0.0) || (kappa_aniso_TE_perp > 0.0) || (eta_hall > 0.0))
      && Globals::my_rank == 0 && pmb->gid == 0 && RIEMANN_SOLVER != "hlle_mod") {
    std::cout
    << "### Warning in FieldDiffusion" << std::endl
    << "Field dispersion is active without modified Riemann solver signal speeds"
    << std::endl
    << "Grid scale instabilies may occur"
    << std::endl;
  }

  if (((kappa_aniso_TE_para > 0.0) || (kappa_aniso_TE_perp > 0.0)
      || (kappa_aniso_TE_wedge > 0.0)) && Globals::my_rank == 0 && pmb->gid == 0) {
    std::cout
    << "### Caution in FieldDiffusion" << std::endl
    << "If using the wedge thermoelectric term ensure that perpendicular"
    <<"(and/or neutral) thermal conduction and ohmic and/or ambipolar diffusion are on"
    << std::endl;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void FieldDiffusion::DiffusionEMF
//! \brief Calculate diffusion EMF(Ohmic & Ambipolar for now)

void FieldDiffusion::CalcDiffusionEMF(FaceField &bi, const AthenaArray<Real> &bc,
                                      EdgeField &e) {
  Field *pf = pmy_block->pfield;
  Hydro *ph = pmy_block->phydro;

  if ((eta_ohm == 0.0) && (eta_ad == 0.0) && (eta_hall == 0.0)
  && (kappa_aniso_TE_para == 0) && (kappa_aniso_TE_perp == 0)
  && (kappa_aniso_TE_wedge == 0)) return;

  if(!NON_BAROTROPIC_EOS && ( use_TE_para || use_TE_perp ||use_TE_wedge)) {
    std::stringstream msg;
    msg << "### FATAL ERROR in FieldDiffusion" << std::endl
        << "Thermoelectric effects are incompatible with isothermal MHD" << std::endl;
    ATHENA_ERROR(msg);}

  SetDiffusivity(ph->w, pf->bcc);
  CalcCurrent(bi);

  if (eta_ohm > 0.0 || eta_ad > 0.0 || use_TE_wedge) {
    ClearEMF(e_oa);
    if (eta_ohm > 0.0) OhmicEMF(bi, bc, e_oa);
    if (eta_ad  > 0.0) AmbipolarEMF(bi, bc, e_oa);
  }
  if (eta_hall > 0.0 || use_TE_para || use_TE_perp) {
    ClearEMF(e_h);
    if (eta_hall > 0.0) HallEMF(bi, bc, e_h);
  }

  if (NON_BAROTROPIC_EOS) {
    // add TE.
    if ((use_TE_para) || (use_TE_perp) || (use_TE_wedge)) {
        if (debug_therm_conduct_only && !debug_therm_force_only) {
          CalcEdgeTemperature(ph->w, Tcc, Te1, Te2, Te3);
          ph->hdif.ClearFlux(teflx); ph->hdif.ClearFlux(teflx_disp);
          ThermoElectricConduction(pmy_block, bi, bc, teflx, teflx_disp);
      } else if(!debug_therm_conduct_only && debug_therm_force_only) {
          CalcEdgeTemperature(ph->w, Tcc, Te1, Te2, Te3);
          ThermoElectricForce(pmy_block, bi, bc, e_oa, e_h);
      } else {
          CalcEdgeTemperature(ph->w, Tcc, Te1, Te2, Te3);
          ph->hdif.ClearFlux(teflx); ph->hdif.ClearFlux(teflx_disp);
          ThermoElectricConduction(pmy_block, bi, bc, teflx, teflx_disp);
          ThermoElectricForce(pmy_block, bi, bc, e_oa, e_h);}
   }
    // calculate the Poynting flux pflux and add to energy flux in Hydro class
    if (field_oa_defined) PoyntingFlux(e_oa, bc, pflux);
    if (field_disp_defined) PoyntingFlux(e_h, bc, pflux_disp);
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void FieldDiffusion::AddEMF(EdgeField &e_src, EdgeField &e_des)
//! \brief Add source EMF to destination EMF

void FieldDiffusion::AddEMF(const EdgeField &e_src, EdgeField &e_des) {
  int size1 = e_src.x1e.GetSize();
  int size2 = e_src.x2e.GetSize();
  int size3 = e_src.x3e.GetSize();

#pragma omp simd
  for (int i=0; i<size1; ++i)
    e_des.x1e(i) += e_src.x1e(i);

#pragma omp simd
  for (int i=0; i<size2; ++i)
    e_des.x2e(i) += e_src.x2e(i);

#pragma omp simd
  for (int i=0; i<size3; ++i)
    e_des.x3e(i) += e_src.x3e(i);

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void FieldDiffusion::ClearEMF(EdgeField &e)
//! \brief Clear EMF

// TODO(felker): move out of FieldDiffusion class. Completely general operation
void FieldDiffusion::ClearEMF(EdgeField &e) {
  e.x1e.ZeroClear();
  e.x2e.ZeroClear();
  e.x3e.ZeroClear();
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void FieldDiffusion::SetDiffusivity
//! \brief Set magnetic diffusion coefficients

void FieldDiffusion::SetDiffusivity(const AthenaArray<Real> &w,
                                    const AthenaArray<Real> &bc) {
  MeshBlock *pmb = pmy_block;
  int il = pmb->is - NGHOST; int jl = pmb->js; int kl = pmb->ks;
  int iu = pmb->ie + NGHOST; int ju = pmb->je; int ku = pmb->ke;
  if (pmb->pmy_mesh->f2) {
    jl -= NGHOST; ju += NGHOST;
  }
  if (pmb->pmy_mesh->f3) {
    kl -= NGHOST; ku += NGHOST;
  }

  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
        Real Bsq = SQR(bc(IB1,k,j,i)) + SQR(bc(IB2,k,j,i)) + SQR(bc(IB3,k,j,i));
        bmag_(k,j,i) = std::sqrt(Bsq);
      }
    }
  }
  // set diffusivities
  CalcMagDiffCoeff_(this, pmb, w, bmag_, il, iu, jl, ju, kl, ku);

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void FieldDiffusion::AddPoyntingFlux
//! \brief Add Poynting flux to the hydro energy flux

void FieldDiffusion::AddPoyntingFlux(FaceField &p_src) {
  MeshBlock *pmb = pmy_block;
  AthenaArray<Real> &x1flux = pmb->phydro->flux[X1DIR];
  AthenaArray<Real> &x2flux = pmb->phydro->flux[X2DIR];
  AthenaArray<Real> &x3flux = pmb->phydro->flux[X3DIR];
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  AthenaArray<Real> &f1 = p_src.x1f, &f2 = p_src.x2f, &f3 = p_src.x3f;

  // 1D update:
  if (pmb->block_size.nx2 == 1) {
#pragma omp simd
    for (int i=is; i<=ie+1; ++i) {
      x1flux(IEN,ks,js,i) += f1(ks,js,i);
    }
    return;
  }

  // 2D update:
  if (pmb->block_size.nx3 == 1) {
    for (int j=js; j<=je+1; ++j) {
#pragma omp simd
      for (int i=is; i<=ie+1; ++i) {
        x1flux(IEN,ks,j,i) += f1(ks,j,i);
        x2flux(IEN,ks,j,i) += f2(ks,j,i);
      }
    }
    return;
  }

  // 3D update:
  for (int k=ks; k<=ke+1; ++k) {
    for (int j=js; j<=je+1; ++j) {
#pragma omp simd
      for (int i=is; i<=ie+1; ++i) {
        x1flux(IEN,k,j,i) += f1(k,j,i);
        x2flux(IEN,k,j,i) += f2(k,j,i);
        x3flux(IEN,k,j,i) += f3(k,j,i);
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void FieldDiffusion::NewDiffusionDt
//! \brief Get the non-ideal MHD timestep

void FieldDiffusion::NewDiffusionDt(Real &dt_oa, Real &dt_h, Real &dt_OAH,
                                    Real &dt_TE_para, AthenaArray<Real> &inv_dt_OAH,
                                    AthenaArray<Real> &inv_dt_TE,
                                    AthenaArray<Real> &inv_dt_TE_hyp) {
  MeshBlock *pmb = pmy_block;
  Coordinates *pco = pmb->pcoord;
  AthenaArray<Real> &bc = pmb->pfield->bcc;
  HydroDiffusion &hdif_ = pmb->phydro->hdif;
  FaceField &b = pmb->pfield->b;
  const bool f2 = pmb->pmy_mesh->f2;
  const bool f3 = pmb->pmy_mesh->f3;

  inv_dt_OAH.ZeroClear();
  inv_dt_TE.ZeroClear();
  inv_dt_TE_hyp.ZeroClear();

  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  Real fac_oa = 0; Real fac_h = 0;

// Parabolic CFL stable to rk1+ (1/2d)
  if (f3) { // 3D
    fac_oa = 1.0/6.0;
  } else {
    if (f2) { // 2D
      fac_oa = 0.25;
    } else { // 1D
      fac_oa = 0.5;
    }
  }

// Hyperbolic CFL stable to rk3+ sqrt(3)/4d
  if (f3) { // 3D
    fac_h = 0.1443375673;
  } else {
    if (f2) { // 2D
      fac_h = 0.2165063509;
    } else { // 1D
      fac_h = 0.4330127019;
    }
  }

  Real real_max = std::numeric_limits<Real>::max();
  dt_oa = real_max;
  dt_h  = real_max;
  dt_OAH = real_max;
  dt_TE_para = real_max;
  Real gamma = pmb->peos->GetGamma();
  Real gm1   = gamma - 1.0;
  Real HII = 0;
  Real kappa_J_para = 0; Real kappa_J_perp = 0; Real kappa_J_wedge = 0;
  Real e_int = 0;
  Real lam_TE = 0; // unsplit/isolated TE coefficeint
  Real lam_TE_para = 0; // therm+ohmic+ambi+FT_w TE coefficeint
  Real lam_TE_hyp = 0; // Hall+FT_Para+FT_Perp TE coefficeint

  // thermal conduction for TE
  Real kappa0 = 0;
  Real kappa_n =0;
  Real kappa_para = 0;
  Real kappa_perp =  0;
  Real d_kappa = 0;

  AthenaArray<Real> &eta_t = eta_tot_;
  AthenaArray<Real> &len = dx1_, &dx2 = dx2_, &dx3 = dx3_;

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        eta_t(i) = 0.0;
      }
      if (eta_ohm > 0.0) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          eta_t(i) += etaB(DiffProcess::ohmic,k,j,i);
        }
      }
      if (eta_ad > 0.0) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          eta_t(i) += etaB(DiffProcess::ambipolar,k,j,i);
        }
      }
      pco->CenterWidth1(k, j, is, ie, len);
      pco->CenterWidth2(k, j, is, ie, dx2);
      pco->CenterWidth3(k, j, is, ie, dx3);
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        len(i) = (f2) ? std::min(len(i), dx2(i)):len(i);
        len(i) = (f3) ? std::min(len(i), dx3(i)):len(i);
      }
      if(STS_ENABLED){ // split ohmic+ambi (parabolic) and Hall (hyperbolic) CFLs
        if ((eta_ohm > 0.0) || (eta_ad > 0.0)) {
          for (int i=is; i<=ie; ++i){
            dt_oa = std::min(dt_oa, static_cast<Real>(
                fac_oa*SQR(len(i)) / (eta_t(i) + TINY_NUMBER)));
            } // i loop
        } // end ohmic+ambi dt
        if (eta_hall > 0.0) {
          for (int i=is; i<=ie; ++i){ // dt_h is one minima of dt_TE
            inv_dt_TE_hyp(k,j,i) = 1.0 / static_cast<Real>(fac_h*SQR(len(i)) /
                                                  (std::abs(etaB(DiffProcess::hall,k,j,i))
                                                  + TINY_NUMBER));
            dt_h = std::min(dt_h, 1.0 / inv_dt_TE_hyp(k,j,i));
            } // i loop
        } // end hall dt
      } else { // unsplit CFL (RK3 stable, RK1/2 is too restrictve)
        if ((eta_ohm > 0.0) || (eta_ad > 0.0) || (eta_hall > 0.0)) {
          for (int i=is; i<=ie; ++i) {
            Real etaH = (eta_hall > 0.0) ? std::abs(etaB(DiffProcess::hall,k,j,i)) : 0.0;
            dt_OAH = std::min(dt_OAH, static_cast<Real>(
                                      fac_h*SQR(len(i)) /
                                      (std::sqrt(eta_t(i)*eta_t(i)+ etaH*etaH)
                                      + TINY_NUMBER)));
            inv_dt_OAH(k,j,i) = 1.0 / (static_cast<Real>(fac_oa*SQR(len(i))*(eta_t(i)) /
                                      ((eta_t(i)*eta_t(i) + etaH*etaH)
                                      + TINY_NUMBER)));
          } // i loop
        } // end unsplit dt
      } // end if

      // Thermoelectric CFLS
      if(use_TE_para || use_TE_perp || use_TE_wedge) {
        for (int i=is; i<=ie; ++i) {
          e_int = pmb->phydro->w(IPR,k,j,i)/(gm1);
          kappa_J_para = thermKappa(ThermElecProcess::kappa_TE_para,k,j,i);
          kappa_J_perp  = thermKappa(ThermElecProcess::kappa_TE_perp,k,j,i);
          kappa_J_wedge = thermKappa(ThermElecProcess::kappa_TE_wedge,k,j,i);
          bool have_ion_frac = hdif_.hydro_diffusion_defined
                                  && hdif_.aniso_conduction
                                  && (hdif_.kappa_aniso_neutral > 0.0);
          HII = have_ion_frac ? hdif_.ion_frac(k,j,i) : 1.0;
          // Parabolic TE: thermal + ohmic + ambi + Nerst/Ettinghausen
          if(use_TE_wedge && STS_ENABLED  && hdif_.conduction_on && (eta_ohm > 0.0
                                                                    || eta_ad > 0.0)){
            if(hdif_.kappa_iso > 0.0)
              kappa0 = hdif_.kappa(HydroDiffusion::DiffProcess::iso,k,j,i);
            if(hdif_.aniso_conduction) {
              kappa_n = hdif_.kappa(HydroDiffusion::DiffProcess::kappa_neutral,k,j,i);
              kappa_para = hdif_.kappa(HydroDiffusion::DiffProcess::kappa_para,k,j,i);
              kappa_perp =  hdif_.kappa(HydroDiffusion::DiffProcess::kappa_perp,k,j,i);
              kappa0 = (1.0-HII)*kappa_n + HII*kappa_perp;
              d_kappa = HII*(kappa_para - kappa_perp);
            }
            Real kappa_Jw = HII*(kappa_J_wedge);
            lam_TE_para = TEParaEigenMax(gm1, kappa0, d_kappa, eta_t(i),
                                              kappa_Jw, e_int);
            dt_TE_para = std::min(dt_TE_para, static_cast<Real>
                                  (fac_oa*SQR(len(i))/(lam_TE_para + TINY_NUMBER)));
          }

          // Hyperbolic TE: Hall + Peltier/Seebeck + (wedge thermal in new block dt)
          if((use_TE_para || use_TE_perp) && STS_ENABLED) {
            Real etaH = (eta_hall > 0.0) ? std::abs(etaB(DiffProcess::hall,k,j,i)) : 0.0;
            lam_TE_hyp = TEHypEigenMax(etaH, HII, kappa_J_para, kappa_J_perp, e_int);
            inv_dt_TE_hyp(k,j,i) =  1.0/(static_cast<Real>(fac_h*SQR(len(i))/
                                        (lam_TE_hyp+ TINY_NUMBER)));
            }

          // Thermoelectric CFL unsplit
          if (!STS_ENABLED) { // extremely restrictive
          lam_TE = TEEigenMax(kappa_J_para, kappa_J_perp, kappa_J_wedge, HII, e_int);
          inv_dt_TE(k,j,i) = 1.0 / (fac_h*SQR(len(i)) / (lam_TE + TINY_NUMBER));
          }
        } // i loop
      } // end TE dt
    } // j loop
  } // k loop
  return;
}

//----------------------------------------------------------------------------------------
//! \fn Real TEEigenMax(...)
//! \brief Unsplit/isolated thermoelectric CFL. FT_w is ill-posed without therm+mag diff
Real FieldDiffusion::TEEigenMax(const Real kJpara, const Real kJperp, const Real kJwedge,
                                const Real HII, const Real E_int) {

  Real dk = HII*std::abs(kJpara - kJperp);
  Real kw = HII*kJwedge;
  Real lam1 = std::abs(kw)/std::sqrt(E_int);

  Real lam2 = 0.0;
  Real u_star = 0.0;
  if(dk != 0) {
    u_star = (dk*dk + kw*kw)/(2.0*dk*dk);
    if(u_star >= 0.0 && u_star <= 1.0){
      lam2 = std::sqrt((dk*dk - kw*kw)*(dk*dk - kw*kw)/(4.0*E_int*dk*dk));
    }
  }
  Real lam3 = 0.0;
  if(kw == 0.0 || u_star < 0.0 || u_star > 1.0){
    lam3 = 0.5*dk/std::sqrt(E_int);
  }

  Real lam = std::max({lam1, lam2, lam3});

  return lam;
}

//----------------------------------------------------------------------------------------
//! \fn Real TEParaEigenMax(...)
//! \brief Max eigenvalue of the PARABOLIC thermal+ohmic+ambi+Nerst/Ettinghausen system
Real FieldDiffusion::TEParaEigenMax(const Real gm1, const Real kappa_0,
                                    const Real d_kappa, const Real eta_oa,
                                    const Real kappa_Jw, const Real E_int) {

  Real lam0 = 0.5*(gm1*kappa_0 + eta_oa + std::sqrt(SQR(gm1*kappa_0 - eta_oa)
                   + 4.0*SQR(kappa_Jw)/E_int));
  Real lam1 = std::max(gm1*(kappa_0 + d_kappa), eta_oa);
  return std::max(lam0, lam1);
}

//----------------------------------------------------------------------------------------
//! \fn Real TEHypeEigenMax(...)
//! \brief Max |eigenvalue| of the HYPERBOLIC Hall + Peltier/Seebeck system.
Real FieldDiffusion::TEHypEigenMax(const Real etaH, const Real HII, const Real kJpara,
                                   const Real kJperp, const Real E_int) {

  const Real dk = HII*std::abs(kJpara - kJperp);
  Real lam = 0.0;
  if (dk == 0.0) {
    lam = std::abs(etaH);
    return lam;
  }
  const Real u_star = SQR(etaH)*E_int/(2.0*SQR(dk)) + 0.5;
  if (u_star > 1.0 || u_star < 0.0) {
    lam = std::abs(etaH);
    return lam;
  }
  lam = std::sqrt(0.5*SQR(etaH) + 0.25*(SQR(dk)/E_int + SQR(SQR(etaH))*E_int/SQR(dk)));
  return lam;
}
