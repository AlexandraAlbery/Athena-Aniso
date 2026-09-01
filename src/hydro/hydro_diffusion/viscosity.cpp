//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file viscosity.cpp
//! \brief functions to calculate viscous stresses

// C headers

// C++ headers

// Athena++ headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../../coordinates/coordinates.hpp"
#include "../../eos/eos.hpp"
#include "../../mesh/mesh.hpp"
#include "../hydro.hpp"
#include "hydro_diffusion.hpp"
#ifdef MAGNETIC_FIELDS_ENABLED
#include "../../field/field.hpp"              // Field
#endif

//----------------------------------------------------------------------------------------
//! \fn void HydroDiffusion::ViscousFluxBulk
//! \brief Calculate bulk viscous stress as fluxes
void HydroDiffusion::ViscousFluxBulk(const AthenaArray<Real> &p,
                     const AthenaArray<Real> &p_i, AthenaArray<Real> *flx) {
  const bool f2 = pmb_->pmy_mesh->f2;
  const bool f3 = pmb_->pmy_mesh->f3;
  AthenaArray<Real> &x1flux = flx[X1DIR];
  AthenaArray<Real> &x2flux = flx[X2DIR];
  AthenaArray<Real> &x3flux = flx[X3DIR];
  int il, iu, jl, ju, kl, ku;
  int is = pmb_->is; int js = pmb_->js; int ks = pmb_->ks;
  int ie = pmb_->ie; int je = pmb_->je; int ke = pmb_->ke;
  Real zeta, flx1, flx2, flx3;

  DivVelocity(p_i, div_vel_);

  // Calculate the flux across each face.
  // i-direction
  jl = js, ju = je, kl = ks, ku = ke;
  if (MAGNETIC_FIELDS_ENABLED) {
    if (f2) {
      if (!f3) // 2D MHD limits
        jl = js-1, ju = je+1, kl = ks, ku = ke;
      else // 3D MHD limits
        jl = js-1, ju = je+1, kl = ks-1, ku = ke+1;
    }
  }
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma omp simd private(zeta, flx1)
      for (int i=is; i<=ie+1; ++i) {
        zeta  = 0.5*(nu(DiffProcess::zeta,k,j,i)   + nu(DiffProcess::zeta,k,j,i-1));
        flx1 = -zeta*0.5*(div_vel_(k,j,i-1)+div_vel_(k,j,i));
        x1flux(IM1,k,j,i) += flx1;
        if (NON_BAROTROPIC_EOS)
          x1flux(IEN,k,j,i) += 0.5*(p(IM1,k,j,i-1) + p(IM1,k,j,i))*flx1;
      }
    }
  }

  // j-direction
  il = is, iu = ie, kl = ks, ku = ke;
  if (MAGNETIC_FIELDS_ENABLED) {
    if (!f3) // 2D MHD limits
      il = is-1, iu = ie+1, kl = ks, ku = ke;
    else // 3D MHD limits
      il = is-1, iu = ie+1, kl = ks-1, ku = ke+1;
  }
  if (f2) { // modify x2flux for 2D or 3D
    for (int k=kl; k<=ku; ++k) {
      for (int j=js; j<=je+1; ++j) {
#pragma omp simd private(zeta, flx2)
        for (int i=il; i<=iu; i++) {
          zeta  = 0.5*(nu(DiffProcess::zeta,k,j,i)    + nu(DiffProcess::zeta,k,j-1,i));
          flx2 = -zeta*0.5*(div_vel_(k,j-1,i)+div_vel_(k,j,i));
          x2flux(IM2,k,j,i) += flx2;
          if (NON_BAROTROPIC_EOS)
            x2flux(IEN,k,j,i) += 0.5*(p(IM2,k,j,i) + p(IM2,k,j-1,i))*flx2;
        }
      }
    }
  } else { // modify x2flux for 1D
    // store fluxes
#pragma omp simd private(zeta, flx2)
    for (int i=il; i<=iu; i++) {
      zeta  = nu(DiffProcess::zeta,ks,js,i);
      flx2 = -zeta*div_vel_(ks,js,i);
      x2flux(IM2,ks,js,i) += flx2;
      if (NON_BAROTROPIC_EOS)
        x2flux(IEN,ks,js,i) += p(IM2,ks,js,i)*flx2;
    }
#pragma omp simd
    for (int i=il; i<=iu; i++) {
      x2flux(IM1,ks,je+1,i) = x2flux(IM1,ks,js,i);
      x2flux(IM2,ks,je+1,i) = x2flux(IM2,ks,js,i);
      x2flux(IM3,ks,je+1,i) = x2flux(IM3,ks,js,i);
      if (NON_BAROTROPIC_EOS)
        x2flux(IEN,ks,je+1,i) = x2flux(IEN,ks,js,i);
    }
  }
  // k-direction
  // set the loop limits
  il = is, iu = ie, jl = js, ju = je;
  if (MAGNETIC_FIELDS_ENABLED) {
    if (f2) // 2D or 3D MHD limits
      il = is-1, iu = ie+1, jl = js-1, ju = je+1;
    else // 1D MHD limits
      il = is-1, iu = ie+1;
  }
  if (pmb_->block_size.nx3 > 1) { // modify x3flux for 3D
    for (int k=ks; k<=ke+1; ++k) {
      for (int j=jl; j<=ju; ++j) {
        // store fluxes
#pragma omp simd private(zeta, flx3)
        for (int i=il; i<=iu; i++) {
          zeta  = 0.5*(nu(DiffProcess::zeta,k,j,i)     + nu(DiffProcess::zeta,k-1,j,i));
          flx3 = -zeta*0.5*((div_vel_(k-1,j,i) + div_vel_(k,j,i)));
          x3flux(IM3,k,j,i) += flx3;
          if (NON_BAROTROPIC_EOS)
            x3flux(IEN,k,j,i) += 0.5*((p(IM3,k,j,i) + p(IM3,k-1,j,i))*flx3);
        }
      }
    }
  } else { // modify x3flux for 1D or 2D
    for (int j=jl; j<=ju; ++j) {
      // compute fluxes
      // store fluxes
#pragma omp simd private(zeta, flx3)
      for (int i=il; i<=iu; i++) {
        zeta = nu(DiffProcess::zeta,ks,j,i);
        flx3 = -zeta*div_vel_(ks,j,i);
        x3flux(IM3,ks,j,i) += flx3;
        x3flux(IM3,ke+1,j,i) = x3flux(IM3,ks,j,i);
        if (NON_BAROTROPIC_EOS) {
          x3flux(IEN,ks,j,i) += p(IM3,ks,j,i)*flx3;
          x3flux(IEN,ke+1,j,i) = x3flux(IEN,ks,j,i);
        }
      }
    }
  }
return;
}

//----------------------------------------------------------------------------------------
//! \fn void HydroDiffusion::ViscousFluxIso
//! \brief Calculate isotropic viscous stress as fluxes

void HydroDiffusion::ViscousFluxIso(const AthenaArray<Real> &p,
                                    const AthenaArray<Real> &p_i,
                                    AthenaArray<Real> *flx) {
  const bool f2 = pmb_->pmy_mesh->f2;
  const bool f3 = pmb_->pmy_mesh->f3;
  AthenaArray<Real> &x1flux = flx[X1DIR];
  AthenaArray<Real> &x2flux = flx[X2DIR];
  AthenaArray<Real> &x3flux = flx[X3DIR];
  int il, iu, jl, ju, kl, ku;
  int is = pmb_->is; int js = pmb_->js; int ks = pmb_->ks;
  int ie = pmb_->ie; int je = pmb_->je; int ke = pmb_->ke;
  Real nu1, denf, flx1, flx2, flx3;
  Real nuiso2 = - TWO_3RD;

  DivVelocity(p_i, div_vel_);

  // Calculate the flux across each face.
  // i-direction
  jl = js, ju = je, kl = ks, ku = ke;
  if (MAGNETIC_FIELDS_ENABLED) {
    if (f2) {
      if (!f3) // 2D MHD limits
        jl = js-1, ju = je+1, kl = ks, ku = ke;
      else // 3D MHD limits
        jl = js-1, ju = je+1, kl = ks-1, ku = ke+1;
    }
  }
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      FaceXdx(k, j, is, ie+1, p_i, fx_);
      FaceXdy(k, j, is, ie+1, p_i, fy_);
      FaceXdz(k, j, is, ie+1, p_i, fz_);
#pragma omp simd private(nu1, denf, flx1, flx2, flx3)
      for (int i=is; i<=ie+1; ++i) {
        nu1  = 0.5*(nu(DiffProcess::iso,k,j,i)   + nu(DiffProcess::iso,k,j,i-1));
        denf = 0.5*(p_i(IDN,k,j,i) + p_i(IDN,k,j,i-1));
        flx1 = -denf*nu1*(fx_(i) + nuiso2*0.5*(div_vel_(k,j,i) + div_vel_(k,j,i-1)));
        flx2 = -denf*nu1*fy_(i);
        flx3 = -denf*nu1*fz_(i);
        x1flux(IM1,k,j,i) += flx1;
        x1flux(IM2,k,j,i) += flx2;
        x1flux(IM3,k,j,i) += flx3;
        if (NON_BAROTROPIC_EOS)
          x1flux(IEN,k,j,i) += 0.5*((p(IM1,k,j,i-1) + p(IM1,k,j,i))*flx1 +
                                    (p(IM2,k,j,i-1) + p(IM2,k,j,i))*flx2 +
                                    (p(IM3,k,j,i-1) + p(IM3,k,j,i))*flx3);
      }
    }
  }

  // j-direction
  il = is, iu = ie, kl = ks, ku = ke;
  if (MAGNETIC_FIELDS_ENABLED) {
    if (!f3) // 2D MHD limits
      il = is-1, iu = ie+1, kl = ks, ku = ke;
    else // 3D MHD limits
      il = is-1, iu = ie+1, kl = ks-1, ku = ke+1;
  }
  if (f2) { // modify x2flux for 2D or 3D
    for (int k=kl; k<=ku; ++k) {
      for (int j=js; j<=je+1; ++j) {
        // compute fluxes
        FaceYdx(k, j, is, ie, p_i, fx_);
        FaceYdy(k, j, is, ie, p_i, fy_);
        FaceYdz(k, j, is, ie, p_i, fz_);
        // store fluxes
#pragma omp simd private(nu1, denf, flx1, flx2, flx3)
        for (int i=il; i<=iu; i++) {
          nu1  = 0.5*(nu(DiffProcess::iso,k,j,i)    + nu(DiffProcess::iso,k,j-1,i));
          denf = 0.5*(p_i(IDN,k,j-1,i)+ p_i(IDN,k,j,i));
          flx1 = -denf*nu1*fx_(i);
          flx2 = -denf*nu1*(fy_(i) + nuiso2*0.5*(div_vel_(k,j-1,i) + div_vel_(k,j,i)));
          flx3 = -denf*nu1*fz_(i);
          x2flux(IM1,k,j,i) += flx1;
          x2flux(IM2,k,j,i) += flx2;
          x2flux(IM3,k,j,i) += flx3;
          if (NON_BAROTROPIC_EOS)
            x2flux(IEN,k,j,i) += 0.5*((p(IM1,k,j,i) + p(IM1,k,j-1,i))*flx1 +
                                      (p(IM2,k,j,i) + p(IM2,k,j-1,i))*flx2 +
                                      (p(IM3,k,j,i) + p(IM3,k,j-1,i))*flx3);
        }
      }
    }
  } else { // modify x2flux for 1D
    // compute fluxes
    FaceYdx(ks, js, is, ie, p_i, fx_);
    FaceYdy(ks, js, is, ie, p_i, fy_);
    FaceYdz(ks, js, is, ie, p_i, fz_);
    // store fluxes
#pragma omp simd private(nu1, denf, flx1, flx2, flx3)
    for (int i=il; i<=iu; i++) {
      nu1  = nu(DiffProcess::iso,ks,js,i);
      denf = p_i(IDN,ks,js,i);
      flx1 = -denf*nu1*fx_(i);
      flx2 = -denf*nu1*(fy_(i) + nuiso2*div_vel_(ks,js,i));
      flx3 = -denf*nu1*fz_(i);
      x2flux(IM1,ks,js,i) += flx1;
      x2flux(IM2,ks,js,i) += flx2;
      x2flux(IM3,ks,js,i) += flx3;
      if (NON_BAROTROPIC_EOS)
        x2flux(IEN,ks,js,i) += p(IM1,ks,js,i)*flx1 +
                               p(IM2,ks,js,i)*flx2 +
                               p(IM3,ks,js,i)*flx3;
    }
#pragma omp simd
    for (int i=il; i<=iu; i++) {
      x2flux(IM1,ks,je+1,i) = x2flux(IM1,ks,js,i);
      x2flux(IM2,ks,je+1,i) = x2flux(IM2,ks,js,i);
      x2flux(IM3,ks,je+1,i) = x2flux(IM3,ks,js,i);
      if (NON_BAROTROPIC_EOS)
        x2flux(IEN,ks,je+1,i) = x2flux(IEN,ks,js,i);
    }
  }
  // k-direction
  // set the loop limits
  il = is, iu = ie, jl = js, ju = je;
  if (MAGNETIC_FIELDS_ENABLED) {
    if (f2) // 2D or 3D MHD limits
      il = is-1, iu = ie+1, jl = js-1, ju = je+1;
    else // 1D MHD limits
      il = is-1, iu = ie+1;
  }
  if (pmb_->block_size.nx3 > 1) { // modify x3flux for 3D
    for (int k=ks; k<=ke+1; ++k) {
      for (int j=jl; j<=ju; ++j) {
        // compute fluxes
        FaceZdx(k, j, is, ie, p_i, fx_);
        FaceZdy(k, j, is, ie, p_i, fy_);
        FaceZdz(k, j, is, ie, p_i, fz_);
        // store fluxes
#pragma omp simd private(nu1, denf, flx1, flx2, flx3)
        for (int i=il; i<=iu; i++) {
          nu1  = 0.5*(nu(DiffProcess::iso,k,j,i)     + nu(DiffProcess::iso,k-1,j,i));
          denf = 0.5*(p_i(IDN,k-1,j,i) + p_i(IDN,k,j,i));
          flx1 = -denf*nu1*fx_(i);
          flx2 = -denf*nu1*fy_(i);
          flx3 = -denf*nu1*(fz_(i) + nuiso2*0.5*(div_vel_(k-1,j,i) + div_vel_(k,j,i)));
          x3flux(IM1,k,j,i) += flx1;
          x3flux(IM2,k,j,i) += flx2;
          x3flux(IM3,k,j,i) += flx3;
          if (NON_BAROTROPIC_EOS)
            x3flux(IEN,k,j,i) += 0.5*((p(IM1,k,j,i) + p(IM1,k-1,j,i))*flx1 +
                                      (p(IM2,k,j,i) + p(IM2,k-1,j,i))*flx2 +
                                      (p(IM3,k,j,i) + p(IM3,k-1,j,i))*flx3);
        }
      }
    }
  } else { // modify x2flux for 1D or 2D
    for (int j=jl; j<=ju; ++j) {
      // compute fluxes
      FaceZdx(ks, j, is, ie, p_i, fx_);
      FaceZdy(ks, j, is, ie, p_i, fy_);
      FaceZdz(ks, j, is, ie, p_i, fz_);
      // store fluxes
#pragma omp simd private(nu1, denf, flx1, flx2, flx3)
      for (int i=il; i<=iu; i++) {
        nu1 = nu(DiffProcess::iso,ks,j,i);
        denf = p_i(IDN,ks,j,i);
        flx1 = -denf*nu1*fx_(i);
        flx2 = -denf*nu1*fy_(i);
        flx3 = -denf*nu1*(fz_(i) + nuiso2*div_vel_(ks,j,i));
        x3flux(IM1,ks,j,i) += flx1;
        x3flux(IM2,ks,j,i) += flx2;
        x3flux(IM3,ks,j,i) += flx3;
        x3flux(IM1,ke+1,j,i) = x3flux(IM1,ks,j,i);
        x3flux(IM2,ke+1,j,i) = x3flux(IM2,ks,j,i);
        x3flux(IM3,ke+1,j,i) = x3flux(IM3,ks,j,i);
        if (NON_BAROTROPIC_EOS) {
          x3flux(IEN,ks,j,i) += p(IM1,ks,j,i)*flx1 +
                                p(IM2,ks,j,i)*flx2 +
                                p(IM3,ks,j,i)*flx3;
          x3flux(IEN,ke+1,j,i) = x3flux(IEN,ks,j,i);
        }
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void HydroDiffusion::ViscousFluxAniso
//! \brief Calculate anisotropic viscous stress as fluxes
//  BS1: Braginskii, S., “Transport Processes in a Plasma”, Reviews of Plasma Physics,
//  Jan. 1965, vol. 1, pp. 205,
//  https://static.ias.edu/pitp/2016/sites/pitp/files/braginskii_1965-1.pdf.
//  RF1: https://farside.ph.utexas.edu/teaching/plasma/Plasma/node53.html
//  Solves BS1 eqn(4.41) & eqn(4.42)
//  RF1 eqn (4.128) - eqn(4.133)

  void HydroDiffusion::ViscousFluxAniso(const AthenaArray<Real> &p,
                                        const AthenaArray<Real> &p_i,
                                        AthenaArray<Real> *flx,
                                        AthenaArray<Real> *flx_disp) {
  const bool f2 = pmb_->pmy_mesh->f2;
  const bool f3 = pmb_->pmy_mesh->f3;
  AthenaArray<Real> &x1flux = flx[X1DIR];
  AthenaArray<Real> &x2flux = flx[X2DIR];
  AthenaArray<Real> &x3flux = flx[X3DIR];
  AthenaArray<Real> &x1flux_disp = flx_disp[X1DIR];
  AthenaArray<Real> &x2flux_disp = flx_disp[X2DIR];
  AthenaArray<Real> &x3flux_disp = flx_disp[X3DIR];
  int is = pmb_->is; int js = pmb_->js; int ks = pmb_->ks;
  int ie = pmb_->ie; int je = pmb_->je; int ke = pmb_->ke;
  int il=is-1, iu=ie+1, jl=js-1, ju=je+1, kl=ks-1, ku=ke+1;

  DivVelocity(p_i, div_vel_);

  const bool use_neutral = (HydroDiffusion::nu_aniso_neutral > 0.0) && calc_para_;
  const bool use_para    = (HydroDiffusion::nu_aniso_para    > 0.0) && calc_para_;
  const bool use_perp1   = (HydroDiffusion::nu_aniso_perp1   > 0.0) && calc_para_;
  const bool use_perp2   = (HydroDiffusion::nu_aniso_perp2   > 0.0) && calc_para_;
  const bool use_wedge1  = (HydroDiffusion::nu_aniso_wedge1  > 0.0) && calc_disp_;
  const bool use_wedge2  = (HydroDiffusion::nu_aniso_wedge2  > 0.0) && calc_disp_;

  const bool use_B   = use_para || use_perp1 || use_perp2 || use_wedge1 || use_wedge2;
  const bool use_I   = use_perp1 || use_perp2 || use_wedge1;
  const bool use_WI  = use_perp1 || use_wedge1;
  const bool use_Wbb = use_perp2 || use_wedge2;

  // Calculate the flux across each face.
  // i-direction
  jl = js, ju = je, kl = ks, ku = ke;
  if (MAGNETIC_FIELDS_ENABLED) {
    if (f2) {
      if (!f3) // 2D MHD limits
        jl = js-1, ju = je+1, kl = ks, ku = ke;
      else // 3D MHD limits
        jl = js-1, ju = je+1, kl = ks-1, ku = ke+1;
    }
  }
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
    #pragma omp simd
      for (int i=is; i<=ie+1; ++i) {
        Real nu_iso = 0.0, nu_para = 0.0, nu_perp = 0.0, nu_wedge = 0.0;
        Real rho_face = 0.0;
        Real B = 0.0, Bx = 0.0, By = 0.0, Bz = 0.0;
        Real bx = 0.0, by = 0.0, bz = 0.0;
        Real bxx = 0.0, bxy = 0.0, bxz = 0.0, byy = 0.0, byz = 0.0, bzz = 0.0;
        Real Ixx = 0.0, Ixy = 0.0, Ixz = 0.0, Iyy = 0.0, Iyz = 0.0, Izz = 0.0;
        Real WIxx = 0.0, WIxy = 0.0, WIxz = 0.0,
             WIyx = 0.0, WIyy = 0.0, WIyz = 0.0,
             WIzx = 0.0, WIzy = 0.0, WIzz = 0.0;
        Real Wbxx = 0.0, Wbxy = 0.0, Wbxz = 0.0,
             Wbyx = 0.0, Wbyy = 0.0, Wbyz = 0.0,
             Wbzx = 0.0, Wbzy = 0.0, Wbzz = 0.0;
        Real bWb = 0.0;
        Real Pi_xx = 0.0, Pi_xy = 0.0, Pi_xz = 0.0,
             Pi_yx = 0.0, Pi_yy = 0.0, Pi_yz = 0.0,
             Pi_zx = 0.0, Pi_zy = 0.0, Pi_zz = 0.0;
        Real Pi_xx_disp = 0.0, Pi_xy_disp = 0.0, Pi_xz_disp = 0.0,
        Pi_yx_disp = 0.0, Pi_yy_disp = 0.0, Pi_yz_disp = 0.0,
        Pi_zx_disp = 0.0, Pi_zy_disp = 0.0, Pi_zz_disp = 0.0;

        rho_face = 0.5*(p_i(IDN,k,j,i) + p_i(IDN,k,j,i-1));

        Real dvx_dx = (p_i(IM1,k,j,i) - p_i(IM1,k,j,i-1)) / (pco_->dx1v(i-1));

        Real dvx_dy = 0.5 * ( (p_i(IM1,k,j+1,i) + p_i(IM1,k,j+1,i-1))
                            - (p_i(IM1,k,j-1,i) + p_i(IM1,k,j-1,i-1)) )
                      / pco_->h2f(i)
                      / (pco_->dx2v(j-1) + pco_->dx2v(j));

        Real dvx_dz = 0.5 * ( (p_i(IM1,k+1,j,i) + p_i(IM1,k+1,j,i-1))
                            - (p_i(IM1,k-1,j,i) + p_i(IM1,k-1,j,i-1)) )
                      / (pco_->h31f(i) * pco_->h32v(j))
                      / (pco_->dx3v(k-1) + pco_->dx3v(k));

        Real dvy_dx = pco_->h2f(i)
                      * ( p_i(IM2,k,j,i)   / pco_->h2v(i)
                        - p_i(IM2,k,j,i-1) / pco_->h2v(i-1) )
                      / pco_->dx1v(i-1);

        Real dvy_dy = 0.5 * ( (p_i(IM2,k,j+1,i) + p_i(IM2,k,j+1,i-1))
                            - (p_i(IM2,k,j-1,i) + p_i(IM2,k,j-1,i-1)) )
                      / pco_->h2f(i)
                      / (pco_->dx2v(j-1) + pco_->dx2v(j));

        Real dvy_dz = 0.5 * ( (p_i(IM2,k+1,j,i) + p_i(IM2,k+1,j,i-1))
                            - (p_i(IM2,k-1,j,i) + p_i(IM2,k-1,j,i-1)) )
                      / (pco_->h31f(i) * pco_->h32v(j))
                      / (pco_->dx3v(k-1) + pco_->dx3v(k));

        Real dvz_dx = pco_->h31f(i)
                      * ( p_i(IM3,k,j,i)   / pco_->h31v(i)
                        - p_i(IM3,k,j,i-1) / pco_->h31v(i-1) )
                      / pco_->dx1v(i-1);

        Real dvz_dy = 0.5 * ( (p_i(IM3,k,j+1,i) + p_i(IM3,k,j+1,i-1))
                            - (p_i(IM3,k,j-1,i) + p_i(IM3,k,j-1,i-1)) )
                      / pco_->h2f(i)
                      / (pco_->dx2v(j-1) + pco_->dx2v(j));

        Real dvz_dz = 0.5 * ( (p_i(IM3,k+1,j,i) + p_i(IM3,k+1,j,i-1))
                            - (p_i(IM3,k-1,j,i) + p_i(IM3,k-1,j,i-1)) )
                      / (pco_->h31f(i) * pco_->h32v(j))
                      / (pco_->dx3v(k-1) + pco_->dx3v(k));

        Real divv = 0.5 * (div_vel_(k,j,i) + div_vel_(k,j,i-1));

        Real Wxx_f = 2.0*(dvx_dx - divv/3.0);
        Real Wyy_f = 2.0*(dvy_dy - divv/3.0);
        Real Wzz_f = 2.0*(dvz_dz - divv/3.0);

        Real Wxy_f = (dvx_dy + dvy_dx);
        Real Wxz_f = (dvx_dz + dvz_dx);
        Real Wyz_f = (dvy_dz + dvz_dy);

        Real HII = (HydroDiffusion::nu_aniso_neutral>0.0) ?
             0.5*(HydroDiffusion::ion_frac(k,j,i)+HydroDiffusion::ion_frac(k,j,i-1))
             : 1.0;

        if (use_neutral) { // neutral part is isotropic
          nu_iso =  0.5*(nu(DiffProcess::nu_neutral,k,j,i)
                       + nu(DiffProcess::nu_neutral,k,j,i-1));

          Pi_xx += -(1-HII)*rho_face*nu_iso*Wxx_f;
          Pi_xy += -(1-HII)*rho_face*nu_iso*Wxy_f;
          Pi_xz += -(1-HII)*rho_face*nu_iso*Wxz_f;
          }

        // build anisotropic dependencies
        if (use_B) {
          Bx =  pmb_->pfield->b.x1f(k,j,i);
          By = 0.5*(pmb_->pfield->bcc(IB2, k,j,i) + pmb_->pfield->bcc(IB2, k,j,i-1));
          Bz = 0.5*(pmb_->pfield->bcc(IB3, k,j,i) + pmb_->pfield->bcc(IB3, k,j,i-1));
          B = std::sqrt(Bx*Bx + By*By + Bz*Bz);

          if (B > TINY_NUMBER) {
            bx = Bx/B;
            by = By/B;
            bz = Bz/B;
          } else {
            bx=by=bz=0.0;
          }

          // bb
          bxx = bx*bx;
          bxy = bx*by;
          bxz = bx*bz;
          byy = by*by;
          byz = by*bz;
          bzz = bz*bz;

          // b dot W dot b
          Real wbx = Wxx_f*bx + Wxy_f*by + Wxz_f*bz;
          Real wby = Wxy_f*bx + Wyy_f*by + Wyz_f*bz;
          Real wbz = Wxz_f*bx + Wyz_f*by + Wzz_f*bz;
          bWb = bx*wbx + by*wby + bz*wbz;
        }
        if (use_I) {
          // I_perp tensor
          Ixx = 1-bxx;
          Ixy = -bxy;
          Ixz = -bxz;
          Iyy = 1-byy;
          Iyz = -byz;
          Izz = 1-bzz;

          //W dot I_perp
          WIxx = Wxx_f*Ixx + Wxy_f*Ixy + Wxz_f*Ixz;
          WIxy = Wxx_f*Ixy + Wxy_f*Iyy + Wxz_f*Iyz;
          WIxz = Wxx_f*Ixz + Wxy_f*Iyz + Wxz_f*Izz;

          WIyx = Wxy_f*Ixx + Wyy_f*Ixy + Wyz_f*Ixz;
          WIyy = Wxy_f*Ixy + Wyy_f*Iyy + Wyz_f*Iyz;
          WIyz = Wxy_f*Ixz + Wyy_f*Iyz + Wyz_f*Izz;

          WIzx = Wxz_f*Ixx + Wyz_f*Ixy + Wzz_f*Ixz;
          WIzy = Wxz_f*Ixy + Wyz_f*Iyy + Wzz_f*Iyz;
          WIzz = Wxz_f*Ixz + Wyz_f*Iyz + Wzz_f*Izz;
        }
        if (use_Wbb) {
          // W dot bb
          Wbxx = Wxx_f*bxx + Wxy_f*bxy + Wxz_f*bxz;
          Wbxy = Wxx_f*bxy + Wxy_f*byy + Wxz_f*byz;
          Wbxz = Wxx_f*bxz + Wxy_f*byz + Wxz_f*bzz;

          Wbyx = Wxy_f*bxx + Wyy_f*bxy + Wyz_f*bxz;
          Wbyy = Wxy_f*bxy + Wyy_f*byy + Wyz_f*byz;
          Wbyz = Wxy_f*bxz + Wyy_f*byz + Wyz_f*bzz;

          Wbzx = Wxz_f*bxx + Wyz_f*bxy + Wzz_f*bxz;
          Wbzy = Wxz_f*bxy + Wyz_f*byy + Wzz_f*byz;
          Wbzz = Wxz_f*bxz + Wyz_f*byz + Wzz_f*bzz;
        }
        if (use_para) { // parallel viscosity (Pi0) RF1 eqn (4.129)
        nu_para = 0.5*(nu(DiffProcess::nu_para,k,j,i) + nu(DiffProcess::nu_para,k,j,i-1));

        // Pi0
        Pi_xx += -1.5 * HII * rho_face * nu_para * bWb * (bxx - 1.0/3.0);
        Pi_xy += -1.5 * HII * rho_face * nu_para * bWb * bxy;
        Pi_xz += -1.5 * HII * rho_face * nu_para * bWb * bxz;
        }
        if (use_perp1) { // perpendicular viscosity (Pi1) RF1 eqn (4.130)
          nu_perp = 0.5*(nu(DiffProcess::nu_perp1,k,j,i)
                       + nu(DiffProcess::nu_perp1,k,j,i-1));;
          //W_Perp = I_Perp dot W_Perp dot I_Perp
          Real WPerpxx = Ixx*WIxx + Ixy*WIyx + Ixz*WIzx;
          Real WPerpxy = Ixx*WIxy + Ixy*WIyy + Ixz*WIzy;
          Real WPerpxz = Ixx*WIxz + Ixy*WIyz + Ixz*WIzz;

          // Pi1
          Pi_xx += -HII * rho_face * nu_perp * (WPerpxx + 0.5*(1.0-bxx)*bWb);
          Pi_xy += -HII * rho_face * nu_perp * (WPerpxy + 0.5*(-bxy)*bWb);
          Pi_xz += -HII * rho_face * nu_perp * (WPerpxz + 0.5*(-bxz)*bWb);
        }
        if (use_perp2) { // Perendicular viscosity (Pi2) RF1 eqn (4.131)
        nu_perp = 0.5*(nu(DiffProcess::nu_perp2,k,j,i)
                     + nu(DiffProcess::nu_perp2,k,j,i-1));

          //I_perp dot W dot bb
          Real IWbxx = Ixx*Wbxx + Ixy*Wbyx + Ixz*Wbzx;
          Real IWbxy = Ixx*Wbxy + Ixy*Wbyy + Ixz*Wbzy;
          Real IWbxz = Ixx*Wbxz + Ixy*Wbyz + Ixz*Wbzz;
          // bb dot W dot I_Perp
          Real bWIxx = bxx*WIxx + bxy*WIyx + bxz*WIzx;
          Real bWIxy = bxx*WIxy + bxy*WIyy + bxz*WIzy;
          Real bWIxz = bxx*WIxz + bxy*WIyz + bxz*WIzz;

          // Pi 2
          Pi_xx += -HII * rho_face * nu_perp * 4* (IWbxx + bWIxx);
          Pi_xy += -HII * rho_face * nu_perp * 4* (IWbxy + bWIxy);
          Pi_xz += -HII * rho_face * nu_perp * 4* (IWbxz + bWIxz);
        }
        if (use_wedge1) {  // Wedge viscosity (Pi3) RF1 eqn (4.132)
          nu_wedge =  0.5*(nu(DiffProcess::nu_wedge1,k,j,i)
                         + nu(DiffProcess::nu_wedge1,k,j,i-1));
          // I_perp dot W
          Real IWxx = Ixx*Wxx_f + Ixy*Wxy_f + Ixz*Wxz_f;
          Real IWxy = Ixx*Wxy_f + Ixy*Wyy_f + Ixz*Wyz_f;
          Real IWxz = Ixx*Wxz_f + Ixy*Wyz_f + Ixz*Wzz_f;

          // b cross W I_perp
          Real bcWIxx = by*WIzx - bz*WIyx;
          Real bcWIxy = by*WIzy - bz*WIyy;
          Real bcWIxz = by*WIzz - bz*WIyz;

          // I_Perp W cross B
          Real IWcbxx = IWxy*bz - IWxz*by;
          Real IWcbxy = IWxz*bx - IWxx*bz;
          Real IWcbxz = IWxx*by - IWxy*bx;

          // Pi 3
          Pi_xx_disp += HII*0.5*rho_face*nu_wedge*(bcWIxx - IWcbxx);
          Pi_xy_disp += HII*0.5*rho_face*nu_wedge*(bcWIxy - IWcbxy);
          Pi_xz_disp += HII*0.5*rho_face*nu_wedge*(bcWIxz - IWcbxz);
        }
        if (use_wedge2) { // Wedge viscosity (Pi4) RF1 eqn (4.133)
          nu_wedge =  0.5*(nu(DiffProcess::nu_wedge2,k,j,i)
                         + nu(DiffProcess::nu_wedge2,k,j,i-1));
          // bb dot W
          Real bWxx = bxx*Wxx_f + bxy*Wxy_f + bxz*Wxz_f;
          Real bWxy = bxx*Wxy_f + bxy*Wyy_f + bxz*Wyz_f;
          Real bWxz = bxx*Wxz_f + bxy*Wyz_f + bxz*Wzz_f;

          // b cross (W dot bb)
          Real bcWbxx = by*Wbzx - bz*Wbyx;
          Real bcWbxy = by*Wbzy - bz*Wbyy;
          Real bcWbxz = by*Wbzz - bz*Wbyz;

          // (bb dot W) cross b
          Real bWcbxx = bWxy*bz - bWxz*by;
          Real bWcbxy = bWxz*bx - bWxx*bz;
          Real bWcbxz = bWxx*by - bWxy*bx;

          // Pi 4
          Pi_xx_disp += HII*2.0*rho_face*nu_wedge*(bcWbxx - bWcbxx);
          Pi_xy_disp += HII*2.0*rho_face*nu_wedge*(bcWbxy -  bWcbxy);
          Pi_xz_disp += HII*2.0*rho_face*nu_wedge*(bcWbxz - bWcbxz);
        }
        if(use_neutral || use_para || use_perp1 || use_perp2) {
        x1flux(IM1,k,j,i) += Pi_xx;
        x1flux(IM2,k,j,i) += Pi_xy;
        x1flux(IM3,k,j,i) += Pi_xz;
          }
        if (use_wedge1 || use_wedge2) {
        x1flux_disp(IM1,k,j,i) += Pi_xx_disp;
        x1flux_disp(IM2,k,j,i) += Pi_xy_disp;
        x1flux_disp(IM3,k,j,i) += Pi_xz_disp;
          if(!STS_ENABLED) {
            x1flux(IM1,k,j,i) += Pi_xx_disp;
            x1flux(IM2,k,j,i) += Pi_xy_disp;
            x1flux(IM3,k,j,i) += Pi_xz_disp;
          }
        }
        if (NON_BAROTROPIC_EOS) {
          if(use_neutral || use_para || use_perp1 || use_perp2) {
            x1flux(IEN,k,j,i) += 0.5*((p(IM1,k,j,i-1) + p(IM1,k,j,i))*Pi_xx +
                                      (p(IM2,k,j,i-1) + p(IM2,k,j,i))*Pi_xy +
                                      (p(IM3,k,j,i-1) + p(IM3,k,j,i))*Pi_xz);
          }
          if (use_wedge1 || use_wedge2) {
            x1flux_disp(IEN,k,j,i) += 0.5*((p(IM1,k,j,i-1) + p(IM1,k,j,i))*Pi_xx_disp +
                                          (p(IM2,k,j,i-1) + p(IM2,k,j,i))*Pi_xy_disp +
                                          (p(IM3,k,j,i-1) + p(IM3,k,j,i))*Pi_xz_disp);
            if(!STS_ENABLED) {
              x1flux(IEN,k,j,i) += 0.5*((p(IM1,k,j,i-1) + p(IM1,k,j,i))*Pi_xx_disp +
                                        (p(IM2,k,j,i-1) + p(IM2,k,j,i))*Pi_xy_disp +
                                        (p(IM3,k,j,i-1) + p(IM3,k,j,i))*Pi_xz_disp);
            } // end if STS
          } // end if wedge
        } // end if NON_BAROTROPIC_EOS
      } // i loop
    } // j loop
  } // k loop

  // j-direction
  il = is, iu = ie, kl = ks, ku = ke;
  if (MAGNETIC_FIELDS_ENABLED) {
    if (!f3) // 2D MHD limits
      il = is-1, iu = ie+1, kl = ks, ku = ke;
    else // 3D MHD limits
      il = is-1, iu = ie+1, kl = ks-1, ku = ke+1;
  }
  if (f2) { // j direction
    for (int k=kl; k<=ku; ++k) {
      for (int j=js; j<=je+1; ++j) {
  #pragma omp simd
        for (int i=il; i<=iu; ++i) {
          Real nu_iso = 0.0, nu_para = 0.0, nu_perp = 0.0, nu_wedge = 0.0;
          Real rho_face = 0.0;
          Real B = 0.0, Bx = 0.0, By = 0.0, Bz = 0.0;
          Real bx = 0.0, by = 0.0, bz = 0.0;
          Real bxx = 0.0, bxy = 0.0, bxz = 0.0, byy = 0.0, byz = 0.0, bzz = 0.0;
          Real Ixx = 0.0, Ixy = 0.0, Ixz = 0.0, Iyy = 0.0, Iyz = 0.0, Izz = 0.0;
          Real WIxx = 0.0, WIxy = 0.0, WIxz = 0.0,
               WIyx = 0.0, WIyy = 0.0, WIyz = 0.0,
               WIzx = 0.0, WIzy = 0.0, WIzz = 0.0;
          Real Wbxx = 0.0, Wbxy = 0.0, Wbxz = 0.0,
               Wbyx = 0.0, Wbyy = 0.0, Wbyz = 0.0,
               Wbzx = 0.0, Wbzy = 0.0, Wbzz = 0.0;
          Real bWb = 0.0;
          Real Pi_xx = 0.0, Pi_xy = 0.0, Pi_xz = 0.0,
               Pi_yx = 0.0, Pi_yy = 0.0, Pi_yz = 0.0,
               Pi_zx = 0.0, Pi_zy = 0.0, Pi_zz = 0.0;
          Real Pi_xx_disp = 0.0, Pi_xy_disp = 0.0, Pi_xz_disp = 0.0,
               Pi_yx_disp = 0.0, Pi_yy_disp = 0.0, Pi_yz_disp = 0.0,
               Pi_zx_disp = 0.0, Pi_zy_disp = 0.0, Pi_zz_disp = 0.0;
          rho_face = 0.5*(p_i(IDN,k,j,i) + p_i(IDN,k,j-1,i));

          Real dvx_dx =
          ( 0.5*(p_i(IM1,k,j,  i+1) + p_i(IM1,k,j-1,i+1))
          - 0.5*(p_i(IM1,k,j,  i-1) + p_i(IM1,k,j-1,i-1)) )
          / (pco_->dx1v(i-1) + pco_->dx1v(i));

        Real dvy_dx =
          pco_->h2v(i)
          * ( 0.5*(p_i(IM2,k,j,  i+1) + p_i(IM2,k,j-1,i+1)) / pco_->h2v(i+1)
            - 0.5*(p_i(IM2,k,j,  i-1) + p_i(IM2,k,j-1,i-1)) / pco_->h2v(i-1) )
          / (pco_->dx1v(i-1) + pco_->dx1v(i));

        Real dvz_dx =
          pco_->h31v(i)
          * ( 0.5*(p_i(IM3,k,j,  i+1) + p_i(IM3,k,j-1,i+1)) / pco_->h31v(i+1)
            - 0.5*(p_i(IM3,k,j,  i-1) + p_i(IM3,k,j-1,i-1)) / pco_->h31v(i-1) )
          / (pco_->dx1v(i-1) + pco_->dx1v(i));

        Real dvx_dy =
          (p_i(IM1,k,j,i) - p_i(IM1,k,j-1,i))
          / (pco_->h2v(i) * pco_->dx2v(j-1));

          Real dvy_dy =
          (p_i(IM2,k,j,i) - p_i(IM2,k,j-1,i))
          / (pco_->h2v(i) * pco_->dx2v(j-1))
          + 0.5*(p_i(IM1,k,j,i) + p_i(IM1,k,j-1,i)) / pco_->h2v(i) * pco_->dh2vd1(i);

        Real dvz_dy =
          pco_->h32f(j)
          * ( p_i(IM3,k,j,  i) / pco_->h32v(j)
            - p_i(IM3,k,j-1,i) / pco_->h32v(j-1) )
          / (pco_->h2v(i) * pco_->dx2v(j-1));

        Real dvx_dz =
          0.5*( (p_i(IM1,k+1,j,  i) + p_i(IM1,k+1,j-1,i))
              - (p_i(IM1,k-1,j,  i) + p_i(IM1,k-1,j-1,i)) )
          / (pco_->h31v(i) * pco_->h32f(j))
          / (pco_->dx3v(k-1) + pco_->dx3v(k));

        Real dvy_dz =
          0.5*( (p_i(IM2,k+1,j,  i) + p_i(IM2,k+1,j-1,i))
              - (p_i(IM2,k-1,j,  i) + p_i(IM2,k-1,j-1,i)) )
          / (pco_->h31v(i) * pco_->h32f(j))
          / (pco_->dx3v(k-1) + pco_->dx3v(k));

        Real dvz_dz =
          0.5*( (p_i(IM3,k+1,j,  i) + p_i(IM3,k+1,j-1,i))
              - (p_i(IM3,k-1,j,  i) + p_i(IM3,k-1,j-1,i)) )
          / (pco_->h31v(i) * pco_->h32f(j))
          / (pco_->dx3v(k-1) + pco_->dx3v(k));

        Real divv = 0.5 * (div_vel_(k,j,i) + div_vel_(k,j-1,i));

        Real Wxx_f = 2.0*(dvx_dx - divv/3.0);
        Real Wyy_f = 2.0*(dvy_dy - divv/3.0);
        Real Wzz_f = 2.0*(dvz_dz - divv/3.0);

        Real Wxy_f = (dvx_dy + dvy_dx);
        Real Wxz_f = (dvx_dz + dvz_dx);
        Real Wyz_f = (dvy_dz + dvz_dy);

        Real HII = (HydroDiffusion::nu_aniso_neutral>0.0) ?
                   0.5*(HydroDiffusion::ion_frac(k,j,i)
                      + HydroDiffusion::ion_frac(k,j-1,i))
                    : 1.0;

        if (use_neutral) { // neutral part is isotropic
          nu_iso =  0.5*(nu(DiffProcess::nu_neutral,k,j,i)
                        + nu(DiffProcess::nu_neutral,k,j-1,i));

          Pi_yx += -(1-HII)*rho_face*nu_iso*Wxy_f;
          Pi_yy += -(1-HII)*rho_face*nu_iso*Wyy_f;
          Pi_yz += -(1-HII)*rho_face*nu_iso*Wyz_f;
          }

        // build anisotropic dependencies
        if (use_B) {
          Bx = 0.5*(pmb_->pfield->bcc(IB1, k,j,i) + pmb_->pfield->bcc(IB1, k,j-1,i));
          By =  pmb_->pfield->b.x2f(k,j,i);
          Bz = 0.5*(pmb_->pfield->bcc(IB3, k,j,i) + pmb_->pfield->bcc(IB3, k,j-1,i));
          B = std::sqrt(Bx*Bx + By*By + Bz*Bz);

          if (B > TINY_NUMBER) {
            bx = Bx/B;
            by = By/B;
            bz = Bz/B;
          } else {
            bx=by=bz=0.0;
          }

          // bb
          bxx = bx*bx;
          bxy = bx*by;
          bxz = bx*bz;
          byy = by*by;
          byz = by*bz;
          bzz = bz*bz;

          // b dot W dot b
          Real wbx = Wxx_f*bx + Wxy_f*by + Wxz_f*bz;
          Real wby = Wxy_f*bx + Wyy_f*by + Wyz_f*bz;
          Real wbz = Wxz_f*bx + Wyz_f*by + Wzz_f*bz;
          bWb = bx*wbx + by*wby + bz*wbz;
        }
        if (use_I) { // I_perp tensor
          Ixx = 1-bxx;
          Ixy = -bxy;
          Ixz = -bxz;
          Iyy = 1-byy;
          Iyz = -byz;
          Izz = 1-bzz;

          // W dot I_perp
          WIxx = Wxx_f*Ixx + Wxy_f*Ixy + Wxz_f*Ixz;
          WIxy = Wxx_f*Ixy + Wxy_f*Iyy + Wxz_f*Iyz;
          WIxz = Wxx_f*Ixz + Wxy_f*Iyz + Wxz_f*Izz;

          WIyx = Wxy_f*Ixx + Wyy_f*Ixy + Wyz_f*Ixz;
          WIyy = Wxy_f*Ixy + Wyy_f*Iyy + Wyz_f*Iyz;
          WIyz = Wxy_f*Ixz + Wyy_f*Iyz + Wyz_f*Izz;

          WIzx = Wxz_f*Ixx + Wyz_f*Ixy + Wzz_f*Ixz;
          WIzy = Wxz_f*Ixy + Wyz_f*Iyy + Wzz_f*Iyz;
          WIzz = Wxz_f*Ixz + Wyz_f*Iyz + Wzz_f*Izz;
        }

        if (use_Wbb) { // W dot bb
          Wbxx = Wxx_f*bxx + Wxy_f*bxy + Wxz_f*bxz;
          Wbxy = Wxx_f*bxy + Wxy_f*byy + Wxz_f*byz;
          Wbxz = Wxx_f*bxz + Wxy_f*byz + Wxz_f*bzz;

          Wbyx = Wxy_f*bxx + Wyy_f*bxy + Wyz_f*bxz;
          Wbyy = Wxy_f*bxy + Wyy_f*byy + Wyz_f*byz;
          Wbyz = Wxy_f*bxz + Wyy_f*byz + Wyz_f*bzz;

          Wbzx = Wxz_f*bxx + Wyz_f*bxy + Wzz_f*bxz;
          Wbzy = Wxz_f*bxy + Wyz_f*byy + Wzz_f*byz;
          Wbzz = Wxz_f*bxz + Wyz_f*byz + Wzz_f*bzz;
        }

        if (use_para) { // parallel viscosity (Pi0) RF1 eqn(4.129)
        nu_para = 0.5*(nu(DiffProcess::nu_para,k,j,i)
                      + nu(DiffProcess::nu_para,k,j-1,i));
        // Pi 0
        Pi_yx += -1.5 * HII * rho_face * nu_para * bWb * bxy;
        Pi_yy += -1.5 * HII * rho_face * nu_para * bWb * (byy - 1.0/3.0);
        Pi_yz += -1.5 * HII * rho_face * nu_para * bWb * byz;
        }
        if (use_perp1) { // perpendicular viscosity (Pi1) RF1 eqn(4.130)
          nu_perp = 0.5*(nu(DiffProcess::nu_perp1,k,j,i)
                        + nu(DiffProcess::nu_perp1,k,j-1,i));;

          Real WPerpyx = Ixy*WIxx + Iyy*WIyx + Iyz*WIzx;
          Real WPerpyy = Ixy*WIxy + Iyy*WIyy + Iyz*WIzy;
          Real WPerpyz = Ixy*WIxz + Iyy*WIyz + Iyz*WIzz;

          // Pi1
          Pi_yx += -HII * rho_face * nu_perp * (WPerpyx + 0.5*(-bxy)*bWb);
          Pi_yy += -HII * rho_face * nu_perp * (WPerpyy + 0.5*(1.0-byy)*bWb);
          Pi_yz += -HII * rho_face * nu_perp * (WPerpyz + 0.5*(-byz)*bWb);
        }
        if (use_perp2) { // Perpendicular viscosity (Pi2) RF1 eqn(4.131)
        nu_perp = 0.5*(nu(DiffProcess::nu_perp2,k,j,i)
                      + nu(DiffProcess::nu_perp2,k,j-1,i));
          // I_perp dot W dot bb
          Real IWbyx = Ixy*Wbxx + Iyy*Wbyx + Iyz*Wbzx;
          Real IWbyy = Ixy*Wbxy + Iyy*Wbyy + Iyz*Wbzy;
          Real IWbyz = Ixy*Wbxz + Iyy*Wbyz + Iyz*Wbzz;

          // bb dot W dot I_Perp
          Real bWIyx = bxy*WIxx + byy*WIyx + byz*WIzx;
          Real bWIyy = bxy*WIxy + byy*WIyy + byz*WIzy;
          Real bWIyz = bxy*WIxz + byy*WIyz + byz*WIzz;

          // Pi 2
          Pi_yx += -HII * rho_face * nu_perp * 4* (IWbyx + bWIyx);
          Pi_yy += -HII * rho_face * nu_perp * 4* (IWbyy + bWIyy);
          Pi_yz += -HII * rho_face * nu_perp * 4* (IWbyz + bWIyz);
        }
        if (use_wedge1) { // Wedge viscosity Pi3 RF1 eqn(4.132)
          nu_wedge =  0.5*(nu(DiffProcess::nu_wedge1,k,j,i)
                          + nu(DiffProcess::nu_wedge1,k,j-1,i));
          // I_perp dot W
          Real IWyx = Ixy*Wxx_f + Iyy*Wxy_f + Iyz*Wxz_f;
          Real IWyy = Ixy*Wxy_f + Iyy*Wyy_f + Iyz*Wyz_f;
          Real IWyz = Ixy*Wxz_f + Iyy*Wyz_f + Iyz*Wzz_f;

          // b cross WI
          Real bcWIyx = bz*WIxx - bx*WIzx;
          Real bcWIyy = bz*WIxy - bx*WIzy;
          Real bcWIyz = bz*WIxz - bx*WIzz;

          //IW cross B
          Real IWcbyx = IWyy*bz - IWyz*by;
          Real IWcbyy = IWyz*bx - IWyx*bz;
          Real IWcbyz = IWyx*by - IWyy*bx;

          // Pi 3
          Pi_yx_disp += HII*0.5*rho_face*nu_wedge*(bcWIyx - IWcbyx);
          Pi_yy_disp += HII*0.5*rho_face*nu_wedge*(bcWIyy - IWcbyy);
          Pi_yz_disp += HII*0.5*rho_face*nu_wedge*(bcWIyz - IWcbyz);
        }
        if (use_wedge2) { // Wedge viscoisty Pi4 RF1 eqn(4.133)
          nu_wedge =  0.5*(nu(DiffProcess::nu_wedge2,k,j,i)
                          + nu(DiffProcess::nu_wedge2,k,j-1,i));
          // bb dot W
          Real bWyx = bxy*Wxx_f + byy*Wxy_f + byz*Wxz_f;
          Real bWyy = bxy*Wxy_f + byy*Wyy_f + byz*Wyz_f;
          Real bWyz = bxy*Wxz_f + byy*Wyz_f + byz*Wzz_f;

          // b cross (W dot bb)
          Real bcWbyx = bz*Wbxx - bx*Wbzx;
          Real bcWbyy = bz*Wbxy - bx*Wbzy;
          Real bcWbyz = bz*Wbxz - bx*Wbzz;

          // (bb dot W) cross b
          Real bWcbyx = bWyy*bz - bWyz*by;
          Real bWcbyy = bWyz*bx - bWyx*bz;
          Real bWcbyz = bWyx*by - bWyy*bx;

          // Pi 4
          Pi_yx_disp += HII*2.0*rho_face*nu_wedge*(bcWbyx - bWcbyx);
          Pi_yy_disp += HII*2.0*rho_face*nu_wedge*(bcWbyy - bWcbyy);
          Pi_yz_disp += HII*2.0*rho_face*nu_wedge*(bcWbyz - bWcbyz);
          }
        if (use_neutral || use_para || use_perp1 || use_perp2) {
          x2flux(IM1,k,j,i) += Pi_yx;
          x2flux(IM2,k,j,i) += Pi_yy;
          x2flux(IM3,k,j,i) += Pi_yz;
        }
        if (use_wedge1 || use_wedge2) {
          x2flux_disp(IM1,k,j,i) += Pi_yx_disp;
          x2flux_disp(IM2,k,j,i) += Pi_yy_disp;
          x2flux_disp(IM3,k,j,i) += Pi_yz_disp;
          if(!STS_ENABLED) {
            x2flux(IM1,k,j,i) += Pi_yx_disp;
            x2flux(IM2,k,j,i) += Pi_yy_disp;
            x2flux(IM3,k,j,i) += Pi_yz_disp;
          }
        }
        if (NON_BAROTROPIC_EOS) {
          if (use_neutral || use_para || use_perp1 || use_perp2) {
            x2flux(IEN,k,j,i) += 0.5*((p(IM1,k,j-1,i) + p(IM1,k,j,i))*Pi_yx +
                                      (p(IM2,k,j-1,i) + p(IM2,k,j,i))*Pi_yy +
                                      (p(IM3,k,j-1,i) + p(IM3,k,j,i))*Pi_yz);
          }
          if(use_wedge1 || use_wedge2) {
          x2flux_disp(IEN,k,j,i) += 0.5*((p(IM1,k,j-1,i) + p(IM1,k,j,i))*Pi_yx_disp +
                                         (p(IM2,k,j-1,i) + p(IM2,k,j,i))*Pi_yy_disp +
                                         (p(IM3,k,j-1,i) + p(IM3,k,j,i))*Pi_yz_disp);
            if(!STS_ENABLED) {
              x2flux(IEN,k,j,i) += 0.5*((p(IM1,k,j-1,i) + p(IM1,k,j,i))*Pi_yx_disp +
                                        (p(IM2,k,j-1,i) + p(IM2,k,j,i))*Pi_yy_disp +
                                        (p(IM3,k,j-1,i) + p(IM3,k,j,i))*Pi_yz_disp);
            } // end if STS
          } // end if wedge
        } // end if NON_BAROTROPIC_EOS
      } // i loop
    } // j loop
  } // k loop
}  // end if
  il = is, iu = ie, jl = js, ju = je;
  if (MAGNETIC_FIELDS_ENABLED) {
    if (f2) // 2D or 3D MHD limits
      il = is-1, iu = ie+1, jl = js-1, ju = je+1;
    else // 1D MHD limits
      il = is-1, iu = ie+1;
  }
  if (f3) { // k direction
    for (int k=ks; k<=ke+1; ++k) {
      for (int j=jl; j<=ju; ++j) {
      #pragma omp simd
        for (int i=il; i<=iu; ++i) {
          Real nu_iso = 0.0, nu_para = 0.0, nu_perp = 0.0, nu_wedge = 0.0;
          Real rho_face = 0.0;
          Real B = 0.0, Bx = 0.0, By = 0.0, Bz = 0.0;
          Real bx = 0.0, by = 0.0, bz = 0.0;
          Real bxx = 0.0, bxy = 0.0, bxz = 0.0, byy = 0.0, byz = 0.0, bzz = 0.0;
          Real Ixx = 0.0, Ixy = 0.0, Ixz = 0.0, Iyy = 0.0, Iyz = 0.0, Izz = 0.0;
          Real WIxx = 0.0, WIxy = 0.0, WIxz = 0.0,
               WIyx = 0.0, WIyy = 0.0, WIyz = 0.0,
               WIzx = 0.0, WIzy = 0.0, WIzz = 0.0;
          Real Wbxx = 0.0, Wbxy = 0.0, Wbxz = 0.0,
               Wbyx = 0.0, Wbyy = 0.0, Wbyz = 0.0,
               Wbzx = 0.0, Wbzy = 0.0, Wbzz = 0.0;
          Real bWb = 0.0;
          Real Pi_xx = 0.0, Pi_xy = 0.0, Pi_xz = 0.0,
               Pi_yx = 0.0, Pi_yy = 0.0, Pi_yz = 0.0,
               Pi_zx = 0.0, Pi_zy = 0.0, Pi_zz = 0.0;
          Real Pi_xx_disp = 0.0, Pi_xy_disp = 0.0, Pi_xz_disp = 0.0,
               Pi_yx_disp = 0.0, Pi_yy_disp = 0.0, Pi_yz_disp = 0.0,
               Pi_zx_disp = 0.0, Pi_zy_disp = 0.0, Pi_zz_disp = 0.0;

          rho_face = 0.5*(p_i(IDN,k,j,i) + p_i(IDN,k-1,j,i));

          Real dvx_dx =
          ( 0.5*(p_i(IM1,k,  j,i+1) + p_i(IM1,k-1,j,i+1))
          - 0.5*(p_i(IM1,k,  j,i-1) + p_i(IM1,k-1,j,i-1)) )
          / (pco_->dx1v(i-1) + pco_->dx1v(i));

          Real dvy_dx =
            pco_->h2v(i)
            * ( 0.5*(p_i(IM2,k,  j,i+1) + p_i(IM2,k-1,j,i+1)) / pco_->h2v(i+1)
              - 0.5*(p_i(IM2,k,  j,i-1) + p_i(IM2,k-1,j,i-1)) / pco_->h2v(i-1) )
            / (pco_->dx1v(i-1) + pco_->dx1v(i));

          Real dvz_dx =
            pco_->h31v(i)
            * ( 0.5*(p_i(IM3,k,  j,i+1) + p_i(IM3,k-1,j,i+1)) / pco_->h31v(i+1)
              - 0.5*(p_i(IM3,k,  j,i-1) + p_i(IM3,k-1,j,i-1)) / pco_->h31v(i-1) )
            / (pco_->dx1v(i-1) + pco_->dx1v(i));

          Real dvx_dy =
            0.5*( (p_i(IM1,k,  j+1,i) + p_i(IM1,k-1,j+1,i))
                - (p_i(IM1,k,  j-1,i) + p_i(IM1,k-1,j-1,i)) )
            / pco_->h2v(i)
            / (pco_->dx2v(j-1) + pco_->dx2v(j));

          Real dvy_dy =
            0.5*( (p_i(IM2,k,  j+1,i) + p_i(IM2,k-1,j+1,i))
                - (p_i(IM2,k,  j-1,i) + p_i(IM2,k-1,j-1,i)) )
            / pco_->h2v(i)
            / (pco_->dx2v(j-1) + pco_->dx2v(j));

          Real dvz_dy =
          0.5 * pco_->h32v(j)
          * ( (p_i(IM3,k,  j+1,i) + p_i(IM3,k-1,j+1,i)) / pco_->h32v(j+1)
            - (p_i(IM3,k,  j-1,i) + p_i(IM3,k-1,j-1,i)) / pco_->h32v(j-1) )
          / pco_->h2v(i)
          / (pco_->dx2v(j-1) + pco_->dx2v(j));

            Real dvx_dz =
            (p_i(IM1,k,j,i) - p_i(IM1,k-1,j,i))
            / pco_->dx3v(k-1);

          Real dvy_dz =
            (p_i(IM2,k,j,i) - p_i(IM2,k-1,j,i))
            / (pco_->h31v(i) * pco_->h32v(j) * pco_->dx3v(k-1));

          Real dvz_dz =
          (p_i(IM3,k,j,i) - p_i(IM3,k-1,j,i))
          / (pco_->dx3v(k-1) * pco_->h31v(i) * pco_->h32v(j))
          + 0.5*(p_i(IM1,k,j,i) + p_i(IM1,k-1,j,i)) * pco_->dh31vd1(i) / pco_->h31v(i)
          + 0.5*(p_i(IM2,k,j,i) + p_i(IM2,k-1,j,i))
               * pco_->dh32vd2(j) / pco_->h32v(j) / pco_->h2v(i);

          Real divv = 0.5 * (div_vel_(k,j,i) + div_vel_(k-1,j,i));

          Real Wxx_f = 2.0*(dvx_dx - divv/3.0);
          Real Wyy_f = 2.0*(dvy_dy - divv/3.0);
          Real Wzz_f = 2.0*(dvz_dz - divv/3.0);

          Real Wxy_f = (dvx_dy + dvy_dx);
          Real Wxz_f = (dvx_dz + dvz_dx);
          Real Wyz_f = (dvy_dz + dvz_dy);

          Real HII = (HydroDiffusion::nu_aniso_neutral>0.0) ?
                     0.5*(HydroDiffusion::ion_frac(k,j,i)
                        + HydroDiffusion::ion_frac(k-1,j,i))
                     : 1.0;

          if (use_neutral) { // neutral part is isotropic
            nu_iso = 0.5*(nu(DiffProcess::nu_neutral,k,j,i)
                        + nu(DiffProcess::nu_neutral,k-1,j,i));
            Pi_zx += -(1-HII)*rho_face*nu_iso*Wxz_f;
            Pi_zy += -(1-HII)*rho_face*nu_iso*Wyz_f;
            Pi_zz += -(1-HII)*rho_face*nu_iso*Wzz_f;
            }

          // build anisotropic dependencies
          if (use_B) {
            Bx = 0.5*(pmb_->pfield->bcc(IB1, k,j,i) + pmb_->pfield->bcc(IB1,k-1,j,i));
            By = 0.5*(pmb_->pfield->bcc(IB2,k,j,i) + pmb_->pfield->bcc(IB2,k-1,j,i));
            Bz = pmb_->pfield->b.x3f(k,j,i);
            B = std::sqrt(Bx*Bx + By*By + Bz*Bz);

            if (B > TINY_NUMBER) {
              bx = Bx/B;
              by = By/B;
              bz = Bz/B;
            } else {
              bx=by=bz=0.0;
            }

            // bb
            bxx = bx*bx;
            bxy = bx*by;
            bxz = bx*bz;
            byy = by*by;
            byz = by*bz;
            bzz = bz*bz;

            // b dot W dot b
            Real wbx = Wxx_f*bx + Wxy_f*by + Wxz_f*bz;
            Real wby = Wxy_f*bx + Wyy_f*by + Wyz_f*bz;
            Real wbz = Wxz_f*bx + Wyz_f*by + Wzz_f*bz;
            bWb = bx*wbx + by*wby + bz*wbz;
          }
          if (use_I) {
            // I_perp tensor
            Ixx = 1-bxx;
            Ixy = -bxy;
            Ixz = -bxz;
            Iyy = 1-byy;
            Iyz = -byz;
            Izz = 1-bzz;

            // W dot I_perp
            WIxx = Wxx_f*Ixx + Wxy_f*Ixy + Wxz_f*Ixz;
            WIxy = Wxx_f*Ixy + Wxy_f*Iyy + Wxz_f*Iyz;
            WIxz = Wxx_f*Ixz + Wxy_f*Iyz + Wxz_f*Izz;

            WIyx = Wxy_f*Ixx + Wyy_f*Ixy + Wyz_f*Ixz;
            WIyy = Wxy_f*Ixy + Wyy_f*Iyy + Wyz_f*Iyz;
            WIyz = Wxy_f*Ixz + Wyy_f*Iyz + Wyz_f*Izz;

            WIzx = Wxz_f*Ixx + Wyz_f*Ixy + Wzz_f*Ixz;
            WIzy = Wxz_f*Ixy + Wyz_f*Iyy + Wzz_f*Iyz;
            WIzz = Wxz_f*Ixz + Wyz_f*Iyz + Wzz_f*Izz;
          }

          if (use_Wbb) {
            // W dot bb
            Wbxx = Wxx_f*bxx + Wxy_f*bxy + Wxz_f*bxz;
            Wbxy = Wxx_f*bxy + Wxy_f*byy + Wxz_f*byz;
            Wbxz = Wxx_f*bxz + Wxy_f*byz + Wxz_f*bzz;

            Wbyx = Wxy_f*bxx + Wyy_f*bxy + Wyz_f*bxz;
            Wbyy = Wxy_f*bxy + Wyy_f*byy + Wyz_f*byz;
            Wbyz = Wxy_f*bxz + Wyy_f*byz + Wyz_f*bzz;

            Wbzx = Wxz_f*bxx + Wyz_f*bxy + Wzz_f*bxz;
            Wbzy = Wxz_f*bxy + Wyz_f*byy + Wzz_f*byz;
            Wbzz = Wxz_f*bxz + Wyz_f*byz + Wzz_f*bzz;
          }
          if (use_para) { // parallel viscosity (Pi0) RF1 eqn(4.129)
          nu_para = 0.5*(nu(DiffProcess::nu_para,k,j,i)
                       + nu(DiffProcess::nu_para,k-1,j,i));
          // Pi 0
          Pi_zx += -1.5 * HII * rho_face * nu_para * bWb * bxz;
          Pi_zy += -1.5 * HII * rho_face * nu_para * bWb * byz;
          Pi_zz += -1.5 * HII * rho_face * nu_para * bWb * (bzz - 1.0/3.0);
          }

          if (use_perp1) { // perpendicular viscosity (Pi1) RF1 eqn(4.130)
            nu_perp = 0.5*(nu(DiffProcess::nu_perp1,k,j,i)
                         + nu(DiffProcess::nu_perp1,k-1,j,i));

            // W_Perp = I_Perp dot W_Perp dot I_Perp
            Real WPerpzx = Ixz*WIxx + Iyz*WIyx + Izz*WIzx;
            Real WPerpzy = Ixz*WIxy + Iyz*WIyy + Izz*WIzy;
            Real WPerpzz = Ixz*WIxz + Iyz*WIyz + Izz*WIzz;

            // Pi1
            Pi_zx += -HII * rho_face * nu_perp * (WPerpzx + 0.5*(-bxz)*bWb);
            Pi_zy += -HII * rho_face * nu_perp * (WPerpzy + 0.5*(-byz)*bWb);
            Pi_zz += -HII * rho_face * nu_perp * (WPerpzz + 0.5*(1.0-bzz)*bWb);
          }
          if (use_perp2) { // Perendicular viscoisty (Pi2) RF1 eqn(4.131)
          nu_perp = 0.5*(nu(DiffProcess::nu_perp2,k,j,i)
                       + nu(DiffProcess::nu_perp2,k-1,j,i));
            // I_perp dot W dot bb
            Real IWbzx = Ixz*Wbxx + Iyz*Wbyx + Izz*Wbzx;
            Real IWbzy = Ixz*Wbxy + Iyz*Wbyy + Izz*Wbzy;
            Real IWbzz = Ixz*Wbxz + Iyz*Wbyz + Izz*Wbzz;

            // bb dot W dot I_Perp
            Real bWIzx = bxz*WIxx + byz*WIyx + bzz*WIzx;
            Real bWIzy = bxz*WIxy + byz*WIyy + bzz*WIzy;
            Real bWIzz = bxz*WIxz + byz*WIyz + bzz*WIzz;

            // Pi 2
            Pi_zx += -HII * rho_face * nu_perp * 4* (IWbzx + bWIzx);
            Pi_zy += -HII * rho_face * nu_perp * 4* (IWbzy + bWIzy);
            Pi_zz += -HII * rho_face * nu_perp * 4* (IWbzz + bWIzz);
          }
          if (use_wedge1) { // Wedge viscoisty (Pi3) RF1 eqn(4.132)
            nu_wedge =  0.5*(nu(DiffProcess::nu_wedge1,k,j,i)
                           + nu(DiffProcess::nu_wedge1,k-1,j,i));

            // I_perp dot W
            Real IWzx = Ixz*Wxx_f + Iyz*Wxy_f + Izz*Wxz_f;
            Real IWzy = Ixz*Wxy_f + Iyz*Wyy_f + Izz*Wyz_f;
            Real IWzz = Ixz*Wxz_f + Iyz*Wyz_f + Izz*Wzz_f;

            // b cross WI
            Real bcWIzx = bx*WIyx - by*WIxx;
            Real bcWIzy = bx*WIyy - by*WIxy;
            Real bcWIzz = bx*WIyz - by*WIxz;

            // IW cross B
            Real IWcbzx = IWzy*bz - IWzz*by;
            Real IWcbzy = IWzz*bx - IWzx*bz;
            Real IWcbzz = IWzx*by - IWzy*bx;

            // Pi 3
            Pi_zx_disp += HII*0.5*rho_face*nu_wedge*(bcWIzx - IWcbzx);
            Pi_zy_disp += HII*0.5*rho_face*nu_wedge*(bcWIzy - IWcbzy);
            Pi_zz_disp += HII*0.5*rho_face*nu_wedge*(bcWIzz - IWcbzz);
          }
          if (use_wedge2) { // Wedge viscoisty (Pi4) RF1 eqn(4.133)
            nu_wedge =  0.5*(nu(DiffProcess::nu_wedge2,k,j,i)
                           + nu(DiffProcess::nu_wedge2,k-1,j,i));

            // bb dot W
            Real bWzx = bxz*Wxx_f + byz*Wxy_f + bzz*Wxz_f;
            Real bWzy = bxz*Wxy_f + byz*Wyy_f + bzz*Wyz_f;
            Real bWzz = bxz*Wxz_f + byz*Wyz_f + bzz*Wzz_f;

            // b cross (W dot bb)
            Real bcWbzx = bx*Wbyx - by*Wbxx;
            Real bcWbzy = bx*Wbyy - by*Wbxy;
            Real bcWbzz = bx*Wbyz - by*Wbxz;

            // (bb dot W) cross b
            Real bWcbzx = bWzy*bz - bWzz*by;
            Real bWcbzy = bWzz*bx - bWzx*bz;
            Real bWcbzz = bWzx*by - bWzy*bx;

            // Pi 4
            Pi_zx_disp += HII*2*rho_face*nu_wedge*(bcWbzx - bWcbzx);
            Pi_zy_disp += HII*2*rho_face*nu_wedge*(bcWbzy - bWcbzy);
            Pi_zz_disp += HII*2*rho_face*nu_wedge*(bcWbzz - bWcbzz);
            }
          if (use_neutral || use_para || use_perp1 || use_perp2) {
          x3flux(IM1,k,j,i) += Pi_zx;
          x3flux(IM2,k,j,i) += Pi_zy;
          x3flux(IM3,k,j,i) += Pi_zz;
          }
          if (use_wedge1 || use_wedge2) {
          x3flux_disp(IM1,k,j,i) += Pi_zx_disp;
          x3flux_disp(IM2,k,j,i) += Pi_zy_disp;
          x3flux_disp(IM3,k,j,i) += Pi_zz_disp;
            if (!STS_ENABLED) {
              x3flux(IM1,k,j,i) += Pi_zx_disp;
              x3flux(IM2,k,j,i) += Pi_zy_disp;
              x3flux(IM3,k,j,i) += Pi_zz_disp;
            }
        }
          if (NON_BAROTROPIC_EOS) {
            if (use_neutral || use_para || use_perp1 || use_perp2) {
              x3flux(IEN,k,j,i) += 0.5*((p(IM1,k-1,j,i) + p(IM1,k,j,i))*Pi_zx +
                                        (p(IM2,k-1,j,i) + p(IM2,k,j,i))*Pi_zy +
                                        (p(IM3,k-1,j,i) + p(IM3,k,j,i))*Pi_zz);
            }
            if (use_wedge1 || use_wedge2) {
              x3flux_disp(IEN,k,j,i) += 0.5*((p(IM1,k-1,j,i) + p(IM1,k,j,i))*Pi_zx_disp +
                                             (p(IM2,k-1,j,i) + p(IM2,k,j,i))*Pi_zy_disp +
                                             (p(IM3,k-1,j,i) + p(IM3,k,j,i))*Pi_zz_disp);
              if (!STS_ENABLED) {
                x3flux(IEN,k,j,i) += 0.5*((p(IM1,k-1,j,i) + p(IM1,k,j,i))*Pi_zx_disp +
                                          (p(IM2,k-1,j,i) + p(IM2,k,j,i))*Pi_zy_disp +
                                          (p(IM3,k-1,j,i) + p(IM3,k,j,i))*Pi_zz_disp);
              } // end if STS
            } // end if wedge
          } // end if NON_BAROTROPIC_EOS
        } // i loop
      } // j loop
    } // k loop
  } // end if
  return;
}

//---------------------------------------------------------------------------------------
//! \fn void HydroDiffusion::DivVelocity
//! \brief Calculate divergence of momenta

void HydroDiffusion::DivVelocity(const AthenaArray<Real> &prim,
                                 AthenaArray<Real> &div_vel) {
  const bool f2 = pmb_->pmy_mesh->f2;
  const bool f3 = pmb_->pmy_mesh->f3;

  int is = pmb_->is; int js = pmb_->js; int ks = pmb_->ks;
  int ie = pmb_->ie; int je = pmb_->je; int ke = pmb_->ke;
  int il = is - 1; int iu = ie + 1;
  int jl, ju, kl, ku;
  Real area_p1, area;
  Real vel_p1, vel;

  if (!f2) // 1D
    jl = js, ju = je, kl = ks, ku = ke;
  else if (!f3) // 2D
    jl = js-1, ju = je+1, kl = ks, ku = ke;
  else // 3D
    jl = js-1, ju = je+1, kl = ks-1, ku = ke+1;

  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      // calculate x1-flux divergence
      pmb_->pcoord->Face1Area(k, j, il, iu+1, x1area_);
#pragma omp simd private(area_p1, area, vel_p1, vel)
      for (int i=il; i<=iu; ++i) {
        area_p1 = x1area_(i+1);
        area    = x1area_(i);
        vel_p1  = 0.5*(prim(IM1,k,j,i+1) + prim(IM1,k,j,i  ));
        vel     = 0.5*(prim(IM1,k,j,i  ) + prim(IM1,k,j,i-1));
        div_vel(k,j,i) = area_p1*vel_p1 - area*vel;
      }
      // calculate x2-flux divergnece
      if (f2) {
        pmb_->pcoord->Face2Area(k, j  , il, iu, x2area_);
        pmb_->pcoord->Face2Area(k, j+1, il, iu, x2area_p1_);
#pragma omp simd private(area_p1, area, vel_p1, vel)
        for (int i=il; i<=iu; ++i) {
          area_p1 = x2area_p1_(i);
          area    = x2area_(i);
          vel_p1  = 0.5*(prim(IM2,k,j+1,i) + prim(IM2,k,j  ,i));
          vel     = 0.5*(prim(IM2,k,j  ,i) + prim(IM2,k,j-1,i));
          div_vel(k,j,i) += area_p1*vel_p1 - area*vel;
        }
      }
      // calculate x3-flux divergnece
      if (f3) {
        pmb_->pcoord->Face3Area(k  , j, il, iu, x3area_);
        pmb_->pcoord->Face3Area(k+1, j, il, iu, x3area_p1_);
#pragma omp simd private(area_p1, area, vel_p1, vel)
        for (int i=il; i<=iu; ++i) {
          area_p1 = x3area_p1_(i);
          area    = x3area_(i);
          vel_p1  = 0.5*(prim(IM3,k+1,j,i) + prim(IM3, k  ,j,i));
          vel     = 0.5*(prim(IM3,k  ,j,i) + prim(IM3, k-1,j,i));
          div_vel(k,j,i) += area_p1*vel_p1 - area*vel;
        }
      }
      pmb_->pcoord->CellVolume(k,j,il,iu,vol_);
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
        div_vel(k,j,i) = div_vel(k,j,i)/vol_(i);
      }
    }
  }
  return;
}

// v_{x1;x1}  covariant derivative at x1 interface
void HydroDiffusion::FaceXdx(const int k, const int j, const int il, const int iu,
                             const AthenaArray<Real> &prim, AthenaArray<Real> &len) {
#pragma omp simd
  for (int i=il; i<=iu; ++i) {
    len(i) = 2.0*(prim(IM1,k,j,i) - prim(IM1,k,j,i-1)) / pco_->dx1v(i-1);
  }
  return;
}

// v_{x2;x1}+v_{x1;x2}  covariant derivative at x1 interface
void HydroDiffusion::FaceXdy(const int k, const int j, const int il, const int iu,
                             const AthenaArray<Real> &prim, AthenaArray<Real> &len) {
  if (pmb_->pmy_mesh->f2) {
#pragma omp simd
    for (int i=il; i<=iu; ++i) {
      len(i) = pco_->h2f(i)
               * (prim(IM2,k,j,i)/pco_->h2v(i) - prim(IM2,k,j,i-1)/pco_->h2v(i-1))
               / pco_->dx1v(i-1)
               // KGF: add the off-centered quantities first to preserve FP symmetry
               + 0.5*(   (prim(IM1,k,j+1,i) + prim(IM1,k,j+1,i-1))
                         - (prim(IM1,k,j-1,i) + prim(IM1,k,j-1,i-1)) )
               / pco_->h2f(i)
               / (pco_->dx2v(j-1) + pco_->dx2v(j));
    }
  } else {
#pragma omp simd
    for (int i=il; i<=iu; ++i) {
      len(i) = pco_->h2f(i)
               * ( prim(IM2,k,j,i)/pco_->h2v(i) - prim(IM2,k,j,i-1)/pco_->h2v(i-1) )
               / pco_->dx1v(i-1);
    }
  }
  return;
}

// v_{x3;x1}+v_{x1;x3}  covariant derivative at x1 interface
void HydroDiffusion::FaceXdz(const int k, const int j, const int il, const int iu,
                             const AthenaArray<Real> &prim, AthenaArray<Real> &len) {
  if (pmb_->pmy_mesh->f3) {
#pragma omp simd
    for (int i=il; i<=iu; ++i) {
      len(i) = pco_->h31f(i)
               * (prim(IM3,k,j,i)/pco_->h31v(i) - prim(IM3,k,j,i-1)/pco_->h31v(i-1))
               / pco_->dx1v(i-1)
               // KGF: add the off-centered quantities first to preserve FP symmetry
               + 0.5*(   (prim(IM1,k+1,j,i) + prim(IM1,k+1,j,i-1))
                         - (prim(IM1,k-1,j,i) + prim(IM1,k-1,j,i-1)) )
               / pco_->h31f(i)/pco_->h32v(j) // note, more terms than FaceXdy() line
               / (pco_->dx3v(k-1) + pco_->dx3v(k));
    }
  } else {
#pragma omp simd
    for (int i=il; i<=iu; ++i)
      len(i) = pco_->h31f(i)
               * ( prim(IM3,k,j,i)/pco_->h31v(i) - prim(IM3,k,j,i-1)/pco_->h31v(i-1) )
               / pco_->dx1v(i-1);
  }
  return;
}

// v_{x1;x2}+v_{x2;x1}  covariant derivative at x2 interface
void HydroDiffusion::FaceYdx(const int k, const int j, const int il, const int iu,
                             const AthenaArray<Real> &prim, AthenaArray<Real> &len) {
  if (pmb_->pmy_mesh->f2) {
#pragma omp simd
    for (int i=il; i<=iu; ++i) {
      len(i) = (prim(IM1,k,j,i) - prim(IM1,k,j-1,i)) / pco_->h2v(i) / pco_->dx2v(j-1)
               + pco_->h2v(i)*0.5*
               (  (prim(IM2,k,j,i+1) + prim(IM2,k,j-1,i+1)) /pco_->h2v(i+1)
                  - (prim(IM2,k,j,i-1) + prim(IM2,k,j-1,i-1)) /pco_->h2v(i-1)
                  ) / (pco_->dx1v(i-1) + pco_->dx1v(i));
    }
  } else {
#pragma omp simd
    for (int i=il; i<=iu; ++i) {
      len(i) = pco_->h2v(i)
               * ( prim(IM2,k,j,i+1)/pco_->h2v(i+1) - prim(IM2,k,j,i-1)/pco_->h2v(i-1) )
               / (pco_->dx1v(i-1) + pco_->dx1v(i));
    }
  }
  return;
}

// v_{x2;x2}  covariant derivative at x2 interface
void HydroDiffusion::FaceYdy(const int k, const int j, const int il, const int iu,
                             const AthenaArray<Real> &prim, AthenaArray<Real> &len) {
  if (pmb_->pmy_mesh->f2) {
#pragma omp simd
    for (int i=il; i<=iu; ++i) {
      len(i) = 2.0*(prim(IM2,k,j,i) - prim(IM2,k,j-1,i)) / pco_->h2v(i) / pco_->dx2v(j-1)
               + (prim(IM1,k,j,i) + prim(IM1,k,j-1,i)) / pco_->h2v(i) * pco_->dh2vd1(i);
    }
  } else {
#pragma omp simd
    for (int i=il; i<=iu; ++i)
      len(i) = 2.0*prim(IM1,k,j,i) / pco_->h2v(i) * pco_->dh2vd1(i);
  }
  return;
}

// v_{x3;x2}+v_{x2;x3}  covariant derivative at x2 interface
void HydroDiffusion::FaceYdz(const int k, const int j, const int il, const int iu,
                             const AthenaArray<Real> &prim, AthenaArray<Real> &len) {
  if (pmb_->pmy_mesh->f3) {
#pragma omp simd
    for (int i=il; i<=iu; ++i) {
      len(i) = pco_->h32f(j)
               * ( prim(IM3,k,j,i)/pco_->h32v(j) - prim(IM3,k,j-1,i)/pco_->h32v(j-1) )
               / pco_->h2v(i) / pco_->dx2v(j-1)
               // KGF: add the off-centered quantities first to preserve FP symmetry
               + 0.5*(    (prim(IM2,k+1,j,i) + prim(IM2,k+1,j-1,i))
                          - (prim(IM2,k-1,j,i) + prim(IM2,k-1,j-1,i)) )
               / pco_->h31v(i)
               / (pco_->dx3v(k-1) + pco_->dx3v(k))
               // edge case for polar boundary
               / ( pco_->IsPole(j) ? 1.0 : pco_->h32f(j) );
    }
  } else if (pmb_->pmy_mesh->f2) {
#pragma omp simd
    for (int i=il; i<=iu; ++i) {
      len(i) = pco_->h32f(j)
               * ( prim(IM3,k,j,i)/pco_->h32v(j) - prim(IM3,k,j-1,i)/pco_->h32v(j-1) )
               / pco_->h2v(i) / pco_->dx2v(j-1);
    }
  } else {
#pragma omp simd
    for (int i=il; i<=iu; ++i)
      len(i) = 0.0;
  }
  return;
}

// v_{x1;x3}+v_{x3;x1}  covariant derivative at x3 interface
void HydroDiffusion::FaceZdx(const int k, const int j, const int il, const int iu,
                             const AthenaArray<Real> &prim, AthenaArray<Real> &len) {
  if (pmb_->pmy_mesh->f3) {
#pragma omp simd
    for (int i=il; i<=iu; ++i) {
      len(i) = (prim(IM1,k,j,i) - prim(IM1,k-1,j,i))/pco_->dx3v(k-1)
               + 0.5*pco_->h31v(i)*(
                   (prim(IM3,k,j,i+1) + prim(IM3,k-1,j,i+1))/pco_->h31v(i+1)
                   -(prim(IM3,k,j,i-1) + prim(IM3,k-1,j,i-1))/pco_->h31v(i-1)
                                    ) / (pco_->dx1v(i-1) + pco_->dx1v(i));
    }
  } else {
#pragma omp simd
    for (int i=il; i<=iu; ++i) {
      len(i) = pco_->h31v(i)
               * ( prim(IM3,k,j,i+1)/pco_->h31v(i+1) - prim(IM3,k,j,i-1)/pco_->h31v(i-1) )
               / (pco_->dx1v(i-1) + pco_->dx1v(i));
    }
  }
  return;
}

// v_{x2;x3}+v_{x3;x2}  covariant derivative at x3 interface
void HydroDiffusion::FaceZdy(const int k, const int j, const int il, const int iu,
                             const AthenaArray<Real> &prim, AthenaArray<Real> &len) {
  if (pmb_->pmy_mesh->f3) {
#pragma omp simd
    for (int i=il; i<=iu; ++i) {
      len(i) = (prim(IM2,k,j,i) - prim(IM2,k-1,j,i))
               / pco_->h31v(i) / pco_->h32v(j) / pco_->dx3v(k-1)
               + 0.5*pco_->h32v(j)
               * ( (prim(IM3,k,j+1,i) + prim(IM3,k-1,j+1,i))/pco_->h32v(j+1)
                   -(prim(IM3,k,j-1,i) + prim(IM3,k-1,j-1,i))/pco_->h32v(j-1) )
               / pco_->h2v(i) / (pco_->dx2v(j-1) + pco_->dx2v(j));
    }
  } else if (pmb_->pmy_mesh->f2) {
#pragma omp simd
    for (int i=il; i<=iu; ++i) {
      len(i) = pco_->h32v(j)
               * ( prim(IM3,k,j+1,i)/pco_->h32v(j+1) - prim(IM3,k,j-1,i)/pco_->h32v(j-1) )
               / pco_->h2v(i) / (pco_->dx2v(j-1) + pco_->dx2v(j));
    }
  } else {
#pragma omp simd
    for (int i=il; i<=iu; ++i)
      len(i) = 0.0;
  }
  return;
}

// v_{x3;x3}  covariant derivative at x3 interface
void HydroDiffusion::FaceZdz(const int k, const int j, const int il, const int iu,
                             const AthenaArray<Real> &prim, AthenaArray<Real> &len) {
  if (pmb_->pmy_mesh->f3) {
#pragma omp simd
    for (int i=il; i<=iu; ++i) {
      len(i) = 2.0*(prim(IM3,k,j,i) - prim(IM3,k-1,j,i))
               / pco_->dx3v(k-1) / pco_->h31v(i) / pco_->h32v(j)
               + ((prim(IM1,k,j,i) + prim(IM1,k-1,j,i))
                  * pco_->dh31vd1(i)/pco_->h31v(i))
               + ((prim(IM2,k,j,i) + prim(IM2,k-1,j,i))
                  * pco_->dh32vd2(j)/pco_->h32v(j)/pco_->h2v(i));
    }
  } else {
#pragma omp simd
    for (int i=il; i<=iu; ++i) {
      len(i) = 2.0*prim(IM1,k,j,i)*pco_->dh31vd1(i)/pco_->h31v(i)
               + 2.0*prim(IM2,k,j,i)*pco_->dh32vd2(j)/pco_->h32v(j)/pco_->h2v(i);
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn ConstViscosity
//! \brief Constant viscosity
void ConstViscosity(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &prim,
                    const AthenaArray<Real> &bcc, int is, int ie, int js, int je,
                    int ks, int ke) {
    if (phdif->zeta_bulk > 0.0) { // bulk viscosity
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i)
          phdif->nu(HydroDiffusion::DiffProcess::zeta,k,j,i) = phdif->zeta_bulk;
      }
    }
  }
  if (phdif->nu_iso > 0.0) { // isotropic viscosity
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i)
          phdif->nu(HydroDiffusion::DiffProcess::iso,k,j,i) = phdif->nu_iso;
      }
    }
  }
  if (phdif->nu_aniso_neutral > 0.0 || phdif->ion_frac_const > 0.0) { // ionisation
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          phdif->nu(HydroDiffusion::DiffProcess::nu_neutral,k,j,i)
                    = phdif->nu_aniso_neutral;
          phdif->ion_frac(k,j,i) = phdif->ion_frac_const;
        }
      }
    }
  }
  if (phdif->nu_aniso_para > 0.0) { // anisotropic parallel viscosity
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          phdif->nu(HydroDiffusion::DiffProcess::nu_para,k,j,i) = phdif->nu_aniso_para;
        }
      }
    }
  }
  if (phdif->nu_aniso_perp1 > 0.0) { // anisotropic perpendicular 1 viscosity
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          phdif->nu(HydroDiffusion::DiffProcess::nu_perp1,k,j,i) = phdif->nu_aniso_perp1;
        }
      }
    }
  }
  if (phdif->nu_aniso_perp2 > 0.0) { // anisotropic perpendicular 2 viscosity
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          phdif->nu(HydroDiffusion::DiffProcess::nu_perp2,k,j,i) = phdif->nu_aniso_perp2;
        }
      }
    }
  }
  if (phdif->nu_aniso_wedge1 > 0.0) { // anisotropic wedge 1 viscosity
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          phdif->nu(HydroDiffusion::DiffProcess::nu_wedge1,k,j,i)
                    = phdif->nu_aniso_wedge1;
        }
      }
    }
  }
  if (phdif->nu_aniso_wedge2 > 0.0) { // anisotropic wedge 2 viscosity
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          phdif->nu(HydroDiffusion::DiffProcess::nu_wedge2,k,j,i)
                    = phdif->nu_aniso_wedge2;
        }
      }
    }
  }
  return;
}
