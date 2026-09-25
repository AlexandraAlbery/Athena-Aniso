//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file conduction.cpp
//! \brief Thermal conduction

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
#include "../../field/field.hpp"
#endif


// Used to prevent energy from cold cells flowing into hot cells due to discretisation
// error with strong parallel conduction
namespace ConductionLimiters{
  // Monotonized-central limiter: minmod(2a, 2b, (a+b)/2)
  inline Real LimiterMC(const Real a, const Real b) {
    if (a*b <= 0.0) return 0.0;
    return std::signbit(a)
           *std::min(std::min(2.0*std::abs(a), 2.0*std::abs(b)), 0.5*std::abs(a + b));
  }
  // SH1: P. Sharma & G.W. "Preserving Monotonicity in Anisotropic Diffusion",
  // Journal of Computational Physics, Nov. 2007, vol. 227, No. 1, pp. 123–142,
  // url: https://w3.pppl.gov/~hammett/gyrofluid/papers/2007/Sharma-Hammett-2007
  // _anisotropic_diffusion.pdf
  // limited transverse gradient at a face
  // limit the two one-sided slopesinside each adjacent cell
  // then limit between the two cells
  inline Real LimitedTransverse(const Real up_R, const Real dn_R,
                                const Real up_L, const Real dn_L) {
    return LimiterMC(LimiterMC(up_R, dn_R), LimiterMC(up_L, dn_L));
  }
  } // namespace

//---------------------------------------------------------------------------------------
//! \fn void HydroDiffusion::ThermalFluxIso
//! \brief Isotropic thermal conduction
void HydroDiffusion::ThermalFluxIso(
     const AthenaArray<Real> &p, AthenaArray<Real> *flx) {
  const bool f2 = pmb_->pmy_mesh->f2;
  const bool f3 = pmb_->pmy_mesh->f3;
  AthenaArray<Real> &x1flux = flx[X1DIR];
  AthenaArray<Real> &x2flux = flx[X2DIR];
  AthenaArray<Real> &x3flux = flx[X3DIR];
  int il, iu, jl, ju, kl, ku;
  int is = pmb_->is; int js = pmb_->js; int ks = pmb_->ks;
  int ie = pmb_->ie; int je = pmb_->je; int ke = pmb_->ke;
  Real kappaf, denf, dTdx, dTdy, dTdz;

  // i-direction
  jl = js, ju = je, kl = ks, ku = ke;
  if (MAGNETIC_FIELDS_ENABLED) {
    if (f2) {
      if (!f3) // 2D
        jl = js - 1, ju = je + 1, kl = ks, ku = ke;
      else // 3D
        jl = js - 1, ju = je + 1, kl = ks - 1, ku = ke + 1;
    }
  }
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma omp simd private(kappaf, denf, dTdx)
      for (int i=is; i<=ie+1; ++i) {
        kappaf = 0.5*(kappa(DiffProcess::iso,k,j,i) + kappa(DiffProcess::iso,k,j,i-1));
        denf = 0.5*(p(IDN,k,j,i) + p(IDN,k,j,i-1));
        dTdx = (p(IPR,k,j,i)/p(IDN,k,j,i) - p(IPR,k,j,i-1)/
                p(IDN,k,j,i-1))/pco_->dx1v(i-1);
        x1flux(k,j,i) -= kappaf*denf*dTdx;
      }
    }
  }

  // j-direction
  il = is, iu = ie, kl = ks, ku = ke;
  if (MAGNETIC_FIELDS_ENABLED) {
    if (!f3) // 2D
      il = is - 1, iu = ie + 1, kl = ks, ku = ke;
    else // 3D
      il = is - 1, iu = ie + 1, kl = ks - 1, ku = ke + 1;
  }
  if (f2) { // 2D or 3D
    AthenaArray<Real> &x2flux = flx[X2DIR];
    for (int k=kl; k<=ku; ++k) {
      for (int j=js; j<=je+1; ++j) {
#pragma omp simd private(kappaf, denf, dTdy)
        for (int i=il; i<=iu; ++i) {
          kappaf = 0.5*(kappa(DiffProcess::iso,k,j,i) + kappa(DiffProcess::iso,k,j-1,i));
          denf = 0.5*(p(IDN,k,j,i) + p(IDN,k,j-1,i));
          dTdy = (p(IPR,k,j,i)/p(IDN,k,j,i) - p(IPR,k,j-1,i)/
                  p(IDN,k,j-1,i))/pco_->h2v(i)/pco_->dx2v(j-1);
          x2flux(k,j,i) -= kappaf*denf*dTdy;
        }
      }
    }
  } // zero flux for 1D

  // k-direction
  il = is, iu = ie, jl = js, ju = je;
  if (MAGNETIC_FIELDS_ENABLED) {
    if (f2) // 2D or 3D
      il = is - 1, iu = ie + 1, jl = js - 1, ju = je + 1;
    else // 1D
      il = is - 1, iu = ie + 1;
  }
  if (f3) { // 3D
    AthenaArray<Real> &x3flux = flx[X3DIR];
    for (int k=ks; k<=ke+1; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma omp simd private(kappaf, denf, dTdz)
        for (int i=il; i<=iu; ++i) {
          kappaf = 0.5*(kappa(DiffProcess::iso,k,j,i) + kappa(DiffProcess::iso,k-1,j,i));
          denf = 0.5*(p(IDN,k,j,i) + p(IDN,k-1,j,i));
          dTdz = (p(IPR,k,j,i)/p(IDN,k,j,i) - p(IPR,k-1,j,i)/
                  p(IDN,k-1,j,i))/pco_->dx3v(k-1)/pco_->h31v(i)/pco_->h32v(j);
          x3flux(k,j,i) -= kappaf*denf*dTdz;
        }
      }
    }
  } // zero flux for 1D/2D
  return;
}

//---------------------------------------------------------------------------------------
//! \fn void HydroDiffusion::ThermalFluxAniso
//! \brief Braginskii thermal conduction
//  BS1: Braginskii, S., “Transport Processes in a Plasma”, Reviews of Plasma Physics,
//  Jan. 1965, vol. 1, pp. 205,
//  https://static.ias.edu/pitp/2016/sites/pitp/files/braginskii_1965-1.pdf.
//  RF1: https://farside.ph.utexas.edu/teaching/plasma/Plasma/node53.html
//  Solves BS1 eqn(4.33) (page 249). Also see first three terms RF1 eqn(4.120)
//  MG1: A. Mignone et. al. "The PLUTO Code for Adaptive Mesh Computations in
//  Astrophysical Fluid Dynamics", The Astrophysical Journal Supplement Series,
//  Jan. 2012, Vol. 198, No. 1, pp.7,
//  URL: https://iopscience.iop.org/article/10.1088/0067-0049/198/1/7
void HydroDiffusion::ThermalFluxAniso(const AthenaArray<Real> &p,
                                      AthenaArray<Real> *flx,
                                      AthenaArray<Real> *flx_disp) {
  using ConductionLimiters::LimitedTransverse;

  const bool f2 = pmb_->pmy_mesh->f2;
  const bool f3 = pmb_->pmy_mesh->f3;

  const bool use_neutral = (HydroDiffusion::kappa_aniso_neutral > 0.0) && calc_para_;
  const bool use_para    = (HydroDiffusion::kappa_aniso_para  > 0.0) && calc_para_;
  const bool use_perp    = (HydroDiffusion::kappa_aniso_perp  > 0.0) && calc_para_;
  const bool use_wedge   = (HydroDiffusion::kappa_aniso_wedge > 0.0) && calc_disp_;

  const bool use_sat = HydroDiffusion::limit_saturated_conduction;
  const bool lim_c = HydroDiffusion::flux_limit_conduction;

  AthenaArray<Real> &x1flux = flx[X1DIR];
  AthenaArray<Real> &x2flux = flx[X2DIR];
  AthenaArray<Real> &x3flux = flx[X3DIR];
  AthenaArray<Real> &x1flux_disp = flx_disp[X1DIR];
  AthenaArray<Real> &x2flux_disp = flx_disp[X2DIR];
  AthenaArray<Real> &x3flux_disp = flx_disp[X3DIR];

  int il, iu, jl, ju, kl, ku;
  int is = pmb_->is; int js = pmb_->js; int ks = pmb_->ks;
  int ie = pmb_->ie; int je = pmb_->je; int ke = pmb_->ke;

  Real HII = 0.0; Real kappa_neutral = 0.0; Real denf = 0.0;
  Real kappa_para = 0.0; Real kappa_perp = 0.0; Real kappa_wedge = 0.0;
  Real Bx = 0.0; Real By = 0.0; Real Bz = 0.0; Real B = 0.0;
  Real bx = 0.0; Real by = 0.0; Real bz = 0.0;
  Real dTdx = 0.0; Real dTdy = 0.0; Real dTdz = 0.0; // Normal gradients
  Real dTdx_t = 0.0; Real dTdy_t = 0.0; Real dTdz_t = 0.0; // Transverse gradients
  Real iso_flux = 0.0; Real para_flux = 0.0; Real perp_flux = 0.0; Real wedge_flux = 0.0;
  Real b_dot_gradT = 0.0;

  // i-direction
  jl = js, ju = je, kl = ks, ku = ke;
  if (MAGNETIC_FIELDS_ENABLED) {
    if (f2) {
      if (!f3) // 2D
        jl = js - 1, ju = je + 1, kl = ks, ku = ke;
      else // 3D
        jl = js - 1, ju = je + 1, kl = ks - 1, ku = ke + 1;
    }
  }
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
    #pragma omp simd
      for (int i=is; i<=ie+1; ++i) {
      iso_flux = 0.0; para_flux = 0.0; perp_flux = 0.0; wedge_flux = 0.0;

      denf = 0.5*(p(IDN,k,j,i) + p(IDN,k,j,i-1));
      // dT/dx normal to x face
      dTdx = (p(IPR,k,j,i)/p(IDN,k,j,i) - p(IPR,k,j,i-1)/p(IDN,k,j,i-1))
              /pco_->dx1v(i-1);
      // transverse dT/dy normal to x face
      if (f2) {
      dTdy = 0.5*((p(IPR,k,j+1,i)/p(IDN,k,j+1,i) + p(IPR,k,j+1,i-1)/p(IDN,k,j+1,i-1))
                - (p(IPR,k,j-1,i)/p(IDN,k,j-1,i) + p(IPR,k,j-1,i-1)/p(IDN,k,j-1,i-1)))
                / pco_->h2f(i)/ (pco_->dx2v(j-1) + pco_->dx2v(j));
      } else {
        dTdy = 0;
      }
      // transverse dT/dz normal to x face
      if (f3) {
        dTdz = 0.5*((p(IPR,k+1,j,i)/p(IDN,k+1,j,i) + p(IPR,k+1,j,i-1)/p(IDN,k+1,j,i-1))
                  - (p(IPR,k-1,j,i)/p(IDN,k-1,j,i) + p(IPR,k-1,j,i-1)/p(IDN,k-1,j,i-1)))
                  / (pco_->dx3v(k-1) + pco_->dx3v(k))/ pco_->h31f(i) / pco_->h32v(j);
      } else {
        dTdz = 0.0;
      }
      // SH1: eq:17 & 18 limited transverse gradients for para/perp
      dTdy_t = dTdy; dTdz_t = dTdz;
      if(lim_c) {
        if (f2) { // limited dT/dy
          dTdy_t = LimitedTransverse(
              ( (p(IPR,k,j+1,i)/p(IDN,k,j+1,i)) - (p(IPR,k,j,i)/p(IDN,k,j,i)) )
              /(pco_->h2f(i)*pco_->dx2v(j)),

              ( (p(IPR,k,j,i)/p(IDN,k,j,i)) - (p(IPR,k,j-1,i)/p(IDN,k,j-1,i)) )
              /(pco_->h2f(i)*pco_->dx2v(j-1)),

              ( (p(IPR,k,j+1,i-1)/p(IDN,k,j+1,i-1)) - (p(IPR,k,j,i-1)/p(IDN,k,j,i-1)) )
              /(pco_->h2f(i)*pco_->dx2v(j)),

              ( (p(IPR,k,j,i-1)/p(IDN,k,j,i-1)) - (p(IPR,k,j-1,i-1)/p(IDN,k,j-1,i-1)) )
              /(pco_->h2f(i)*pco_->dx2v(j-1)));
        }
        if (f3) { // limited dT/dz
          dTdz_t = LimitedTransverse(
              ( (p(IPR,k+1,j,i)/p(IDN,k+1,j,i)) - (p(IPR,k,j,i)/p(IDN,k,j,i)) )
              /(pco_->dx3v(k)*pco_->h31f(i)*pco_->h32v(j)),

              ( (p(IPR,k,j,i)/p(IDN,k,j,i)) - (p(IPR,k-1,j,i)/p(IDN,k-1,j,i)) )
              /(pco_->dx3v(k-1)*pco_->h31f(i)*pco_->h32v(j)),

              ((p(IPR,k+1,j,i-1)/p(IDN,k+1,j,i-1)) - (p(IPR,k,j,i-1)/p(IDN,k,j,i-1)))
              /(pco_->dx3v(k)*pco_->h31f(i)*pco_->h32v(j)),

              ( (p(IPR,k,j,i-1)/p(IDN,k,j,i-1)) - (p(IPR,k-1,j,i-1)/p(IDN,k-1,j,i-1)) )
              /(pco_->dx3v(k-1)*pco_->h31f(i)*pco_->h32v(j)));
        }
      } // end if lim_c

      Bx = pmb_->pfield->b.x1f(k,j,i);
      By = 0.5*(pmb_->pfield->bcc(IB2, k,j,i) + pmb_->pfield->bcc(IB2, k,j,i-1));
      Bz = 0.5*(pmb_->pfield->bcc(IB3, k,j,i) + pmb_->pfield->bcc(IB3, k,j,i-1));
      B = std::sqrt(Bx*Bx + By*By + Bz*Bz);

      if (B > TINY_NUMBER) {
        bx = Bx/B; by = By/B; bz = Bz/B;
        b_dot_gradT = bx*dTdx + by*dTdy_t + bz*dTdz_t;
      } else {
        bx = 0.0; by = 0.0; bz = 0.0; b_dot_gradT = 0.0;
      }

      HII = (HydroDiffusion::kappa_aniso_neutral>0.0) ?
      0.5*(HydroDiffusion::ion_frac(k,j,i)+HydroDiffusion::ion_frac(k,j,i-1)) : 1.0;

      if (use_neutral) { // Neutral part is isotropic
      kappa_neutral =  0.5*(kappa(DiffProcess::kappa_neutral,k,j,i)
                       + kappa(DiffProcess::kappa_neutral,k,j,i-1));
      iso_flux = -((1-HII)*kappa_neutral*denf*dTdx);
    }
      // BS1: p:249 eq:4.33 - x direction
      if (use_para) {
        kappa_para = 0.5*(kappa(DiffProcess::kappa_para,k,j,i)
                     + kappa(DiffProcess::kappa_para,k,j,i-1));
        para_flux = -(HII*kappa_para*denf*bx*b_dot_gradT);
      }
      if (use_perp) {
        kappa_perp = 0.5*(kappa(DiffProcess::kappa_perp,k,j,i)
                   + kappa(DiffProcess::kappa_perp,k,j,i-1));
        perp_flux = -(HII*kappa_perp*denf*(dTdx-bx*b_dot_gradT));
      } // saturate the total ion flux MG1: A.3
      if (use_sat && (use_para || use_perp)) {
        const Real grad2 = dTdx*dTdx + dTdy_t*dTdy_t + dTdz_t*dTdz_t;
        const Real q_para = use_para ? kappa_para*denf*std::abs(b_dot_gradT) : 0.0;
        const Real q_perp = use_perp ? kappa_perp*denf
                            *std::sqrt(std::max(grad2 - b_dot_gradT*b_dot_gradT, 0.0))
                            : 0.0;
        const int iup = (para_flux + perp_flux > 0.0) ? i-1 : i;  // hot side
        const Real ciso2 = p(IPR,k,j,iup)/p(IDN,k,j,iup);
        const Real qsat = 1.5*p(IDN,k,j,iup)*ciso2*std::sqrt(ciso2);
        const Real f_para = qsat/(qsat + q_para);
        const Real f_perp = qsat/(qsat + q_perp);
        para_flux *= f_para;
        perp_flux *= f_perp;
        sat_x1(k,j,i) = f_perp; // f_perp, reused by the Ettingshausen flux
      }
      if (use_wedge) {
        kappa_wedge = 0.5*(kappa(DiffProcess::kappa_wedge,k,j,i)
                      + kappa(DiffProcess::kappa_wedge,k,j,i-1));
        wedge_flux = -(HII*kappa_wedge*denf*(by*dTdz - bz*dTdy));
      }
      if(use_neutral || use_para || use_perp) {
        x1flux(k,j,i) += iso_flux + para_flux + perp_flux;
      }
      if(use_wedge) {
        x1flux_disp(k,j,i) += wedge_flux;
        if(!STS_ENABLED) {
          x1flux(k,j,i) += wedge_flux;
          } // end if STS
        } // end if wedge
      } // i loop
    } // j loop
  } // k loop

  // j-direction
  il = is, iu = ie, kl = ks, ku = ke;
  if (MAGNETIC_FIELDS_ENABLED) {
    if (!f3) // 2D
      il = is - 1, iu = ie + 1, kl = ks, ku = ke;
    else // 3D
      il = is - 1, iu = ie + 1, kl = ks - 1, ku = ke + 1;
  }
  if (f2) { // 2D or 3D
    for (int k=kl; k<=ku; ++k) {
      for (int j=js; j<=je+1; ++j) {
        #pragma omp simd
        for (int i=il; i<=iu; ++i) {
      iso_flux = 0.0; para_flux = 0.0; perp_flux = 0.0; wedge_flux = 0.0;

      denf = 0.5*(p(IDN,k,j,i) + p(IDN,k,j-1,i));
      // transverse dT/dx at y face
      dTdx = 0.5*((p(IPR,k,j,i+1)/p(IDN,k,j,i+1) + p(IPR,k,j-1,i+1)/p(IDN,k,j-1,i+1))
                - (p(IPR,k,j,i-1)/p(IDN,k,j,i-1)   + p(IPR,k,j-1,i-1)/p(IDN,k,j-1,i-1)))
                / (pco_->dx1v(i-1) + pco_->dx1v(i));
      // normal dT/dy at y
      dTdy = (p(IPR,k,j,i)/p(IDN,k,j,i) - p(IPR,k,j-1,i)/p(IDN,k,j-1,i))
              / (pco_->h2v(i) * pco_->dx2v(j-1));
      // transverse dT/dz at y face
      if (f3) {
      dTdz = 0.5*((p(IPR,k+1,j,i)/p(IDN,k+1,j,i)   + p(IPR,k+1,j-1,i)/p(IDN,k+1,j-1,i))
                - (p(IPR,k-1,j,i)/p(IDN,k-1,j,i)   + p(IPR,k-1,j-1,i)/p(IDN,k-1,j-1,i)))
                / (pco_->dx3v(k-1) + pco_->dx3v(k))/ pco_->h31v(i) / pco_->h32f(j);
      } else {
        dTdz=0;
      }

      // SH1: eq:17 & 18 limited transverse gradients for para/perp
      dTdx_t = dTdx; dTdz_t = dTdz;
      if(lim_c){
        // limited dT/dx
        dTdx_t = LimitedTransverse(
            ( (p(IPR,k,j,i+1)/p(IDN,k,j,i+1)) - (p(IPR,k,j,i)/p(IDN,k,j,i)) )
            /pco_->dx1v(i),

            ( (p(IPR,k,j,i)/p(IDN,k,j,i)) - (p(IPR,k,j,i-1)/p(IDN,k,j,i-1)) )
            /pco_->dx1v(i-1),

            ( (p(IPR,k,j-1,i+1)/p(IDN,k,j-1,i+1)) - (p(IPR,k,j-1,i)/p(IDN,k,j-1,i)) )
            /pco_->dx1v(i),

            ( (p(IPR,k,j-1,i)/p(IDN,k,j-1,i)) - (p(IPR,k,j-1,i-1)/p(IDN,k,j-1,i-1)) )
            /pco_->dx1v(i-1));
        if (f3) { // limited dT/dz
          dTdz_t = LimitedTransverse(
              ( (p(IPR,k+1,j,i)/p(IDN,k+1,j,i)) - (p(IPR,k,j,i)/p(IDN,k,j,i)) )
              /(pco_->dx3v(k)*pco_->h31v(i)*pco_->h32f(j)),

              ( (p(IPR,k,j,i)/p(IDN,k,j,i)) - (p(IPR,k-1,j,i)/p(IDN,k-1,j,i)) )
              /(pco_->dx3v(k-1)*pco_->h31v(i)*pco_->h32f(j)),

              ( (p(IPR,k+1,j-1,i)/p(IDN,k+1,j-1,i)) - (p(IPR,k,j-1,i)/p(IDN,k,j-1,i)) )
              /(pco_->dx3v(k)*pco_->h31v(i)*pco_->h32f(j)),

              ( (p(IPR,k,j-1,i)/p(IDN,k,j-1,i)) - (p(IPR,k-1,j-1,i)/p(IDN,k-1,j-1,i)) )
              /(pco_->dx3v(k-1)*pco_->h31v(i)*pco_->h32f(j)));
        }
      } // end if lim_c

      Bx = 0.5*(pmb_->pfield->bcc(IB1, k,j,i) + pmb_->pfield->bcc(IB1, k,j-1,i));
      By =  pmb_->pfield->b.x2f(k,j,i);
      Bz = 0.5*(pmb_->pfield->bcc(IB3, k,j,i) + pmb_->pfield->bcc(IB3, k,j-1,i));
      B = std::sqrt(Bx*Bx + By*By + Bz*Bz);

      if (B > TINY_NUMBER) {
        bx = Bx/B; by = By/B; bz = Bz/B;
        b_dot_gradT = bx*dTdx_t + by*dTdy + bz*dTdz_t;
      } else {
        bx = 0.0; by = 0.0; bz = 0.0; b_dot_gradT = 0.0;
      }

      HII = (HydroDiffusion::kappa_aniso_neutral>0.0)
      ? 0.5*(HydroDiffusion::ion_frac(k,j,i) + HydroDiffusion::ion_frac(k,j-1,i)) : 1.0;

       if (use_neutral) { // Neutral part is isotropic
        kappa_neutral =  0.5*(kappa(DiffProcess::kappa_neutral,k,j,i)
                         + kappa(DiffProcess::kappa_neutral,k,j-1,i));
        iso_flux = -(1-HII)*(kappa_neutral*denf*dTdy);
      }

        // BS1: p:249 eq:4.33 - y direction
        if (use_para) {
        kappa_para = 0.5*(kappa(DiffProcess::kappa_para,k,j,i)
                     + kappa(DiffProcess::kappa_para,k,j-1,i));
        para_flux = -(HII*kappa_para*denf*by*b_dot_gradT);
        }
        if (use_perp) {
          kappa_perp = 0.5*(kappa(DiffProcess::kappa_perp,k,j,i)
                      + kappa(DiffProcess::kappa_perp,k,j-1,i));
          perp_flux = -(HII*kappa_perp*denf*(dTdy-by*b_dot_gradT));
        }  // saturate the total ion flux MG1: A.3
        if (use_sat && (use_para || use_perp)) {
          const Real grad2 = dTdx_t*dTdx_t + dTdy*dTdy + dTdz_t*dTdz_t;
          const Real q_para = use_para ? kappa_para*denf*std::abs(b_dot_gradT) : 0.0;
          const Real q_perp = use_perp ? kappa_perp*denf
                              *std::sqrt(std::max(grad2 - b_dot_gradT*b_dot_gradT, 0.0))
                              : 0.0;
          const int jup = (para_flux + perp_flux > 0.0) ? j-1 : j;  // hot side
          const Real ciso2 = p(IPR,k,jup,i)/p(IDN,k,jup,i);
          const Real qsat = 1.5*p(IDN,k,jup,i)*ciso2*std::sqrt(ciso2);
          const Real f_para = qsat/(qsat + q_para);
          const Real f_perp = qsat/(qsat + q_perp);
          para_flux *= f_para;
          perp_flux *= f_perp;
          sat_x2(k,j,i) = f_perp; // f_perp, reused by the Ettingshausen flux
        }
        if (use_wedge) {
          kappa_wedge = 0.5*(kappa(DiffProcess::kappa_wedge,k,j,i)
                        + kappa(DiffProcess::kappa_wedge,k,j-1,i));
          wedge_flux = -(HII*kappa_wedge*denf*(bz*dTdx - bx*dTdz));
        }
        if(use_neutral || use_para || use_perp) {
          x2flux(k,j,i) += iso_flux + para_flux + perp_flux;
        }
        if(use_wedge) {
          x2flux_disp(k,j,i) += wedge_flux;
          if(!STS_ENABLED) {
            x2flux(k,j,i) += wedge_flux;
          } // end if STS
        } // end if wedge
      } // i loop
    } // j loop
  } // k loop
} // zero flux for 1D

  // k-direction
  il = is, iu = ie, jl = js, ju = je;
  if (MAGNETIC_FIELDS_ENABLED) {
    if (f2) // 2D or 3D
      il = is - 1, iu = ie + 1, jl = js - 1, ju = je + 1;
    else // 1D
      il = is - 1, iu = ie + 1;
  }
  if (f3) { // 3D
    for (int k=ks; k<=ke+1; ++k) {
      for (int j=jl; j<=ju; ++j) {
        #pragma omp simd
        for (int i=il; i<=iu; ++i) {
        iso_flux = 0.0;
        para_flux = 0.0;
        perp_flux = 0.0;
        wedge_flux = 0.0;

        denf = 0.5*(p(IDN,k,j,i) + p(IDN,k-1,j,i));

        // transverse dT/dx at z face
        dTdx = 0.5*((p(IPR,k,j,i+1)/p(IDN,k,j,i+1) + p(IPR,k-1,j,i+1)/p(IDN,k-1,j,i+1))
                  - (p(IPR,k,j,i-1)/p(IDN,k,j,i-1) + p(IPR,k-1,j,i-1)/p(IDN,k-1,j,i-1)))
                  / (pco_->dx1v(i-1) + pco_->dx1v(i));
        if (f2) {
        // transverse dT/dy at z face
        dTdy = 0.5*((p(IPR,k,j+1,i)/p(IDN,k,j+1,i)
                  + p(IPR,k-1,j+1,i)/p(IDN,k-1,j+1,i))
                  - (p(IPR,k,j-1,i)/p(IDN,k,j-1,i) + p(IPR,k-1,j-1,i)/p(IDN,k-1,j-1,i)))
                  / pco_->h2v(i)/ (pco_->dx2v(j-1) + pco_->dx2v(j));
        } else {
          dTdy=0;
        }
        // normal dT/dz
        dTdz = (p(IPR,k,j,i)/p(IDN,k,j,i) - p(IPR,k-1,j,i)/p(IDN,k-1,j,i))
              /(pco_->dx3v(k-1) * pco_->h31v(i) * pco_->h32v(j));

         // SH1: eq:17 & 18 limited transverse gradients for para/perp
         dTdx_t = dTdx; dTdy_t = dTdy;
         if(lim_c){
          dTdx_t = LimitedTransverse( // limited dT/dx
              ( (p(IPR,k,j,i+1)/p(IDN,k,j,i+1)) - (p(IPR,k,j,i)/p(IDN,k,j,i)) )
              /pco_->dx1v(i),

              ( (p(IPR,k,j,i)/p(IDN,k,j,i)) - (p(IPR,k,j,i-1)/p(IDN,k,j,i-1)) )
              /pco_->dx1v(i-1),

              ( (p(IPR,k-1,j,i+1)/p(IDN,k-1,j,i+1)) - (p(IPR,k-1,j,i)/p(IDN,k-1,j,i)) )
              /pco_->dx1v(i),

              ( (p(IPR,k-1,j,i)/p(IDN,k-1,j,i)) - (p(IPR,k-1,j,i-1)/p(IDN,k-1,j,i-1)) )
              /pco_->dx1v(i-1));
          if (f2) { // limited dT/dy
            dTdy_t = LimitedTransverse(
                ( (p(IPR,k,j+1,i)/p(IDN,k,j+1,i)) - (p(IPR,k,j,i)/p(IDN,k,j,i)) )
                /(pco_->h2v(i)*pco_->dx2v(j)),

                ( (p(IPR,k,j,i)/p(IDN,k,j,i)) - (p(IPR,k,j-1,i)/p(IDN,k,j-1,i)) )
                /(pco_->h2v(i)*pco_->dx2v(j-1)),

                ( (p(IPR,k-1,j+1,i)/p(IDN,k-1,j+1,i)) - (p(IPR,k-1,j,i)/p(IDN,k-1,j,i)) )
                /(pco_->h2v(i)*pco_->dx2v(j)),

                ( (p(IPR,k-1,j,i)/p(IDN,k-1,j,i)) - (p(IPR,k-1,j-1,i)/p(IDN,k-1,j-1,i)) )
                /(pco_->h2v(i)*pco_->dx2v(j-1)));
          }
        } // end if lim_c

        Bx = 0.5*(pmb_->pfield->bcc(IB1, k,j,i) + pmb_->pfield->bcc(IB1, k-1,j,i));
        By = 0.5*(pmb_->pfield->bcc(IB2, k,j,i) + pmb_->pfield->bcc(IB2, k-1,j,i));
        Bz = pmb_->pfield->b.x3f(k,j,i);
        B = std::sqrt(Bx*Bx + By*By + Bz*Bz);

        if (B > TINY_NUMBER) {
          bx = Bx/B; by = By/B; bz = Bz/B;
          b_dot_gradT = bx*dTdx_t + by*dTdy_t + bz*dTdz;
        } else {
          bx = 0.0; by = 0.0; bz = 0.0; b_dot_gradT = 0.0;
        }

        HII = (HydroDiffusion::kappa_aniso_neutral>0.0)
        ? 0.5*(HydroDiffusion::ion_frac(k,j,i) + HydroDiffusion::ion_frac(k-1,j,i)) : 1.0;

        if (use_neutral) { // Neutral part is isotropic
          kappa_neutral = 0.5*(kappa(DiffProcess::kappa_neutral,k,j,i)
                          + kappa(DiffProcess::kappa_neutral,k-1,j,i));
          iso_flux = -(1-HII)*(kappa_neutral*denf*dTdz);
        }

        // BS1: p:249 eq:4.33 - z direction
        if (use_para) {
          kappa_para = 0.5*(kappa(DiffProcess::kappa_para,k,j,i)
                        + kappa(DiffProcess::kappa_para,k-1,j,i));
          para_flux = -(HII*kappa_para*denf*bz*b_dot_gradT);
        }
        if (use_perp) {
          kappa_perp = 0.5*(kappa(DiffProcess::kappa_perp,k,j,i)
                      + kappa(DiffProcess::kappa_perp,k-1,j,i));
          perp_flux = -(HII*kappa_perp*denf*(dTdz-bz*b_dot_gradT));
        }  // saturate the total ion flux (Mignone+12 A3)
        if (use_sat && (use_para || use_perp)) {
          const Real grad2 = dTdx_t*dTdx_t + dTdy_t*dTdy_t + dTdz*dTdz;
          const Real q_para = use_para ? kappa_para*denf*std::abs(b_dot_gradT) : 0.0;
          const Real q_perp = use_perp ? kappa_perp*denf
                              *std::sqrt(std::max(grad2 - b_dot_gradT*b_dot_gradT, 0.0))
                              : 0.0;
          const int kup = (para_flux + perp_flux > 0.0) ? k-1 : k;  // hot side
          const Real ciso2 = p(IPR,kup,j,i)/p(IDN,kup,j,i);
          const Real qsat = 1.5*p(IDN,kup,j,i)*ciso2*std::sqrt(ciso2);
          const Real f_para = qsat/(qsat + q_para);
          const Real f_perp = qsat/(qsat + q_perp);
          para_flux *= f_para;
          perp_flux *= f_perp;
          sat_x3(k,j,i) = f_perp; // f_perp, reused by the Ettingshausen flux
        }
        if (use_wedge) {
          kappa_wedge = 0.5*(kappa(DiffProcess::kappa_wedge,k,j,i)
                        + kappa(DiffProcess::kappa_wedge,k-1,j,i));
          wedge_flux = - (HII*kappa_wedge*denf*(bx*dTdy - by*dTdx));
        }
        if(use_neutral || use_para || use_perp) {
          x3flux(k,j,i) += iso_flux + para_flux + perp_flux;
        }
        if(use_wedge) {
          x3flux_disp(k,j,i) += wedge_flux;
          if(!STS_ENABLED) {
            x3flux(k,j,i) += wedge_flux;
          } // end if STS
        } // end if wedge
      } // i loop
    } // j loop
  } // k loop
} // zero flux for 1D/2D
  return;
}

//----------------------------------------------------------------------------------------
//! \fn ConstConduction
//! \brief Constant thermal conduction coefficients

void ConstConduction(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &prim,
                     const AthenaArray<Real> &bcc,
                     int is, int ie, int js, int je, int ks, int ke) {
  if (phdif->kappa_iso > 0.0) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          phdif->kappa(HydroDiffusion::DiffProcess::iso,k,j,i) = phdif->kappa_iso;
        }
      }
    }
  }
  if (phdif->kappa_aniso_neutral > 0.0 || phdif->ion_frac_const > 0.0) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          phdif->kappa(HydroDiffusion::DiffProcess::kappa_neutral,k,j,i)
                      = phdif->kappa_aniso_neutral;
          phdif->ion_frac(k,j,i) = phdif->ion_frac_const;
        }
      }
    }
  }
  if (phdif->kappa_aniso_para > 0.0) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          phdif->kappa(HydroDiffusion::DiffProcess::kappa_para,k,j,i)
                      = phdif->kappa_aniso_para;
        }
      }
    }
  }
  if (phdif->kappa_aniso_perp > 0.0) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          phdif->kappa(HydroDiffusion::DiffProcess::kappa_perp,k,j,i)
                       = phdif->kappa_aniso_perp;
        }
      }
    }
  }

  if (phdif->kappa_aniso_wedge > 0.0) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          phdif->kappa(HydroDiffusion::DiffProcess::kappa_wedge,k,j,i)
                       = phdif->kappa_aniso_wedge;
        }
      }
    }
  }

  return;
}
