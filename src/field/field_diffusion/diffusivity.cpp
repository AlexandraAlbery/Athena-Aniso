//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file diffusivity.cpp
//! \brief implements functions that are related to non-ideal MHD diffusivities

// C headers

// C++ headers
#include <iostream>   // cout, endl

// Athena++ headers
#include "../../athena.hpp"                                // Real
#include "../../athena_arrays.hpp"                         // AthenaArray
#include "../../coordinates/coordinates.hpp"               // Coordinates
#include "../../hydro/hydro.hpp"                           // Fluid
#include "../../hydro/hydro_diffusion/hydro_diffusion.hpp" // Ionisation fraction for TE
#include "../../mesh/mesh.hpp"                             // MeshBlock
#include "../../parameter_input.hpp"                       // ParameterInput
#include "../field.hpp"                                    // Field
#include "field_diffusion.hpp"

//----------------------------------------------------------------------------------------
//! \fn void ConstDiffusivity
//! \brief Magnetic diffusivity from constant coefficients

void ConstDiffusivity(FieldDiffusion *pfdif, MeshBlock *pmb, const AthenaArray<Real> &w,
                      const AthenaArray<Real> &bmag,
                      const int is, const int ie, const int js, const int je,
                      const int ks, const int ke) {
  if (pfdif->eta_ohm > 0.0) { // Ohmic resistivity is turned on
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
#pragma omp simd
        for (int i=is; i<=ie; i++)
          pfdif->etaB(FieldDiffusion::DiffProcess::ohmic, k,j,i) =
              pfdif->eta_ohm;
      }
    }
  }
  if (pfdif->eta_hall != 0.0) { // Hall diffusivity is turned on
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
#pragma omp simd
        for (int i=is; i<=ie; i++)
          pfdif->etaB(FieldDiffusion::DiffProcess::hall, k,j,i) =
              pfdif->eta_hall*bmag(k,j,i)/w(IDN,k,j,i);
      }
    }
  }
  if (pfdif->eta_ad > 0.0) { // ambipolar diffusivity is turned on
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
#pragma omp simd
        for (int i=is; i<=ie; i++)
          pfdif->etaB(FieldDiffusion::DiffProcess::ambipolar, k,j,i) =
              pfdif->eta_ad*SQR(bmag(k,j,i));
      }
    }
  }

  if (pmb->phydro->hdif.ion_frac_const > 0.0) { // ionisation fraction for TE
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
#pragma omp simd
        for (int i=is; i<=ie; i++)
          pmb->phydro->hdif.ion_frac(k,j,i) =
          pmb->phydro->hdif.ion_frac_const;
      }
    }
  }

  if (pfdif->kappa_aniso_TE_para > 0.0) { // parallel TE
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
#pragma omp simd
        for (int i=is; i<=ie; i++)
          pfdif->thermKappa(FieldDiffusion::ThermElecProcess::kappa_TE_para, k,j,i) =
              pfdif->kappa_aniso_TE_para;
      }
    }
  }
  if (pfdif->kappa_aniso_TE_perp > 0.0) { // perpedicular TE
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
#pragma omp simd
        for (int i=is; i<=ie; i++)
          pfdif->thermKappa(FieldDiffusion::ThermElecProcess::kappa_TE_perp, k,j,i) =
              pfdif->kappa_aniso_TE_perp;
      }
    }
  }

  if (pfdif->kappa_aniso_TE_wedge > 0.0) { // wedge TE
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
#pragma omp simd
        for (int i=is; i<=ie; i++)
          pfdif->thermKappa(FieldDiffusion::ThermElecProcess::kappa_TE_wedge, k,j,i) =
              pfdif->kappa_aniso_TE_wedge;
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void FieldDiffusion::CalcEdgeTemperature
//! \brief Calculate temperature at cell edges for thermoelectric force

void FieldDiffusion::CalcEdgeTemperature(const AthenaArray<Real> &w,
                                         AthenaArray<Real> &Tcc, AthenaArray<Real> &Te1,
                                         AthenaArray<Real> &Te2, AthenaArray<Real> &Te3) {
  Te1.ZeroClear();
  Te2.ZeroClear();
  Te3.ZeroClear();
  MeshBlock *pmb = pmy_block;
  Coordinates *pco = pmb->pcoord;

  int ext2 = 0,ext3 = 0;
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  const bool nx2 = (pmb->block_size.nx2 > 1);
  const bool nx3 = (pmb->block_size.nx3 > 1);

  AthenaArray<Real> T_;

  if (!nx2 && !nx3) {
    for (int i=is-NGHOST; i<=ie+NGHOST+1; i++) {
      Tcc(ks,js,i) = w(IPR, ks, js, i)/w(IDN, ks, js, i);
    }
  } else if (!nx3) {
    for (int j=js - NGHOST; j<=je+NGHOST; j++) {
      for (int i = is - NGHOST; i<=ie+NGHOST; i++) {
        Tcc(ks,j,i) = w(IPR, ks, j, i)/w(IDN, ks, j, i);
      }
    }
  } else {
    for(int k = ks - NGHOST; k<=ke+NGHOST; k++) {
      for (int j=js - NGHOST; j<=je + NGHOST; j++) {
        for (int i = is - NGHOST; i<=ie + NGHOST; i++) {
          Tcc(k,j,i) = w(IPR, k, j, i)/w(IDN, k, j, i);
        }
      }
    }
  }

  if (!nx2 && !nx3) { // 1D T edge
    for (int i=is; i<=ie+2; i++) {
    Te2(ks,js,i) = 0.5*(Tcc(ks,js,i) + Tcc(ks,js,i-1));
    Te3(ks,js,i) = Te2(ks,js,i);
    }
  } else if (!nx3) { // 2D T edge
    for (int j=js; j<=je+2; j++) {
      for (int i = is; i<=ie+2; i++) {
        Te1(ks,j,i) = 0.5*(Tcc(ks,j-1,i) + Tcc(ks,j,i));
        Te2(ks,j,i) = 0.5*(Tcc(ks,j,i-1) + Tcc(ks, j, i));
        Te3(ks,j,i) = 0.25*(Tcc(ks,j,i) + Tcc(ks,j-1,i)
                    + Tcc(ks,j,i-1) + Tcc(ks,j-1,i-1));
      }
    }
  } else { // 3D T edge
    for (int k = ks; k<=ke+2; k++) {
      for (int j=js; j<=je+2; j++) {
        for (int i = is; i<=ie+2; i++) {
          Te1(k,j,i) = 0.25*(Tcc(k,j,i) + Tcc(k-1,j,i) + Tcc(k,j-1,i) + Tcc(k-1,j-1,i));
          Te2(k,j,i) = 0.25*(Tcc(k,j,i) + Tcc(k-1,j,i) + Tcc(k,j,i-1) + Tcc(k-1,j,i-1));
          Te3(k,j,i) = 0.25*(Tcc(k,j,i) + Tcc(k,j-1,i) + Tcc(k,j,i-1) + Tcc(k,j-1,i-1));
        }
      }
    }
  }
  return;
}



//----------------------------------------------------------------------------------------
//! \fn void FieldDiffusion::CalcCurrent
//! \brief Calculate current density

void FieldDiffusion::CalcCurrent(FaceField &b) {
  MeshBlock *pmb = pmy_block;
  Coordinates *pco = pmb->pcoord;

  int ext2 = 0,ext3 = 0;
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;
  if (pmb->block_size.nx2 > 1) ext2 = 1;
  if (pmb->block_size.nx3 > 1) ext3 = 1;

  AthenaArray<Real> &J1 = jedge_.x1e, &J2 = jedge_.x2e, &J3 = jedge_.x3e,
                   &b1i = b.x1f, &b2i = b.x2f, &b3i = b.x3f;

  AthenaArray<Real> &area = face_area_, &len = edge_length_, &len_m1 = edge_length_m1_;

  for (int k=ks-ext3; k<=ke+2*ext3; ++k) {
    for (int j=js-2*ext2; j<=je+2*ext2; ++j) {
      pco->VolCenterFace2Area(k-ext3, j, is-2, ie+1, area);
      pco->VolCenter3Length(k-ext3, j, is-2, ie+2, len);
#pragma omp simd
      for (int i=is-1; i<=ie+2; ++i) {
        J2(k,j,i) = -(len(i)*b3i(k,j,i) - len(i-1)*b3i(k,j,i-1))/area(i-1);
      }
    }
  }

  for (int k=ks-2*ext3; k<=ke+2*ext3; ++k) {
    for (int j=js-ext2; j<=je+2*ext2; ++j) {
      pco->VolCenterFace3Area(k, j-ext2, is-2, ie+1, area);
      pco->VolCenter2Length(k, j-ext2, is-2, ie+2, len);
#pragma omp simd
      for (int i=is-1; i<=ie+2; ++i) {
        J3(k,j,i) = (len(i)*b2i(k,j,i) - len(i-1)*b2i(k,j,i-1))/area(i-1);
      }
    }
  }

  if (pmb->block_size.nx2 > 1) {
    for (int k=ks-ext3; k<=ke+2*ext3; ++k) {
      for (int j=js-1; j<=je+2; ++j) {
        pco->VolCenterFace1Area(k-ext3, j-1, is-2, ie+2, area);
        pco->VolCenter3Length(k-ext3, j-1, is-2, ie+2, len_m1);
        pco->VolCenter3Length(k-ext3, j  , is-2, ie+2, len);
#pragma omp simd
        for (int i=is-2; i<=ie+2; ++i) {
          J1(k,j,i) = (len(i)*b3i(k,j,i) - len_m1(i)*b3i(k,j-1,i))/area(i);
        }
      }
    }

    for (int k=ks-2*ext3; k<=ke+2*ext3; ++k) {
      for (int j=js-1; j<=je+2; ++j) {
        pco->VolCenterFace3Area(k, j-1, is-2, ie+1, area);
        pco->VolCenter1Length(k, j-1, is-2, ie+1, len_m1);
        pco->VolCenter1Length(k, j  , is-2, ie+1, len);
#pragma omp simd
        for (int i=is-1; i<=ie+2; ++i) {
          J3(k,j,i) -= (len(i-1)*b1i(k,j,i) - len_m1(i-1)*b1i(k,j-1,i))/area(i-1);
        }
      }
    }
  }

  if (pmb->block_size.nx3 > 1) {
    for (int k=ks-1; k<=ke+2; ++k) {
      for (int j=js-1; j<=je+2; ++j) {
        pco->VolCenterFace1Area(k-1, j-1, is-2, ie+2, area);
        pco->VolCenter2Length(k-1, j-1, is-2, ie+2, len_m1);
        pco->VolCenter2Length(k  , j-1, is-2, ie+2, len);
#pragma omp simd
        for (int i=is-2; i<=ie+2; ++i) {
          J1(k,j,i) -= (len(i)*b2i(k,j,i) - len_m1(i)*b2i(k-1,j,i))/area(i);
        }
      }
    }

    for (int k=ks-1; k<=ke+2; ++k) {
      for (int j=js-2; j<=je+2; ++j) {
        pco->VolCenterFace2Area(k-1, j, is-2, ie+1, area);
        pco->VolCenter1Length(k-1, j, is-2, ie+1, len_m1);
        pco->VolCenter1Length(k  , j, is-2, ie+1, len);
#pragma omp simd
        for (int i=is-1; i<=ie+2; ++i) {
          J2(k,j,i) += (len(i-1)*b1i(k,j,i) - len_m1(i-1)*b1i(k-1,j,i))/area(i-1);
        }
      }
    }
  }

  // build centre and face J for thermoelectric conduction
  if (kappa_aniso_TE_para > 0.0 || kappa_aniso_TE_perp > 0.0
      || kappa_aniso_TE_wedge > 0.0) {
    jcc.ZeroClear();
    jfx.ZeroClear();
    jfy.ZeroClear();
    jfz.ZeroClear();

    // cell-centered J (i,j,k)
    for (int k=ks-ext3; k<=ke+ext3; ++k) {
      for (int j=js-ext2; j<=je+ext2; ++j) {
      #pragma omp simd
        for (int i=is-1; i<=ie+1; ++i) {
          // Jx at cc
          jcc(IB1,k,j,i) = 0.25*(J1(k,j,i) + J1(k+ext3,j,i) + J1(k,j+ext2,i)
                          + J1(k+ext3,j+ext2,i));
          // Jy at cc
          jcc(IB2,k,j,i) = 0.25*(J2(k,j,i) + J2(k+ext3,j,i) + J2(k,j,i+1)
                         + J2(k+ext3,j,i+1));
          // Jz at cc
          jcc(IB3,k,j,i) = 0.25*(J3(k,j,i) + J3(k,j+ext2,i) + J3(k,j,i+1)
                         + J3(k,j+ext2,i+1));
        }
      }
    }

    // x-face J (i-1/2, j, k)
    for (int k=ks-ext3; k<=ke+ext3; ++k) {
      for (int j=js-ext2; j<=je+ext2; ++j) {
    #pragma omp simd
        for (int i=is; i<=ie+1; ++i) {
          jfx(IB1,k,j,i) = 0.5*(jcc(IB1,k,j,i-1) + jcc(IB1,k,j,i));
          jfx(IB2,k,j,i) = 0.5*(jcc(IB2,k,j,i-1) + jcc(IB2,k,j,i));
          jfx(IB3,k,j,i) = 0.5*(jcc(IB3,k,j,i-1) + jcc(IB3,k,j,i));
        }
      }
    }

    // y-face J (i,j-1/2,k)
    if (pmb->block_size.nx2 > 1) {
      for (int k=ks-ext3; k<=ke+ext3; ++k) {
        for (int j=js; j<=je+1; ++j) {
    #pragma omp simd
          for (int i=is-1; i<=ie+1; ++i) {
            jfy(IB1,k,j,i) = 0.5*(jcc(IB1,k,j-1,i) + jcc(IB1,k,j,i));
            jfy(IB2,k,j,i) = 0.5*(jcc(IB2,k,j-1,i) + jcc(IB2,k,j,i));
            jfy(IB3,k,j,i) = 0.5*(jcc(IB3,k,j-1,i) + jcc(IB3,k,j,i));
          }
        }
      }
    }

    // z-face J (i,j,k-1/2)
    if (pmb->block_size.nx3 > 1) {
      for (int k=ks; k<=ke+1; ++k) {
        for (int j=js-ext2; j<=je+ext2; ++j) {
    #pragma omp simd
          for (int i=is-1; i<=ie+1; ++i) {
            jfz(IB1,k,j,i) = 0.5*(jcc(IB1,k-1,j,i) + jcc(IB1,k,j,i));
            jfz(IB2,k,j,i) = 0.5*(jcc(IB2,k-1,j,i) + jcc(IB2,k,j,i));
            jfz(IB3,k,j,i) = 0.5*(jcc(IB3,k-1,j,i) + jcc(IB3,k,j,i));
          }
        }
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void FieldDiffusion::OhmicEMF
//! \brief EMF from Ohmic resistivity

void FieldDiffusion::OhmicEMF(const FaceField &b, const AthenaArray<Real> &bc,
                              EdgeField &e) {
  MeshBlock *pmb = pmy_block;
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  AthenaArray<Real> &e1 = e.x1e, &e2 = e.x2e, &e3 = e.x3e,
                    &J1 = jedge_.x1e, &J2 = jedge_.x2e, &J3 = jedge_.x3e;

  // 1D update:
  if (pmb->block_size.nx2 == 1) {
    for (int i=is; i<=ie+1; ++i) {
      Real eta_O = 0.5*(etaB(ohmic,ks,js,i-1)+etaB(ohmic,ks,js,i));

      e2(ks  ,js  ,i) += eta_O * J2(ks,js,i);
      e2(ke+1,js  ,i)  = e2(ks,js,i);
      e3(ks  ,js  ,i) += eta_O * J3(ks,js,i);
      e3(ks  ,je+1,i)  = e3(ks,js,i);
    }
    return;
  }

  // 2D update:
  if (pmb->block_size.nx3 == 1) {
    for (int j=js; j<=je+1; ++j) {
#pragma omp simd
      for (int i=is; i<=ie+1; ++i) {
        Real eta_O = 0.5*(etaB(ohmic,ks,j,i) + etaB(ohmic,ks,j-1,i));
        e1(ks  ,j,i) += eta_O * J1(ks,j,i);
        e1(ke+1,j,i)  = e1(ks,j,i);

        eta_O = 0.5*(etaB(ohmic,ks,j,i) + etaB(ohmic,ks,j,i-1));
        e2(ks  ,j,i) += eta_O * J2(ks,j,i);
        e2(ke+1,j,i)  = e2(ks,j,i);

        eta_O = 0.25*(etaB(ohmic,ks,j  ,i) + etaB(ohmic,ks,j  ,i-1)
                      + etaB(ohmic,ks,j-1,i) + etaB(ohmic,ks,j-1,i-1));
        e3(ks,  j,i) += eta_O * J3(ks,j,i);
      }
    }
    return;
  }

  // 3D update:
  for (int k=ks; k<=ke+1; ++k) {
    for (int j=js; j<=je+1; ++j) {
#pragma omp simd
      for (int i=is; i<=ie+1; ++i) {
        Real eta_O = 0.25*(etaB(ohmic,k  ,j,i) + etaB(ohmic,k  ,j-1,i)
                           + etaB(ohmic,k-1,j,i) + etaB(ohmic,k-1,j-1,i));
        e1(k,j,i) += eta_O * J1(k,j,i);

        eta_O = 0.25*(etaB(ohmic,k  ,j,i) + etaB(ohmic,k  ,j,i-1)
                      + etaB(ohmic,k-1,j,i) + etaB(ohmic,k-1,j,i-1));
        e2(k,j,i) += eta_O * J2(k,j,i);

        eta_O = 0.25*(etaB(ohmic,k,j  ,i) + etaB(ohmic,k,j  ,i-1)
                      + etaB(ohmic,k,j-1,i) + etaB(ohmic,k,j-1,i-1));
        e3(k,j,i) += eta_O * J3(k,j,i);
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void FieldDiffusion::AmbipolarEMF
//! \brief EMF from ambipolar diffusion

void FieldDiffusion::AmbipolarEMF(const FaceField &b, const AthenaArray<Real> &bc,
                                  EdgeField &e) {
  MeshBlock *pmb = pmy_block;
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  AthenaArray<Real> &e1 = e.x1e, &e2 = e.x2e, &e3 = e.x3e,
                    &J1 = jedge_.x1e, &J2 = jedge_.x2e, &J3 = jedge_.x3e;

  // 1D update:
  if (pmb->block_size.nx2 == 1) {
    for (int i=is; i<=ie+1; ++i) {
      Real eta_A = 0.5*(etaB(ambipolar,ks,js,i-1) + etaB(ambipolar,ks,js,i));

      Real intBx = b.x1f(ks,js,i);
      Real intBy = 0.5*(b.x2f(ks,js,i) + b.x2f(ks,js,i-1));
      Real intBz = 0.5*(b.x3f(ks,js,i) + b.x3f(ks,js,i-1));

      Real Bsq   = SQR(intBx) + SQR(intBy) + SQR(intBz) + TINY_NUMBER;
      Real JdotB = J2(ks,js,i)*intBy + J3(ks,js,i)*intBz;

      e2(ks  ,js  ,i) += eta_A * (J2(ks,js,i) - JdotB*intBy/Bsq);
      e3(ks  ,js  ,i) += eta_A * (J3(ks,js,i) - JdotB*intBz/Bsq);

      e2(ke+1,js  ,i)  = e2(ks,js,i);
      e3(ks  ,je+1,i)  = e3(ks,js,i);
    }
    return;
  }

  // 2D update:
  if (pmb->block_size.nx3 == 1) {
    for (int j=js; j<=je+1; ++j) {
#pragma omp simd
      for (int i=is; i<=ie+1; ++i) {
        // emf.x
        Real eta_A = 0.5*(etaB(ambipolar,ks,j,i) + etaB(ambipolar,ks,j-1,i));

        Real intJx = J1(ks,j,i);
        Real intJy = 0.25*(J2(ks,j,  i) + J2(ks,j,  i+1)
                           +J2(ks,j-1,i) + J2(ks,j-1,i+1));
        Real intJz = 0.5 *(J3(ks,j,  i)   + J3(ks,j,i+1));

        Real intBx = 0.5*(bc(IB1,ks,j,i)+bc(IB1,ks,j-1,i));
        Real intBy = b.x2f(ks,j,i);
        Real intBz = 0.5*(b.x3f(ks,j,i)+b.x3f(ks,j-1,i));

        Real Bsq   = SQR(intBx) + SQR(intBy) + SQR(intBz) + TINY_NUMBER;
        Real JdotB = intJx*intBx + intJy*intBy + intJz*intBz;

        e1(ks  ,j,i) += eta_A * (J1(ks,j,i) - JdotB*intBx/Bsq);
        e1(ke+1,j,i)  = e1(ks,j,i);

        // emf.y
        eta_A = 0.5*(etaB(ambipolar,ks,j,i) + etaB(ambipolar,ks,j,i-1));

        intJx = 0.25*(J1(ks,j,i  ) + J1(ks,j+1,i  )
                      +J1(ks,j,i-1) + J1(ks,j+1,i-1));
        intJy = J2(ks,j,i);
        intJz = 0.5 *(J3(ks,j,i  ) + J3(ks,j+1,i));

        intBx = b.x1f(ks,j,i);
        intBy = 0.5*(bc(IB2,ks,j,i)+bc(IB2,ks,j,i-1));
        intBz = 0.5*(b.x3f(  ks,j,i)+b.x3f(  ks,j,i-1));

        Bsq   = SQR(intBx) + SQR(intBy) + SQR(intBz) + TINY_NUMBER;
        JdotB = intJx*intBx + intJy*intBy + intJz*intBz;

        e2(ks  ,j,i) += eta_A * (J2(ks,j,i) - JdotB*intBy/Bsq);
        e2(ke+1,j,i)  = e2(ks,j,i);

        // emf.z
        eta_A = 0.25*(etaB(ambipolar,ks,j  ,i) + etaB(ambipolar,ks,j  ,i-1)
                      + etaB(ambipolar,ks,j-1,i) + etaB(ambipolar,ks,j-1,i-1));

        intJx = 0.5*(J1(ks,j,i) + J1(ks,j,i-1));
        intJy = 0.5*(J2(ks,j,i) + J2(ks,j-1,i));
        intJz = J3(ks,j,i);

        intBx = 0.5*(b.x1f(ks,j,i) + b.x1f(ks,j-1,i));
        intBy = 0.5*(b.x2f(ks,j,i) + b.x2f(ks,j,i-1));
        intBz = 0.25*(b.x3f(ks,j  ,i) + b.x3f(ks,j  ,i-1)
                      +b.x3f(ks,j-1,i) + b.x3f(ks,j-1,i-1));

        Bsq   = SQR(intBx) + SQR(intBy) + SQR(intBz) + TINY_NUMBER;
        JdotB = intJx*intBx + intJy*intBy + intJz*intBz;

        e3(ks,  j,i) += eta_A * (J3(ks,j,i) - JdotB*intBz/Bsq);
        e3(ke+1,j,i)  = e3(ks,j,i);
      }
    }
    return;
  }

  // 3D update:
  for (int k=ks; k<=ke+1; ++k) {
    for (int j=js; j<=je+1; ++j) {
#pragma omp simd
      for (int i=is; i<=ie+1; ++i) {
        // emf.x
        Real eta_A = 0.25*(etaB(ambipolar,k  ,j,i) + etaB(ambipolar,k  ,j-1,i)
                           + etaB(ambipolar,k-1,j,i) + etaB(ambipolar,k-1,j-1,i));

        Real intJx = J1(k,j,i);
        Real intJy = 0.25*(J2(k,j,  i) + J2(k,j,  i+1)
                           +J2(k,j-1,i) + J2(k,j-1,i+1));
        Real intJz = 0.25*(J3(k  ,j,i)   + J3(k  ,j,i+1)
                           +J3(k-1,j,i)   + J3(k-1,j,i+1));

        Real intBx = 0.25*(bc(IB1,k  ,j,i)+bc(IB1,k  ,j-1,i)
                           +bc(IB1,k-1,j,i)+bc(IB1,k-1,j-1,i));
        Real intBy = 0.5*(b.x2f(k,j,i) + b.x2f(k-1,j,i));
        Real intBz = 0.5*(b.x3f(k,j,i) + b.x3f(k,j-1,i));

        Real Bsq   = SQR(intBx) + SQR(intBy) + SQR(intBz) + TINY_NUMBER;
        Real JdotB = intJx*intBx + intJy*intBy + intJz*intBz;

        e1(k,j,i) += eta_A * (J1(k,j,i) - JdotB*intBx/Bsq);

        // emf.y
        eta_A = 0.25*(etaB(ambipolar,k  ,j,i) + etaB(ambipolar,k  ,j,i-1)
                      + etaB(ambipolar,k-1,j,i) + etaB(ambipolar,k-1,j,i-1));

        intJx = 0.25*(J1(k,j,  i) + J1(k,j,  i-1)
                      +J1(k,j+1,i) + J1(k,j+1,i-1));
        intJy = J2(k,j,i);
        intJz = 0.25*(J3(k  ,j,i)   + J3(k  ,j+1,i)
                      +J3(k-1,j,i)   + J3(k-1,j+1,i));

        intBx = 0.5*(b.x1f(k,j,i) + b.x1f(k-1,j,i));
        intBy = 0.25*(bc(IB2,k  ,j,i)+bc(IB2,k  ,j,i-1)
                      +bc(IB2,k-1,j,i)+bc(IB2,k-1,j,i-1));
        intBz = 0.5*(b.x3f(k,j,i) + b.x3f(k,j,i-1));

        Bsq   = SQR(intBx) + SQR(intBy) + SQR(intBz) + TINY_NUMBER;
        JdotB = intJx*intBx + intJy*intBy + intJz*intBz;

        e2(k,j,i) += eta_A * (J2(k,j,i) - JdotB*intBy/Bsq);

        // emf.z
        eta_A = 0.25*(etaB(ambipolar,k,j  ,i) + etaB(ambipolar,k,j  ,i-1)
                      + etaB(ambipolar,k,j-1,i) + etaB(ambipolar,k,j-1,i-1));

        intJx = 0.25*(J1(k  ,j,i) + J1(k  ,j,i-1)
                      +J1(k+1,j,i) + J1(k+1,j,i-1));
        intJy = 0.25*(J2(k  ,j,i) + J2(k  ,j-1,i)
                      +J2(k+1,j,i) + J2(k+1,j-1,i));
        intJz = J3(k,j,i);

        intBx = 0.5*(b.x1f(k,j,i) + b.x1f(k,j-1,i));
        intBy = 0.5*(b.x2f(k,j,i) + b.x2f(k,j,i-1));
        intBz = 0.25*(bc(IB3,k,j  ,i)+bc(IB3,k,j  ,i-1)
                      +bc(IB3,k,j-1,i)+bc(IB3,k,j-1,i-1));

        Bsq   = SQR(intBx) + SQR(intBy) + SQR(intBz) + TINY_NUMBER;
        JdotB = intJx*intBx + intJy*intBy + intJz*intBz;

        e3(k,j,i) += eta_A * (J3(k,j,i) - JdotB*intBz/Bsq);
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void FieldDiffusion::HallrEMF
//! \brief EMF from Hall effect
void FieldDiffusion::HallEMF(const FaceField &b, const AthenaArray<Real> &bc,
                             EdgeField &e) {
  MeshBlock *pmb = pmy_block;
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  AthenaArray<Real> &e1 = e.x1e, &e2 = e.x2e, &e3 = e.x3e,
                    &J1 = jedge_.x1e, &J2 = jedge_.x2e, &J3 = jedge_.x3e;

  Real bx = 0.0; Real by = 0.0; Real bz = 0.0;

  // 1D update:
  if (pmb->block_size.nx2 == 1) {
    for (int i=is; i<=ie+1; ++i) {
      Real eta_H = 0.5*(etaB(hall,ks,js,i-1) + etaB(hall,ks,js,i));

      Real Bx = b.x1f(ks,js,i);
      Real By = 0.5*(b.x2f(ks,js,i) + b.x2f(ks,js,i-1));
      Real Bz = 0.5*(b.x3f(ks,js,i) + b.x3f(ks,js,i-1));
      Real B   = std::sqrt(SQR(Bx) + SQR(By) + SQR(Bz));

      if (B > TINY_NUMBER) {
        bx = Bx/B; by = By/B; bz = Bz/B;
      } else {
        bx = 0.0; by = 0.0; bz = 0.0;
      }

      // Wedge =  b x J
      Real bcJy = bz*J1(ks,js,i) - bx*J3(ks,js,i);
      Real bcJz = bx*J2(ks,js,i) - by*J1(ks,js,i);

      e2(ks  ,js  ,i) -= eta_H*bcJy;
      e3(ks  ,js  ,i) -= eta_H*bcJz;

      e2(ke+1,js  ,i)  = e2(ks,js,i);
      e3(ks  ,je+1,i)  = e3(ks,js,i);
    }
    return;
  }

  // 2D update:
  if (pmb->block_size.nx3 == 1) {
    for (int j=js; j<=je+1; ++j) {
#pragma omp simd
      for (int i=is; i<=ie+1; ++i) {
        // emf.x
        Real eta_H = 0.5*(etaB(hall,ks,j,i) + etaB(hall,ks,j-1,i));

        Real Jx = J1(ks,j,i);
        Real Jy = 0.25*(J2(ks,j,  i) + J2(ks,j,  i+1)
                           +J2(ks,j-1,i) + J2(ks,j-1,i+1));
        Real Jz = 0.5 *(J3(ks,j,  i)   + J3(ks,j,i+1));

        Real Bx = 0.5*(bc(IB1,ks,j,i)+bc(IB1,ks,j-1,i));
        Real By = b.x2f(ks,j,i);
        Real Bz = 0.5*(b.x3f(ks,j,i)+b.x3f(ks,j-1,i));
        Real B   = std::sqrt(SQR(Bx) + SQR(By) + SQR(Bz));

        if (B > TINY_NUMBER) {
          bx = Bx/B; by = By/B; bz = Bz/B;
        } else {
          bx = 0.0; by = 0.0; bz = 0.0;
        }

        // Wedge =  b x J
        Real bcJx = by*Jz - bz*Jy;

        e1(ks  ,j,i) -= eta_H * bcJx;
        e1(ke+1,j,i)  = e1(ks,j,i);

        // emf.y
        eta_H = 0.5*(etaB(hall,ks,j,i) + etaB(hall,ks,j,i-1));

        Jx = 0.25*(J1(ks,j,i  ) + J1(ks,j+1,i  )
                      +J1(ks,j,i-1) + J1(ks,j+1,i-1));
        Jy = J2(ks,j,i);
        Jz = 0.5 *(J3(ks,j,i  ) + J3(ks,j+1,i));

        Bx = b.x1f(ks,j,i);
        By = 0.5*(bc(IB2,ks,j,i)+bc(IB2,ks,j,i-1));
        Bz = 0.5*(b.x3f(  ks,j,i)+b.x3f(  ks,j,i-1));
        B   = std::sqrt(SQR(Bx) + SQR(By) + SQR(Bz));

        if (B > TINY_NUMBER) {
          bx = Bx/B; by = By/B; bz = Bz/B;
        } else {
          bx = 0.0; by = 0.0; bz = 0.0;
        }

        Real bcJy = bx*Jz - bz*Jx;

        e2(ks  ,j,i) -= eta_H * bcJy;
        e2(ke+1,j,i)  = e2(ks,j,i);

        // emf.z
        eta_H = 0.25*(etaB(hall,ks,j  ,i) + etaB(hall,ks,j  ,i-1)
                      + etaB(hall,ks,j-1,i) + etaB(hall,ks,j-1,i-1));

        Jx = 0.5*(J1(ks,j,i) + J1(ks,j,i-1));
        Jy = 0.5*(J2(ks,j,i) + J2(ks,j-1,i));
        Jz = J3(ks,j,i);

        Bx = 0.5*(b.x1f(ks,j,i) + b.x1f(ks,j-1,i));
        By = 0.5*(b.x2f(ks,j,i) + b.x2f(ks,j,i-1));
        Bz = 0.25*(b.x3f(ks,j  ,i) + b.x3f(ks,j  ,i-1)
                      +b.x3f(ks,j-1,i) + b.x3f(ks,j-1,i-1));
        B  = std::sqrt(SQR(Bx) + SQR(By) + SQR(Bz));

        if (B > TINY_NUMBER) {
          bx = Bx/B; by = By/B; bz = Bz/B;
        } else {
          bx = 0.0; by = 0.0; bz = 0.0;
        }

        Real bcJz = bx*Jy - by*Jx;

        e3(ks,  j,i) -= eta_H * bcJz;
        e3(ke+1,j,i)  = e3(ks,j,i);
      }
    }
    return;
  }

  // 3D update:
  for (int k=ks; k<=ke+1; ++k) {
    for (int j=js; j<=je+1; ++j) {
#pragma omp simd
      for (int i=is; i<=ie+1; ++i) {
        // emf.x
        Real eta_H = 0.25*(etaB(hall,k  ,j,i) + etaB(hall,k  ,j-1,i)
                           + etaB(hall,k-1,j,i) + etaB(hall,k-1,j-1,i));

        Real Jx = J1(k,j,i);
        Real Jy = 0.25*(J2(k,j,  i) + J2(k,j,  i+1)
                           +J2(k,j-1,i) + J2(k,j-1,i+1));
        Real Jz = 0.25*(J3(k  ,j,i)   + J3(k  ,j,i+1)
                           +J3(k-1,j,i)   + J3(k-1,j,i+1));

        Real Bx = 0.25*(bc(IB1,k  ,j,i)+bc(IB1,k  ,j-1,i)
                           +bc(IB1,k-1,j,i)+bc(IB1,k-1,j-1,i));
        Real By = 0.5*(b.x2f(k,j,i) + b.x2f(k-1,j,i));
        Real Bz = 0.5*(b.x3f(k,j,i) + b.x3f(k,j-1,i));

        Real B = std::sqrt(SQR(Bx) + SQR(By) + SQR(Bz));

        if (B > TINY_NUMBER) {
          bx = Bx/B; by = By/B; bz = Bz/B;
        } else {
          bx = 0.0; by = 0.0; bz = 0.0;
        }

        Real bcJx = by*Jz - bz*Jy;

        e1(k,j,i) -= eta_H * bcJx;

        // emf.y
        eta_H = 0.25*(etaB(hall,k  ,j,i) + etaB(hall,k  ,j,i-1)
                      + etaB(hall,k-1,j,i) + etaB(hall,k-1,j,i-1));

        Jx = 0.25*(J1(k,j,  i) + J1(k,j,  i-1)
                      +J1(k,j+1,i) + J1(k,j+1,i-1));
        Jy = J2(k,j,i);
        Jz = 0.25*(J3(k  ,j,i)   + J3(k  ,j+1,i)
                      +J3(k-1,j,i)   + J3(k-1,j+1,i));

        Bx = 0.5*(b.x1f(k,j,i) + b.x1f(k-1,j,i));
        By = 0.25*(bc(IB2,k  ,j,i)+bc(IB2,k  ,j,i-1)
                      +bc(IB2,k-1,j,i)+bc(IB2,k-1,j,i-1));
        Bz = 0.5*(b.x3f(k,j,i) + b.x3f(k,j,i-1));
        B  = std::sqrt(SQR(Bx) + SQR(By) + SQR(Bz));

        if (B > TINY_NUMBER) {
          bx = Bx/B; by = By/B; bz = Bz/B;
        } else {
          bx = 0.0; by = 0.0; bz = 0.0;
        }

        Real bcJy = bz*Jx - bx*Jz;

        e2(k,j,i) -= eta_H * bcJy;

        // emf.z
        eta_H = 0.25*(etaB(hall,k,j  ,i) + etaB(hall,k,j  ,i-1)
                      + etaB(hall,k,j-1,i) + etaB(hall,k,j-1,i-1));

        Jx = 0.25*(J1(k  ,j,i) + J1(k  ,j,i-1)
                      +J1(k+1,j,i) + J1(k+1,j,i-1));
        Jy = 0.25*(J2(k  ,j,i) + J2(k  ,j-1,i)
                      +J2(k+1,j,i) + J2(k+1,j-1,i));
        Jz = J3(k,j,i);

        Bx = 0.5*(b.x1f(k,j,i) + b.x1f(k,j-1,i));
        By = 0.5*(b.x2f(k,j,i) + b.x2f(k,j,i-1));
        Bz = 0.25*(bc(IB3,k,j  ,i)+bc(IB3,k,j  ,i-1)
                      +bc(IB3,k,j-1,i)+bc(IB3,k,j-1,i-1));
        B  = std::sqrt(SQR(Bx) + SQR(By) + SQR(Bz));

        if (B > TINY_NUMBER) {
          bx = Bx/B; by = By/B; bz = Bz/B;
        } else {
          bx = 0.0; by = 0.0; bz = 0.0;
        }

        Real bcJz = bx*Jy - by*Jx;

        e3(k,j,i) -= eta_H * bcJz;
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void FieldDiffusion::ThermoElectricConduction(...)
//! \brief Braginskii thermoelectric conduction
//  SB1: Braginskii, S., “Transport Processes in a Plasma”, Reviews of Plasma Physics,
//  Jan. 1965, vol. 1, pp. 205,
//  URL: https://static.ias.edu/pitp/2016/sites/pitp/files/braginskii_1965-1.pdf.
//  RF1: https://farside.ph.utexas.edu/teaching/plasma/Plasma/node53.html
//  The energy flux terms qJx, qJy, qJz solve SB1 eqn(4.32) (page 249).
//  Also see last two terms of RF1 eqn.(4.120)
//    (perpendicular is omitted in strongly magnetised limit)

void FieldDiffusion::ThermoElectricConduction(MeshBlock *pmb, const FaceField &b,
                                              const AthenaArray<Real> &bc,
                                              AthenaArray<Real> *flx,
                                              AthenaArray<Real> *flx_disp) {
  Coordinates* pco_ = pmb->pcoord;
  AthenaArray<Real> &x1flux = flx[X1DIR];
  AthenaArray<Real> &x2flux = flx[X2DIR];
  AthenaArray<Real> &x3flux = flx[X3DIR];
  AthenaArray<Real> &x1flux_disp = flx_disp[X1DIR];
  AthenaArray<Real> &x2flux_disp = flx_disp[X2DIR];
  AthenaArray<Real> &x3flux_disp = flx_disp[X3DIR];

  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  Real therm_para = 0.0; Real therm_perp = 0.0; Real therm_wedge=0.0; Real ion_frac = 0.0;
  Real B = 0.0; Real Bx=0.0; Real By =0.0; Real Bz = 0.0;
  Real bx = 0.0; Real by= 0.0; Real bz = 0.0;
  Real Jx = 0.0; Real Jy =0.0; Real Jz =0.0;
  Real b_dot_J = 0.0;

  Real qJx_para = 0.0; Real qJx_perp = 0.0; Real qJx_wedge = 0.0;
  Real qJy_para = 0.0; Real qJy_perp = 0.0; Real qJy_wedge = 0.0;
  Real qJz_para = 0.0; Real qJz_perp = 0.0; Real qJz_wedge = 0.0;

  // x face thermoelectric conduction
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      #pragma omp simd
      for (int i=is; i<=ie+1; ++i) {
        therm_para = therm_perp = therm_wedge = 0.0;
        qJx_para = 0.0; qJx_perp = 0.0; qJx_wedge = 0.0;

        // B at x1-face
        Bx = b.x1f(k,j,i);
        By = 0.5*(bc(IB2,k,j,i) + bc(IB2,k,j,i-1));
        Bz = 0.5*(bc(IB3,k,j,i) + bc(IB3,k,j,i-1));
        B = std::sqrt(Bx*Bx + By*By + Bz*Bz);

        if (B > TINY_NUMBER) {
          bx = Bx/B; by = By/B; bz = Bz/B;
        } else {
          bx = 0.0; by = 0.0; bz = 0.0;
        }

          // J at x1-face:
          Jx = jfx(IB1,k,j,i);
          Jy = jfx(IB2,k,j,i);
          Jz = jfx(IB3,k,j,i);

          b_dot_J = bx*Jx + by*Jy + bz*Jz;

          if (use_TE_para) { // J_para = b(b dot J)
            therm_para  = 0.5*(thermKappa(ThermElecProcess::kappa_TE_para ,k,j,i)
                          + thermKappa(ThermElecProcess::kappa_TE_para ,k,j,i-1));
            qJx_para = -therm_para*b_dot_J*bx;
          }

          if (use_TE_perp) { // J_perp = J - b(b dot J)
            therm_perp  = 0.5*(thermKappa(ThermElecProcess::kappa_TE_perp ,k,j,i)
                          + thermKappa(ThermElecProcess::kappa_TE_perp ,k,j,i-1));
            Real Jpx = Jx - bx*b_dot_J;
            qJx_perp = -therm_perp*Jpx;
          }

          if (use_TE_wedge) { // J_wedge = b cross J
            therm_wedge = 0.5*(thermKappa(ThermElecProcess::kappa_TE_wedge,k,j,i)
                          + thermKappa(ThermElecProcess::kappa_TE_wedge,k,j,i-1));
            Real bcJx = by*Jz - bz*Jy;
            qJx_wedge = -therm_wedge*bcJx;
          }

          // ion fraction weighting
          if (pmb->phydro->hdif.kappa_aniso_neutral > 0.0
              || pmb->phydro->hdif.nu_aniso_neutral) {
            ion_frac = 0.5*(pmb->phydro->hdif.ion_frac(k,j,i)
                       + pmb->phydro->hdif.ion_frac(k,j,i-1));
            qJx_para *= ion_frac; qJx_perp *= ion_frac; qJx_wedge *= ion_frac;
          }
            x1flux(k,j,i) += qJx_wedge;
            x1flux_disp(k,j,i) += qJx_para + qJx_perp;
            if(!STS_ENABLED) x1flux(k,j,i) += qJx_para + qJx_perp;
      } // i loop
    } // j loop
  } // k loop

  // y face thermoelectric conduction
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je+1; ++j) {
      #pragma omp simd
      for (int i=is; i<=ie; ++i) {
        therm_para = therm_perp = therm_wedge = 0.0;
        qJy_para = qJy_perp = qJy_wedge = 0.0;

        // B at x2-face
        Bx = 0.5*(bc(IB1,k,j,i) + bc(IB1,k,j-1,i));
        By = b.x2f(k,j,i);
        Bz = 0.5*(bc(IB3,k,j,i) + bc(IB3,k,j-1,i));
        B = std::sqrt(Bx*Bx + By*By + Bz*Bz);

        if (B > TINY_NUMBER) {
          bx = Bx/B; by = By/B; bz = Bz/B;
        } else {
          bx = 0.0; by = 0.0; bz = 0.0;
        }

          // J at x2-face:
          Jx = jfy(IB1,k,j,i);
          Jy = jfy(IB2,k,j,i);
          Jz = jfy(IB3,k,j,i);

          b_dot_J = bx*Jx + by*Jy + bz*Jz;

          if (use_TE_para) { // J_para = b(b dot J)
            therm_para  = 0.5*(thermKappa(ThermElecProcess::kappa_TE_para ,k,j,i)
                          + thermKappa(ThermElecProcess::kappa_TE_para ,k,j-1,i));
            qJy_para = -therm_para*b_dot_J*by;
          }
          if (use_TE_perp) { // J_perp = J - b(b dot J)
            therm_perp  = 0.5*(thermKappa(ThermElecProcess::kappa_TE_perp ,k,j,i)
                          + thermKappa(ThermElecProcess::kappa_TE_perp ,k,j-1,i));
            Real Jpy = Jy - by*b_dot_J;
            qJy_perp = -therm_perp*Jpy;
          }
          if (use_TE_wedge) { // J_wedge = b cross J
            therm_wedge = 0.5*(thermKappa(ThermElecProcess::kappa_TE_wedge,k,j,i)
                          + thermKappa(ThermElecProcess::kappa_TE_wedge,k,j-1,i));
            Real bcJy = bz*Jx - bx*Jz;
            qJy_wedge = -therm_wedge*bcJy;
          }

          // ion fraction weighting
          if (pmb->phydro->hdif.kappa_aniso_neutral > 0.0
              || pmb->phydro->hdif.nu_aniso_neutral) {
            ion_frac = 0.5*(pmb->phydro->hdif.ion_frac(k,j,i)
                       + pmb->phydro->hdif.ion_frac(k,j-1,i));
            qJy_para *= ion_frac; qJy_perp *= ion_frac; qJy_wedge *= ion_frac;
          }
          x2flux(k,j,i) += qJy_wedge;
          x2flux_disp(k,j,i) += qJy_para + qJy_perp;
          if(!STS_ENABLED) x2flux(k,j,i) += qJy_para + qJy_perp;
      } // i loop
    } // j loop
  } // k loop

    // z face thermoelectric conduction
    for (int k=ks; k<=ke+1; ++k) {
      for (int j=js; j<=je; ++j) {
        #pragma omp simd
        for (int i=is; i<=ie; ++i) {
          therm_para = therm_perp = therm_wedge = 0.0;
          qJz_para =0.0; qJz_perp =0.0; qJz_wedge =0.0;

          // B at x3-face
          Bx = 0.5*(bc(IB1,k,j,i) + bc(IB1,k-1,j,i));
          By = 0.5*(bc(IB2,k,j,i) + bc(IB2,k-1,j,i));
          Bz = b.x3f(k,j,i);
          B = std::sqrt(Bx*Bx + By*By + Bz*Bz);

          if (B > TINY_NUMBER) {
            bx = Bx/B; by = By/B; bz = Bz/B;
          } else {
            bx = 0.0; by = 0.0; bz = 0.0;
          }

          // J at x3-face:
          Jx = jfz(IB1,k,j,i);
          Jy = jfz(IB2,k,j,i);
          Jz = jfz(IB3,k,j,i);

          b_dot_J = bx*Jx + by*Jy + bz*Jz;

          if (use_TE_para) { // J_para = b(b dot J)
            therm_para  = 0.5*(thermKappa(ThermElecProcess::kappa_TE_para ,k,j,i)
                          + thermKappa(ThermElecProcess::kappa_TE_para ,k-1,j,i));
            qJz_para = -therm_para*b_dot_J*bz;
          }
          if (use_TE_perp) { // J_perp = J - b(b dot J)
            therm_perp  = 0.5*(thermKappa(ThermElecProcess::kappa_TE_perp ,k,j,i)
                          + thermKappa(ThermElecProcess::kappa_TE_perp ,k-1,j,i));
            Real Jpz = Jz - bz*b_dot_J;
            qJz_perp = -therm_perp*Jpz;
          }
          if (use_TE_wedge) { // J_wedge = b cross J
            therm_wedge = 0.5*(thermKappa(ThermElecProcess::kappa_TE_wedge,k,j,i)
                          + thermKappa(ThermElecProcess::kappa_TE_wedge,k-1,j,i));
            Real bcJz = bx*Jy - by*Jx;
            qJz_wedge = -therm_wedge*bcJz;
          }

          // ion fraction weighting
          if (pmb->phydro->hdif.kappa_aniso_neutral > 0.0
              || pmb->phydro->hdif.nu_aniso_neutral) {
            ion_frac = 0.5*(pmb->phydro->hdif.ion_frac(k,j,i)
                       + pmb->phydro->hdif.ion_frac(k-1,j,i));
            qJz_para *= ion_frac; qJz_perp *= ion_frac; qJz_wedge *= ion_frac;
          }
          x3flux(k,j,i) += qJz_wedge;
          x3flux_disp(k,j,i) += qJz_para + qJz_perp;
          if(!STS_ENABLED) x3flux(k,j,i) += qJz_para + qJz_perp;
        } // i loop
      } // j loop
    } // k loop
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void FieldDiffusion::ThermoElectricForce(...)
//! \brief Braginskii thermoelectric force
//  SB1: Braginskii, S., “Transport Processes in a Plasma”, Reviews of Plasma Physics,
//  Jan. 1965, vol. 1, pp. 205,
//  URL: https://static.ias.edu/pitp/2016/sites/pitp/files/braginskii_1965-1.pdf.
//  RF1: https://farside.ph.utexas.edu/teaching/plasma/Plasma/node53.html
//  The Fx, Fy, and Fz terms solve SB1 eqn(4.31) (page 249)

void FieldDiffusion::ThermoElectricForce(MeshBlock *pmb, const FaceField &b,
                                         const AthenaArray<Real> &bc,
                                         EdgeField &e_diff, EdgeField &e_disp) {
  Coordinates* pco_ = pmb->pcoord;
  AthenaArray<Real> &w = pmb->phydro->w;

  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  const bool f2 = pmb->pmy_mesh->f2;
  const bool f3 = pmb->pmy_mesh->f3;

  AthenaArray<Real> &e1_diff = e_diff.x1e, &e2_diff = e_diff.x2e, &e3_diff = e_diff.x3e,
                    &e1_disp = e_disp.x1e, &e2_disp = e_disp.x2e, &e3_disp = e_disp.x3e;

  Real therm_para = 0.0; Real therm_perp = 0.0; Real therm_wedge = 0.0;
  Real Tedge = 0.0; Real dTdx = 0.0; Real dTdy = 0.0; Real dTdz = 0.0;
  Real B = 0.0; Real Bx= 0.0; Real By = 0.0; Real Bz = 0.0;
  Real bx = 0.0; Real by= 0.0; Real bz = 0.0;
  Real ion_frac = 0.0; Real b_dot_gradT = 0.0;
  Real FTx_para = 0.0; Real FTy_para = 0.0; Real FTz_para = 0.0;
  Real FTx_perp = 0.0; Real FTy_perp = 0.0; Real FTz_perp = 0.0;
  Real FTx_wedge = 0.0; Real FTy_wedge = 0.0; Real FTz_wedge = 0.0;

  if (!f2) { // 1D update:
        for (int i=is; i<=ie+1; ++i) {
          // emf.x
          therm_para =0.0; therm_perp =0.0; therm_wedge = 0.0;
          FTy_para = FTy_perp = FTy_wedge = 0.0;
          FTz_para = FTz_perp = FTz_wedge = 0.0;
          Bx = b.x1f(ks,js,i);
          By = 0.5*(b.x2f(ks,js,i) + b.x2f(ks,js,i-1));
          Bz = 0.5*(b.x3f(ks,js,i) + b.x3f(ks,js,i-1));
          B = std::sqrt(Bx*Bx + By*By + Bz*Bz);

          if (B > TINY_NUMBER) {
            bx = Bx/B; by = By/B; bz = Bz/B;
          } else {
            bx = 0.0; by = 0.0; bz = 0.0;
          }

          if (use_TE_para) {
            therm_para = 0.5*(thermKappa(ThermElecProcess::kappa_TE_para,ks,js,i-1)
                          + thermKappa(ThermElecProcess::kappa_TE_para,ks,js,i));}
          if (use_TE_perp) {
            therm_perp = 0.5*(thermKappa(ThermElecProcess::kappa_TE_perp,ks,js,i-1)
                         + thermKappa(ThermElecProcess::kappa_TE_perp,ks,js,i));}
          if (use_TE_wedge) {
            therm_wedge = 0.5*(thermKappa(ThermElecProcess::kappa_TE_wedge,ks,js,i-1)
                          + thermKappa(ThermElecProcess::kappa_TE_wedge,ks,js,i));}

          Tedge = Te2(ks,js,i); // Te2 and Te3 are equal in 1D

          // dT/dx at x edge
          dTdx = (Tcc(ks,js,i) -Tcc(ks,js,i-1))
                  /pco_->dx1v(i-1);

          // para = bdot_gradT
          b_dot_gradT = bx*dTdx;

            // Perp = gradT_perp = gradT - b(b dot gradT)
          Real gTpy = -by*b_dot_gradT;
          Real gTpz = - bz*b_dot_gradT;

            // Wedge =  b x gradT
          Real bcTy = bz*dTdx;
          Real bcTz = -by*dTdx;

          // THERMOELECTRIC FORCE F_T
          FTy_para = -(1.0/Tedge)*(therm_para*by*b_dot_gradT);
          FTy_perp = -(1.0/Tedge)*(therm_perp*gTpy);
          FTy_wedge = -(1.0/Tedge)*(therm_wedge*bcTy);

          FTz_para = -(1.0/Tedge)*(therm_para*bz*b_dot_gradT);
          FTz_perp = -(1.0/Tedge)*(therm_perp*gTpz);
          FTz_wedge = -(1.0/Tedge)*(therm_wedge*bcTz);

          if(pmb->phydro->hdif.kappa_aniso_neutral > 0.0) {
          ion_frac = 0.5*(pmb->phydro->hdif.ion_frac(ks,js,i)
                     + pmb->phydro->hdif.ion_frac(ks,js,i-1));
            FTy_para *= ion_frac; FTy_perp *= ion_frac; FTy_wedge *= ion_frac;
            FTz_para *= ion_frac; FTz_perp *= ion_frac; FTz_wedge *= ion_frac;
        }

          // Update emfs
          e2_diff(ks,js,  i) += FTy_wedge;
          e2_diff(ke+1,js,i)  = e2_diff(ks,js,i);
          e2_disp(ks,js,  i) += FTy_para + FTy_perp;
          e2_disp(ke+1,js,i)  = e2_disp(ks,js,i);

          e3_diff(ks,js,  i) += FTz_wedge;
          e3_diff(ks,je+1,i)  = e3_diff(ks,js,i);
          e3_disp(ks,js,  i) += FTz_para + FTz_perp;
          e3_disp(ks,je+1,i)  = e3_disp(ks,js,i);
        } // i loop
  } else  if (!f3) {// 2D update:
    for (int j=js; j<=je+1; ++j) {
  #pragma omp simd
      for (int i=is; i<=ie+1; ++i) {
        // emf.x
        therm_para = 0.0; therm_perp = 0.0; therm_wedge = 0.0;
        FTx_para = 0.0; FTx_perp = 0.0; FTx_wedge = 0.0;
        Bx = 0.5*(bc(IB1,ks,j-1,i) + bc(IB1,ks,j,i));
        By = b.x2f(ks,j,i);
        Bz = 0.5*(b.x3f(ks,j-1,i) + b.x3f(ks,j,i));
        B = std::sqrt(Bx*Bx + By*By + Bz*Bz);

        if (B > TINY_NUMBER) {
          bx = Bx/B; by = By/B; bz = Bz/B;
        } else {
          bx = 0.0; by = 0.0; bz = 0.0;
        }

        if (use_TE_para) {
          therm_para = 0.5*(thermKappa(ThermElecProcess::kappa_TE_para,ks,j-1,i)
                       + thermKappa(ThermElecProcess::kappa_TE_para,ks,j,i));}
        if (use_TE_perp) {
          therm_perp = 0.5*(thermKappa(ThermElecProcess::kappa_TE_perp,ks,j-1,i)
                       + thermKappa(ThermElecProcess::kappa_TE_perp,ks,j,i));}
        if (use_TE_wedge) {
          therm_wedge = 0.5*(thermKappa(ThermElecProcess::kappa_TE_wedge,ks,j-1,i)
                        + thermKappa(ThermElecProcess::kappa_TE_wedge,ks,j,i));}

        Tedge = Te1(ks,j,i);

        // dT/dxat x edge
        dTdx = (Te1(ks,j,i+1) - Te1(ks,j,i-1))
              /(pco_->dx1v(i-1) + pco_->dx1v(i));

        // dT/dy at x edge
        dTdy = (Tcc(ks,j,i) - Tcc(ks,j-1,i) )
              / (pco_->h2v(i) * pco_->dx2v(j-1));

        // para: b (bdot_gradT)
        b_dot_gradT = bx*dTdx + by*dTdy;

        // Perp = gradT_perp = gradT - b(b dot gradT)
        Real gTpx = dTdx - bx*b_dot_gradT;

        // Wedge =  b x gradT
        Real bcTx = - bz*dTdy;

        // THERMOELECTRIC FORCE F_T
        FTx_para = -(1.0/Te1(ks,j,i))*(therm_para*bx*b_dot_gradT);
        FTx_perp = -(1.0/Te1(ks,j,i))*(therm_perp*gTpx);
        FTx_wedge = -(1.0/Te1(ks,j,i))*(therm_wedge*bcTx);

        if (pmb->phydro->hdif.kappa_aniso_neutral > 0.0) {
          ion_frac = 0.5*(pmb->phydro->hdif.ion_frac(ks,j-1,i)
                     + pmb->phydro->hdif.ion_frac(ks,j,i));
          FTx_para *= ion_frac; FTx_perp *= ion_frac; FTx_wedge *= ion_frac;
        }

        e1_diff(ks  ,j,i) += FTx_wedge;
        e1_diff(ke+1,j,i)  = e1_diff(ks,j,i);
        e1_disp(ks  ,j,i) += FTx_para + FTx_perp;
        e1_disp(ke+1,j,i)  = e1_disp(ks,j,i);

        // emf.y
        therm_para = 0.0; therm_perp = 0.0; therm_wedge = 0.0;
        FTy_para = 0.0; FTy_perp = 0.0; FTy_wedge = 0.0;
        Bx = b.x1f(ks,j,i);
        By = 0.5*(bc(IB2,ks,j,i)+bc(IB2,ks,j,i-1));
        Bz = 0.5*(b.x3f(  ks,j,i)+b.x3f(  ks,j,i-1));
        B = std::sqrt(Bx*Bx + By*By + Bz*Bz);

        if (B > TINY_NUMBER) {
          bx = Bx/B; by = By/B; bz = Bz/B;
        } else {
          bx = 0.0; by = 0.0; bz = 0.0;
        }

        if (use_TE_para) {
          therm_para = 0.5*(thermKappa(ThermElecProcess::kappa_TE_para,ks,j,i-1)
                       + thermKappa(ThermElecProcess::kappa_TE_para,ks,j,i));}
        if (use_TE_perp) {
          therm_perp = 0.5*(thermKappa(ThermElecProcess::kappa_TE_perp,ks,j,i-1)
                       + thermKappa(ThermElecProcess::kappa_TE_perp,ks,j,i));}
        if (use_TE_wedge) {
          therm_wedge = 0.5*(thermKappa(ThermElecProcess::kappa_TE_wedge,ks,j,i-1)
                        + thermKappa(ThermElecProcess::kappa_TE_wedge,ks,j,i));}

        Tedge = Te2(ks,j,i);

        // dT/dx at edge y
        dTdx = (Tcc(ks,j,i) - Tcc(ks,j,i-1))
              /pco_->dx1v(i-1);

        // dT/dy at edge y
        dTdy = (Te2(ks,j+1,i) - Te2(ks,j-1,i))
              / pco_->h2f(i)/ (pco_->dx2v(j-1) + pco_->dx2v(j));

        // para: b (bdot_gradT)
        b_dot_gradT = bx*dTdx + by*dTdy;

        // Perp = gradT_perp = gradT - b(b dot gradT)
        Real gTpy = dTdy - by*b_dot_gradT;

        // Wedge =  b x gradT
        Real bcTy = bz*dTdx;

        // THERMOELECTRIC FORCE F_T
        FTy_para = -(1.0/Tedge)*(therm_para*by*b_dot_gradT);
        FTy_perp = -(1.0/Tedge)*(therm_perp*gTpy);
        FTy_wedge = -(1.0/Tedge)*(therm_wedge*bcTy);

        if (pmb->phydro->hdif.kappa_aniso_neutral > 0.0) {
          ion_frac = 0.5*(pmb->phydro->hdif.ion_frac(ks,j,i-1)
                     + pmb->phydro->hdif.ion_frac(ks,j,i));
          FTy_para *= ion_frac; FTy_perp *= ion_frac; FTy_wedge *= ion_frac;
        }

        e2_diff(ks  ,j,i) += FTy_wedge;
        e2_diff(ke+1,j,i)  = e2_diff(ks,j,i);
        e2_disp(ks  ,j,i) += FTy_para + FTy_perp;
        e2_disp(ke+1,j,i)  = e2_disp(ks,j,i);

        // emf.z
        therm_para = 0.0; therm_perp = 0.0; therm_wedge = 0.0;
        FTz_para = 0.0; FTz_perp = 0.0; FTz_wedge = 0.0;
        Bx = 0.5*(b.x1f(ks,j,i) + b.x1f(ks,j-1,i));
        By = 0.5*(b.x2f(ks,j,i) + b.x2f(ks,j,i-1));
        Bz = 0.25*(b.x3f(ks,j  ,i) + b.x3f(ks,j  ,i-1)
                  +b.x3f(ks,j-1,i) + b.x3f(ks,j-1,i-1));
        B = std::sqrt(Bx*Bx + By*By + Bz*Bz);

        if (B > TINY_NUMBER) {
          bx = Bx/B; by = By/B; bz = Bz/B;
        } else {
          bx = 0.0; by = 0.0; bz = 0.0;
        }

        if (use_TE_para) {
          therm_para = 0.25*(thermKappa(ThermElecProcess::kappa_TE_para,ks,j,i)
                       + thermKappa(ThermElecProcess::kappa_TE_para,ks,j,i-1)
                       + thermKappa(ThermElecProcess::kappa_TE_para,ks,j-1,i)
                       + thermKappa(ThermElecProcess::kappa_TE_para,ks,j-1,i-1));}
        if (use_TE_perp) {
          therm_perp = 0.25*(thermKappa(ThermElecProcess::kappa_TE_perp,ks,j,i)
                       + thermKappa(ThermElecProcess::kappa_TE_perp,ks,j,i-1)
                       +  thermKappa(ThermElecProcess::kappa_TE_perp,ks,j-1,i)
                       +  thermKappa(ThermElecProcess::kappa_TE_perp,ks,j-1,i-1));}
        if (use_TE_wedge) {
          therm_wedge = 0.25*(thermKappa(ThermElecProcess::kappa_TE_wedge,ks,j,i)
                        + thermKappa(ThermElecProcess::kappa_TE_wedge,ks,j,i-1)
                        + thermKappa(ThermElecProcess::kappa_TE_wedge,ks,j-1,i)
                        + thermKappa(ThermElecProcess::kappa_TE_wedge,ks,j-1,i-1));}

        Tedge = Te3(ks,j,i);

        // corner dT/dx
        dTdx = (Te1(ks,j,i) - Te1(ks,j,i-1))
                /(pco_->dx1v(i-1));

        // corner dT/dy
        dTdy = (Te2(ks,j,i) - Te2(ks,j-1,i))
        / (pco_->h2f(i) * pco_->dx2v(j-1));

        // para: b (bdot_gradT)
        b_dot_gradT = bx*dTdx + by*dTdy;

        // Perp = gradT_perp = gradT - b(b dot gradT)
        Real gTpz = - bz*b_dot_gradT;

        // Wedge =  b x gradT
        Real bcTz = bx*dTdy - by*dTdx;

        // THERMOELECTRIC FORCE F_T
        FTz_para = -(1.0/Tedge)*(therm_para*bz*b_dot_gradT);
        FTz_perp = -(1.0/Tedge)*(therm_perp*gTpz);
        FTz_wedge = -(1.0/Tedge)*(therm_wedge*bcTz);

      if (pmb->phydro->hdif.kappa_aniso_neutral > 0.0) {
        ion_frac = 0.25*(pmb->phydro->hdif.ion_frac(ks,j,i)
                   + pmb->phydro->hdif.ion_frac(ks,j,i-1)
                   + pmb->phydro->hdif.ion_frac(ks,j-1,i)
                   + pmb->phydro->hdif.ion_frac(ks,j-1,i-1));
        FTz_para *= ion_frac; FTz_perp *= ion_frac; FTz_wedge *= ion_frac;
      }

        e3_diff(ks,  j,i) += FTz_wedge;
        e3_diff(ke+1,j,i)  = e3_diff(ks,j,i);
        e3_disp(ks,  j,i) += FTz_para + FTz_perp;
        e3_disp(ke+1,j,i)  = e3_disp(ks,j,i);
      } // i loop
    } // j loop
  } else {  // 3D update
    for (int k=ks; k<=ke+1; ++k) {
      for (int j=js; j<=je+1; ++j) {
      #pragma omp simd
        for (int i=is; i<=ie+1; ++i) {
          // emf.x
          therm_para = 0.0; therm_perp = 0.0; therm_wedge = 0.0;
          FTx_para = 0.0; FTx_perp = 0.0; FTx_wedge = 0.0;

          Bx = 0.25*(bc(IB1,k  ,j,i)+bc(IB1,k  ,j-1,i)
                  +bc(IB1,k-1,j,i)+bc(IB1,k-1,j-1,i));
          By = 0.5*(b.x2f(k,j,i) + b.x2f(k-1,j,i));
          Bz = 0.5*(b.x3f(k,j,i) + b.x3f(k,j-1,i));
          B = std::sqrt(Bx*Bx + By*By + Bz*Bz);

          if (B > TINY_NUMBER) {
            bx = Bx/B; by = By/B; bz = Bz/B;
          } else {
            bx = 0.0; by = 0.0; bz = 0.0;
          }

          Tedge = Te1(k,j,i);
          // dT/dx  x edge
          dTdx = (Te1(k,j,i+1) - Te1(k,j,i-1))
          /(pco_->dx1v(i-1) + pco_->dx1v(i));

          // dT/dy at x edge
          dTdy = (Te1(k,j+1,i) - Te1(k,j-1,i))
                / (pco_->h2v(i) * (pco_->dx2f(j-1) + pco_->dx2f(j)));

          // dT/dz at x edge
          dTdz = (Te1(k+1,j,i) - Te1(k-1,j,i))
            / ((pco_->dx3f(k-1) + pco_->dx3f(k)) * pco_->h31v(i) * pco_->h32f(j));

          if (use_TE_para) {
            therm_para = 0.25*(thermKappa(ThermElecProcess::kappa_TE_para,k,j,i)
                         + thermKappa(ThermElecProcess::kappa_TE_para,k-1,j,i)
                         + thermKappa(ThermElecProcess::kappa_TE_para,k,j-1,i)
                         + thermKappa(ThermElecProcess::kappa_TE_para,k-1,j-1,i));}
          if (use_TE_perp) {
            therm_perp = 0.25*(thermKappa(ThermElecProcess::kappa_TE_perp,k,j,i)
                         + thermKappa(ThermElecProcess::kappa_TE_perp,k-1,j,i)
                         +  thermKappa(ThermElecProcess::kappa_TE_perp,k,j-1,i)
                         +  thermKappa(ThermElecProcess::kappa_TE_perp,k-1,j-1,i));}
          if (use_TE_wedge) {
            therm_wedge = 0.25*(thermKappa(ThermElecProcess::kappa_TE_wedge,k,j,i)
                          + thermKappa(ThermElecProcess::kappa_TE_wedge,k-1,j,i)
                          + thermKappa(ThermElecProcess::kappa_TE_wedge,k,j-1,i)
                          + thermKappa(ThermElecProcess::kappa_TE_wedge,k-1,j-1,i));}

          // para = b (bdot_gradT)
          b_dot_gradT = bx*dTdx + by*dTdy + bz*dTdz;

          // Perp = gradT_perp = gradT - b(b dot gradT)
          Real gTpx = dTdx - bx*b_dot_gradT;

          // Wedge =  b x gradT
          Real bcTx = by*dTdz - bz*dTdy;

          // THERMOELECTRIC FORCE F_T
          FTx_para = -(1.0/Tedge)*(therm_para*bx*b_dot_gradT);
          FTx_perp = -(1.0/Tedge)*(therm_perp*gTpx);
          FTx_wedge = -(1.0/Tedge)*(therm_wedge*bcTx);

          if (pmb->phydro->hdif.kappa_aniso_neutral > 0.0 ||
              pmb->phydro->hdif.nu_aniso_neutral > 0.0) {
            ion_frac = 0.25*(pmb->phydro->hdif.ion_frac(k,j,i)
                       + pmb->phydro->hdif.ion_frac(k-1,j,i)
                       + pmb->phydro->hdif.ion_frac(k,j-1,i)
                       + pmb->phydro->hdif.ion_frac(k-1,j-1,i));
            FTx_para *= ion_frac; FTx_perp *= ion_frac; FTx_wedge *= ion_frac;
          }

          e1_diff(k,j,i) += FTx_wedge;
          e1_disp(k,j,i) += FTx_para + FTx_perp;

          // emf.y
          therm_para = therm_perp = therm_wedge = 0.0;
          FTy_para = 0.0; FTy_perp = 0.0; FTy_wedge =0.0;

          Bx = 0.5*(b.x1f(k,j,i) + b.x1f(k-1,j,i));
          By = 0.25*(bc(IB2,k  ,j,i)+bc(IB2,k  ,j,i-1)
                        +bc(IB2,k-1,j,i)+bc(IB2,k-1,j,i-1));
          Bz = 0.5*(b.x3f(k,j,i) + b.x3f(k,j,i-1));
          B = std::sqrt(Bx*Bx + By*By + Bz*Bz);

          if (B > TINY_NUMBER) {
            bx = Bx/B; by = By/B; bz = Bz/B;
          } else {
            bx = 0.0; by = 0.0; bz = 0.0;
          }

          Tedge = Te2(k,j,i);

          // dT/dx at y edge
          dTdx = (Te2(k,j,i+1) - Te2(k,j,i-1))
              / (pco_->dx1f(i-1) + pco_->dx1f(i));

          // dT/dy at y edge
          dTdy = (Te2(k,j+1,i) - Te2(k,j-1,i))
              / (pco_->h2f(i) * (pco_->dx2v(j-1) + pco_->dx2v(j)));

          // dT/dz at y edge
          dTdz = (Te2(k+1,j,i) - Te2(k-1,j,i))
                / ((pco_->dx3f(k-1) + pco_->dx3f(k)) * pco_->h31f(i) * pco_->h32v(j));

          if (use_TE_para) {
            therm_para = 0.25*(thermKappa(ThermElecProcess::kappa_TE_para,k,j,i)
                         + thermKappa(ThermElecProcess::kappa_TE_para,k-1,j,i)
                         + thermKappa(ThermElecProcess::kappa_TE_para,k,j,i-1)
                         + thermKappa(ThermElecProcess::kappa_TE_para,k-1,j,i-1));}
          if (use_TE_perp) {
            therm_perp = 0.25*(thermKappa(ThermElecProcess::kappa_TE_perp,k,j,i)
                         + thermKappa(ThermElecProcess::kappa_TE_perp,k-1,j,i)
                         +  thermKappa(ThermElecProcess::kappa_TE_perp,k,j,i-1)
                         +  thermKappa(ThermElecProcess::kappa_TE_perp,k-1,j,i-1));}
          if (use_TE_wedge) {
            therm_wedge = 0.25*(thermKappa(ThermElecProcess::kappa_TE_wedge,k,j,i)
                          + thermKappa(ThermElecProcess::kappa_TE_wedge,k-1,j,i)
                          + thermKappa(ThermElecProcess::kappa_TE_wedge,k,j,i-1)
                          + thermKappa(ThermElecProcess::kappa_TE_wedge,k-1,j,i-1));}

          // para: b (bdot_gradT)
          b_dot_gradT = bx*dTdx + by*dTdy + bz*dTdz;

          // Perp = gradT_perp = gradT - b(b dot gradT)
          Real gTpy = dTdy - by*b_dot_gradT;

          // Wedge =  b x gradT
          Real bcTy = bz*dTdx - bx*dTdz;

          // THERMOELECTRIC FORCE F_T
          FTy_para = -(1.0/Tedge)*(therm_para*by*b_dot_gradT);
          FTy_perp = -(1.0/Tedge)*(therm_perp*gTpy);
          FTy_wedge = -(1.0/Tedge)*(therm_wedge*bcTy);

          if (pmb->phydro->hdif.kappa_aniso_neutral > 0.0 ||
              pmb->phydro->hdif.nu_aniso_neutral > 0.0) {
              ion_frac = 0.25*(pmb->phydro->hdif.ion_frac(k,j,i)
                         + pmb->phydro->hdif.ion_frac(k-1,j,i)
                         + pmb->phydro->hdif.ion_frac(k,j,i-1)
                         + pmb->phydro->hdif.ion_frac(k-1,j,i-1));
              FTy_para *= ion_frac; FTy_perp *= ion_frac; FTy_wedge *= ion_frac;
            }

          e2_diff(k,j,i) += FTy_wedge;
          e2_disp(k,j,i) += FTy_para + FTy_perp;

          // emf.z
          therm_para = therm_perp = therm_wedge = 0.0;
          FTz_para = 0.0; FTz_perp = 0.0; FTz_wedge = 0.0;

          Bx = 0.5*(b.x1f(k,j,i) + b.x1f(k,j-1,i));
          By = 0.5*(b.x2f(k,j,i) + b.x2f(k,j,i-1));
          Bz = 0.25*(bc(IB3,k,j  ,i)+bc(IB3,k,j  ,i-1)
                +bc(IB3,k,j-1,i)+bc(IB3,k,j-1,i-1));
          B = std::sqrt(Bx*Bx + By*By + Bz*Bz);

          if (B > TINY_NUMBER) {
            bx = Bx/B; by = By/B; bz = Bz/B;
          } else {
            bx = 0.0; by = 0.0; bz = 0.0;
          }

          Tedge = Te3(k,j,i);

          // dT/dx at z edge
          dTdx = (Te3(k,j,i+1) - Te3(k,j,i-1))
              / (pco_->dx1f(i-1) + pco_->dx1f(i));

          // dT/dyat z edge
          dTdy = (Te3(k,j+1,i) - Te3(k,j-1,i))
            / (pco_->h2f(i) * (pco_->dx2f(j-1) + pco_->dx2f(j)));

          // dT/dz at z edge
          dTdz = (Te3(k+1,j,i) - Te3(k-1,j,i))
              / ((pco_->dx3v(k-1) + pco_->dx3v(k)) * pco_->h31f(i) * pco_->h32f(j));

          if (use_TE_para) {
            therm_para = 0.25*(thermKappa(ThermElecProcess::kappa_TE_para,k,j,i)
                          + thermKappa(ThermElecProcess::kappa_TE_para,k,j-1,i)
                          + thermKappa(ThermElecProcess::kappa_TE_para,k,j,i-1)
                          + thermKappa(ThermElecProcess::kappa_TE_para,k,j-1,i-1));}
          if (use_TE_perp) {
            therm_perp = 0.25*(thermKappa(ThermElecProcess::kappa_TE_perp,k,j,i)
                          + thermKappa(ThermElecProcess::kappa_TE_perp,k,j-1,i)
                          +  thermKappa(ThermElecProcess::kappa_TE_perp,k,j,i-1)
                          +  thermKappa(ThermElecProcess::kappa_TE_perp,k,j-1,i-1));}
          if (use_TE_wedge) {
            therm_wedge = 0.25*(thermKappa(ThermElecProcess::kappa_TE_wedge,k,j,i)
                          + thermKappa(ThermElecProcess::kappa_TE_wedge,k,j-1,i)
                          + thermKappa(ThermElecProcess::kappa_TE_wedge,k,j,i-1)
                          + thermKappa(ThermElecProcess::kappa_TE_wedge,k,j-1,i-1));}

          // para: b (bdot_gradT)
          b_dot_gradT = bx*dTdx + by*dTdy + bz*dTdz;

          // Perp = gradT_perp = gradT - b(b dot gradT)
          Real gTpz = dTdz- bz*b_dot_gradT;

          // Wedge =  b x gradT
          Real bcTz = bx*dTdy - by*dTdx;

          // THERMOELECTRIC FORCE F_T
          FTz_para = -(1.0/Tedge)*(therm_para*bz*b_dot_gradT);
          FTz_perp = -(1.0/Tedge)*(therm_perp*gTpz);
          FTz_wedge = -(1.0/Tedge)*(therm_wedge*bcTz);

          if (pmb->phydro->hdif.kappa_aniso_neutral > 0.0 ||
              pmb->phydro->hdif.nu_aniso_neutral > 0.0) {
            ion_frac = 0.25*(pmb->phydro->hdif.ion_frac(k,j,i)
                       + pmb->phydro->hdif.ion_frac(k,j-1,i)
                       + pmb->phydro->hdif.ion_frac(k,j,i-1)
                       + pmb->phydro->hdif.ion_frac(k,j-1,i-1));
            FTz_para *= ion_frac; FTz_perp *= ion_frac; FTz_wedge *= ion_frac;
          }
          e3_diff(k,j,i) += FTz_wedge;
          e3_disp(k,j,i) += FTz_para+FTz_perp;
        } // i loop
      } // j loop
    } // k loop
  } // end if f3
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void FieldDiffusion::PoyntingFlux
//! \brief Poynting flux from non-ideal MHD

void FieldDiffusion::PoyntingFlux(EdgeField &e, const AthenaArray<Real> &bc,
                                  FaceField &p_des) {
  MeshBlock *pmb = pmy_block;
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  AthenaArray<Real> &e1 = e.x1e, &e2 = e.x2e, &e3 = e.x3e,
                    &f1 = p_des.x1f, &f2 = p_des.x2f, &f3 = p_des.x3f;

  // 1D update:
  if (pmb->block_size.nx2 == 1) {
#pragma omp simd
    for (int i=is; i<=ie+1; ++i) {
      f1(ks,js,i) = -0.5*(bc(IB2,ks,js,i) + bc(IB2,ks,js,i-1))*e3(ks,js,i)
                    + 0.5*(bc(IB3,ks,js,i) + bc(IB3,ks,js,i-1))*e2(ks,js,i);
    }
    return;
  }

  // 2D update:
  if (pmb->block_size.nx3 == 1) {
    for (int j=js; j<=je+1; ++j) {
#pragma omp simd
      for (int i=is; i<=ie+1; ++i) {
        f1(ks,j,i) = -0.25*(bc(IB2,ks,j,i) + bc(IB2,ks,j,i-1))
                     *(e3(ks,j,i) + e3(ks,j+1,i))
                     + 0.5*(bc(IB3,ks,j,i) + bc(IB3,ks,j,i-1))*e2(ks,j,i);

        f2(ks,j,i) = -0.5*(bc(IB3,ks,j,i) + bc(IB3,ks,j-1,i))*e1(ks,j,i)
                     + 0.25*(bc(IB1,ks,j,i) + bc(IB1,ks,j-1,i))
                     *(e3(ks,j,i) + e3(ks,j,i+1));
      }
    }
    return;
  }

  // 3D update:
  for (int k=ks; k<=ke+1; ++k) {
    for (int j=js; j<=je+1; ++j) {
#pragma omp simd
      for (int i=is; i<=ie+1; ++i) {
        f1(k,j,i) = -0.25*(bc(IB2,k,j,i) + bc(IB2,k,j,i-1))
                    *(e3(k,j,i) + e3(k,j+1,i))
                    + 0.25*(bc(IB3,k,j,i) + bc(IB3,k,j,i-1))
                    *(e2(k,j,i) + e2(k+1,j,i));

        f2(k,j,i) = -0.25*(bc(IB3,k,j,i) + bc(IB3,k,j-1,i))
                    *(e1(k,j,i) + e1(k+1,j,i))
                    + 0.25*(bc(IB1,k,j,i) + bc(IB1,k,j-1,i))
                    *(e3(k,j,i) + e3(k,j,i+1));

        f3(k,j,i) = -0.25*(bc(IB1,k,j,i) + bc(IB1,k-1,j,i))
                    *(e2(k,j,i) + e2(k,j,i+1))
                    + 0.25*(bc(IB2,k,j,i) + bc(IB2,k-1,j,i))
                    *(e1(k,j,i) + e1(k,j+1,i));
      }
    }
  }
  return;
}
