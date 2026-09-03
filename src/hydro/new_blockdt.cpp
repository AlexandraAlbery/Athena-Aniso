//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file new_blockdt.cpp
//! \brief computes timestep using CFL condition on a MEshBlock

// C headers

// C++ headers
#include <algorithm>  // min()
#include <cmath>      // abs(), sqrt()
#include <limits>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../cr/cr.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../field/field_diffusion/field_diffusion.hpp"
#include "../mesh/mesh.hpp"
#include "../nr_radiation/implicit/radiation_implicit.hpp"
#include "../nr_radiation/radiation.hpp"
#include "../orbital_advection/orbital_advection.hpp"
#include "../scalars/scalars.hpp"
#include "hydro.hpp"
#include "hydro_diffusion/hydro_diffusion.hpp"

// MPI/OpenMP header
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

//----------------------------------------------------------------------------------------
//! \fn void Hydro::NewBlockTimeStep()
//! \brief calculate the minimum timestep within a MeshBlock

void Hydro::NewBlockTimeStep() {
  MeshBlock *pmb = pmy_block;
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;
  int nc1 = pmb->ncells1; int nc2 = pmb->ncells2; int nc3 = pmb->ncells3;
  AthenaArray<Real> &w = pmb->phydro->w;
  // hyperbolic timestep constraint in each (x1-slice) cell along coordinate direction:
  AthenaArray<Real> &dt1 = dt1_, &dt2 = dt2_, &dt3 = dt3_;  // (x1 slices)
  Real wi[NWAVE];

  Real real_max = std::numeric_limits<Real>::max();
  Real min_dt = real_max;
  // Note, "dt_hyperbolic" currently refers to the dt limit imposed by evoluiton of the
  // ideal hydro or MHD fluid by the main integrator (even if not strictly hyperbolic)
  // and dispersive non-ideal terms (thermal wedge term, gyro-viscoisty, Hall, and
  // Peltier/Seebeck).
  Real min_dt_hyperbolic  = real_max;
  // TODO(felker): consider renaming dt_hyperbolic after general execution model is
  // implemented and flexibility from #247 (zero fluid configurations) is
  // addressed. dt_hydro, dt_main (inaccurate since "dt" is actually main), dt_MHD?
  Real min_dt_parabolic  = real_max;
  Real min_dt_user  = real_max;

  inv_dt_ideal.ZeroClear();

  Real cspeed = 0.0;
  if(NR_RADIATION_ENABLED)
    cspeed = pmb->pnrrad->reduced_c;
  if(CR_ENABLED)
    cspeed = std::max(cspeed,pmb->pcr->vmax);

  // TODO(felker): skip this next loop if pm->fluid_setup == FluidFormulation::disabled
  FluidFormulation fluid_status = pmb->pmy_mesh->fluid_setup;
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      // Obtain cell widths
      pmb->pcoord->CenterWidth1(k, j, is, ie, dt1);
      pmb->pcoord->CenterWidth2(k, j, is, ie, dt2);
      pmb->pcoord->CenterWidth3(k, j, is, ie, dt3);

      // Newtonian case: divide cell widths by maximum characteristic speed
      if (!RELATIVISTIC_DYNAMICS) {
#pragma ivdep
        for (int i=is; i<=ie; ++i) {
          wi[IDN] = w(IDN,k,j,i);
          wi[IVX] = w(IVX,k,j,i);
          wi[IVY] = w(IVY,k,j,i);
          wi[IVZ] = w(IVZ,k,j,i);
          if (NON_BAROTROPIC_EOS) wi[IPR] = w(IPR,k,j,i);
          if (fluid_status == FluidFormulation::evolve) {
            if (MAGNETIC_FIELDS_ENABLED) {
              AthenaArray<Real> &bcc = pmb->pfield->bcc, &b_x1f = pmb->pfield->b.x1f,
                              &b_x2f = pmb->pfield->b.x2f, &b_x3f = pmb->pfield->b.x3f;
              Real bx = bcc(IB1,k,j,i) + std::abs(b_x1f(k,j,i) - bcc(IB1,k,j,i));
              wi[IBY] = bcc(IB2,k,j,i);
              wi[IBZ] = bcc(IB3,k,j,i);
              Real cf = pmb->peos->FastMagnetosonicSpeed(wi,bx);
              Real speed = std::max(cspeed,(std::abs(wi[IVX]) + cf));
              dt1(i) /= (speed);

              wi[IBY] = bcc(IB3,k,j,i);
              wi[IBZ] = bcc(IB1,k,j,i);
              bx = bcc(IB2,k,j,i) + std::abs(b_x2f(k,j,i) - bcc(IB2,k,j,i));
              cf = pmb->peos->FastMagnetosonicSpeed(wi,bx);
              speed = std::max(cspeed,(std::abs(wi[IVY]) + cf));
              dt2(i) /= (speed);

              wi[IBY] = bcc(IB1,k,j,i);
              wi[IBZ] = bcc(IB2,k,j,i);
              bx = bcc(IB3,k,j,i) + std::abs(b_x3f(k,j,i) - bcc(IB3,k,j,i));
              cf = pmb->peos->FastMagnetosonicSpeed(wi,bx);
              speed = std::max(cspeed,(std::abs(wi[IVZ]) + cf));
              dt3(i) /= (speed);

              Real dtc = dt1(i);
              if (pmb->block_size.nx2 > 1) dtc = std::min(dtc, dt2(i));
              if (pmb->block_size.nx3 > 1) dtc = std::min(dtc, dt3(i));
              inv_dt_ideal(k,j,i) = 1.0/dtc;
            } else {
              Real cs = pmb->peos->SoundSpeed(wi);
              Real speed1 = std::max(cspeed, (std::abs(wi[IVX]) + cs));
              Real speed2 = std::max(cspeed, (std::abs(wi[IVY]) + cs));
              Real speed3 = std::max(cspeed, (std::abs(wi[IVZ]) + cs));
              dt1(i) /= (speed1);
              dt2(i) /= (speed2);
              dt3(i) /= (speed3);

              Real dtc = dt1(i);
              if (pmb->block_size.nx2 > 1) dtc = std::min(dtc, dt2(i));
              if (pmb->block_size.nx3 > 1) dtc = std::min(dtc, dt3(i));
              inv_dt_ideal(k,j,i) = 1.0/dtc;
            }
          } else { // FluidFormulation::background or disabled. Assume scalar advection:
            dt1(i) /= (std::abs(wi[IVX]));
            dt2(i) /= (std::abs(wi[IVY]));
            dt3(i) /= (std::abs(wi[IVZ]));

            Real dtc = dt1(i);
            if (pmb->block_size.nx2 > 1) dtc = std::min(dtc, dt2(i));
            if (pmb->block_size.nx3 > 1) dtc = std::min(dtc, dt3(i));
            inv_dt_ideal(k,j,i) = 1.0/dtc;
          }
        }
      }

      // Normalising by ndim here puts the ideal term on the same footing as the
      // diffusion bounds, whose fac factors already carry 1/(2*ndim).
      // cfl_number = 1 then means at the stability limit for every term
      int ndim = 1;
      if (pmb->block_size.nx2 > 1) ndim++;
      if (pmb->block_size.nx3 > 1) ndim++;
      if (ndim > 1) {
        for (int i=is; i<=ie; ++i) {
          dt1(i) /= ndim; dt2(i) /= ndim; dt3(i) /= ndim;
          inv_dt_ideal(k,j,i) *= static_cast<Real>(ndim);
        }
      }

      // SR case: do nothing (assume maximum characteristic is c = 1)
      // GR case: divide cell widths by coordinate speed of light (not necessarily unity)
      if (GENERAL_RELATIVITY) {
        pmb->pcoord->CellMetric(k, j, is, ie, g_, gi_);
        for (int i=is; i<=ie; ++i) {
          Real speed1 = -(std::sqrt(SQR(gi_(I01,i)) - gi_(I00,i) * gi_(I11,i))
              + std::abs(gi_(I01,i))) / gi_(I00,i);
          Real speed2 = -(std::sqrt(SQR(gi_(I02,i)) - gi_(I00,i) * gi_(I22,i))
              + std::abs(gi_(I02,i))) / gi_(I00,i);
          Real speed3 = -(std::sqrt(SQR(gi_(I03,i)) - gi_(I00,i) * gi_(I33,i))
              + std::abs(gi_(I03,i))) / gi_(I00,i);
          dt1(i) /= speed1;
          dt2(i) /= speed2;
          dt3(i) /= speed3;
        }
      }

      // compute minimum of (v1 +/- C)
      for (int i=is; i<=ie; ++i) {
        const Real& dt_1 = dt1(i);
        min_dt_hyperbolic = std::min(min_dt_hyperbolic, dt_1);
      }

      // if grid is 2D/3D, compute minimum of (v2 +/- C)
      if (pmb->block_size.nx2 > 1) {
        for (int i=is; i<=ie; ++i) {
          const Real& dt_2 = dt2(i);
          min_dt_hyperbolic = std::min(min_dt_hyperbolic, dt_2);
        }
      }

      // if grid is 3D, compute minimum of (v3 +/- C)
      if (pmb->block_size.nx3 > 1) {
        for (int i=is; i<=ie; ++i) {
          const Real& dt_3 = dt3(i);
          min_dt_hyperbolic = std::min(min_dt_hyperbolic, dt_3);
        }
      }
    }
  }

  // calculate the timestep limited by the diffusion processes
  Real min_dt_vis = real_max, min_dt_cnd = real_max;
  Real min_dt_vis_disp = real_max, min_dt_cnd_disp = real_max;
  Real min_dt_oa = real_max, min_dt_hall = real_max, min_dt_OAH = real_max;
  Real min_dt_TE = real_max; Real min_dt_TE_para = real_max;
  Real min_dt_TE_hyp = real_max;
  FieldDiffusion &fdif_ = pmb->pfield->fdif;

  if (hdif.hydro_diffusion_defined) {
    hdif.NewDiffusionDt(min_dt_vis, min_dt_cnd, min_dt_vis_disp, min_dt_cnd_disp,
                        hdif.inv_dt_cnd, hdif.inv_dt_visc, hdif.inv_dt_cnd_disp,
                        hdif.inv_dt_vis_disp);
  }

  if (MAGNETIC_FIELDS_ENABLED &&
      fdif_.field_diffusion_defined) {
      fdif_.NewDiffusionDt(min_dt_oa, min_dt_hall, min_dt_OAH, min_dt_TE_para,
                           fdif_.inv_dt_OAH, fdif_.inv_dt_TE, fdif_.inv_dt_TE_hyp);
  }


  // Diffusion + dispersion with STS integration
  const bool hd_on = hdif.hydro_diffusion_defined;
  const bool fd_on = MAGNETIC_FIELDS_ENABLED && fdif_.field_diffusion_defined;
  const bool te_active = MAGNETIC_FIELDS_ENABLED
                          && pmb->pfield->fdif.field_diffusion_defined
                          && (pmb->pfield->fdif.use_TE_para ||
                              pmb->pfield->fdif.use_TE_perp ||
                              pmb->pfield->fdif.use_TE_wedge);
  const bool disp_active =
        (hd_on && (hdif.cnd_disp_defined || hdif.visc_disp_defined)) ||
        (fd_on && (fdif_.eta_hall > 0.0 || fdif_.use_TE_para || fdif_.use_TE_perp));

  if (STS_ENABLED) {
    // Parabolic: therm, visc, oa+ft_w
    min_dt_parabolic = std::min({min_dt_parabolic, min_dt_vis, min_dt_cnd,
                                 min_dt_oa, min_dt_TE_para});
    // Hyperbolic: ideal + right-ledeuc + hall/peltier/seebeck + gyro-visc
    if (disp_active) {
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
          for (int i=is; i<=ie; ++i) {
            Real inv = inv_dt_ideal(k,j,i);
            Real inv_TE = 0.0;
            Real inv_vis = 0.0;
            if (hd_on) {
              inv_TE +=  hdif.inv_dt_cnd_disp(k,j,i);
              inv_vis = hdif.inv_dt_vis_disp(k,j,i);
            }
            if (fd_on) inv_TE = fdif_.inv_dt_TE_hyp(k,j,i);
            inv += std::max<Real>(inv_vis, inv_TE);
            if (inv > 0.0)
              min_dt_hyperbolic = std::min<Real>(min_dt_hyperbolic, 1.0/inv);
          }
        }
      }
    }
  } else {
    // unsplit: ideal+ diffusion + dispersion
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          Real inv_tot = inv_dt_ideal(k,j,i);
          Real inv_vis_tot = 0.0;
          Real inv_TE_tot = 0.0;
          if (hd_on) {
            inv_TE_tot += hdif.inv_dt_cnd(k,j,i);
            inv_vis_tot = hdif.inv_dt_visc(k,j,i);
          }
          if (fd_on) {
            inv_TE_tot += fdif_.inv_dt_OAH(k,j,i);
            if (te_active) inv_TE_tot += fdif_.inv_dt_TE(k,j,i);
          }
          inv_tot += std::max<Real>(inv_vis_tot, inv_TE_tot);
          if (inv_tot > 0.0)
            min_dt_hyperbolic = std::min<Real>(min_dt_hyperbolic, 1.0/inv_tot);
        }
      }
    }
  }

  if (NSCALARS > 0 && pmb->pscalars->scalar_diffusion_defined) {
    Real min_dt_scalar_diff = pmb->pscalars->NewDiffusionDt();
    min_dt_parabolic = std::min(min_dt_parabolic, min_dt_scalar_diff);
  } // passive scalar diffusion

  // scale the theoretical stability limit by a safety factor = the hyperbolic CFL limit
  // (user-selected or automaticlaly enforced). May add independent parameter "cfl_diff"
  // in the future (with default = cfl_number).
  min_dt_hyperbolic *= pmb->pmy_mesh->cfl_number;
  min_dt_parabolic *= pmb->pmy_mesh->cfl_number;

  if (IM_RADIATION_ENABLED) {
    min_dt_hyperbolic *= pmb->pmy_mesh->pimrad->cfl_rad;
    min_dt_parabolic  *= pmb->pmy_mesh->pimrad->cfl_rad;
  }

  // For orbital advection, give a restriction on dt_hyperbolic.
  if (pmb->porb->orbital_advection_active) {
    Real min_dt_orb = pmb->porb->NewOrbitalAdvectionDt();
    min_dt_hyperbolic = std::min(min_dt_hyperbolic, min_dt_orb);
  }

  // set main integrator timestep as the minimum of the appropriate timestep constraints:
  // hyperbolic: (skip if fluid is nonexistent or frozen)
  min_dt = std::min(min_dt, min_dt_hyperbolic);
  min_dt = std::min(min_dt, min_dt_parabolic);
  // user:
  if (UserTimeStep_ != nullptr) {
    min_dt_user = UserTimeStep_(pmb);
    min_dt = std::min(min_dt, min_dt_user);
  }
  pmb->new_block_dt_ = min_dt;
  pmb->new_block_dt_hyperbolic_ = min_dt_hyperbolic;
  pmb->new_block_dt_parabolic_ = min_dt_parabolic;
  pmb->new_block_dt_user_ = min_dt_user;

  return;
}
