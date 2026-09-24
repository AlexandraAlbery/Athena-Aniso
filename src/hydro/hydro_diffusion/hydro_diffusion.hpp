#ifndef HYDRO_HYDRO_DIFFUSION_HYDRO_DIFFUSION_HPP_
#define HYDRO_HYDRO_DIFFUSION_HYDRO_DIFFUSION_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file hydro_diffusion.hpp
//! \brief defines class HydroDiffusion
//!
//! Contains data and functions that implement the diffusion processes

// C headers

// C++ headers

// Athena++ headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"

// Forward declarations
class Hydro;
class ParameterInput;
class Coordinates;
class HydroDiffusion;

// currently must be free functions for compatibility with user-defined fn via fn pointers
void ConstViscosity(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &w,
                    const AthenaArray<Real> &bc,
                    int is, int ie, int js, int je, int ks, int ke);

void ConstConduction(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &w,
                     const AthenaArray<Real> &bc,
                     int is, int ie, int js, int je, int ks, int ke);

//! \class HydroDiffusion
//! \brief data and functions for physical diffusion processes in the hydro

class HydroDiffusion {
 public:
  HydroDiffusion(Hydro *phyd, ParameterInput *pin);

  // data
  bool hydro_diffusion_defined = false;
  bool hydro_para_defined = false; // any STS terms
  bool hydro_disp_defined = false; // any non-STS-able (wedge) hydro term is on
  bool conduction_on = false;
  bool viscosity_on = false;
  bool aniso_conduction = false;
  bool aniso_viscosity = false;
  bool calc_para_ = true;    // build STS-able (iso/para/perp) fluxes on this call
  bool calc_disp_ = true;    // build non-STS-able (wedge) fluxes on this call

  bool visc_aniso_para   = false;   // Pi0/Pi1/Pi2
  bool visc_para_defined = false;   // anything that writes visflx under STS
  bool visc_disp_defined = false;   // Pi3/Pi4
  bool cnd_aniso_para    = false;   // kappa neutral/para/perp
  bool cnd_para_defined  = false;   // anything that writes cndflx under STS
  bool cnd_disp_defined  = false;   // kappa wedge;

  bool limit_saturated_conduction = true; // Apply saturation limiting for conduction

  // viscosity
  Real nu_iso, nu_aniso_neutral, nu_aniso_para, nu_aniso_perp1, nu_aniso_perp2,
  nu_aniso_wedge1, nu_aniso_wedge2, zeta_bulk; // viscosity coeffs
  AthenaArray<Real> visflx[3]; // viscous stress tensor
  AthenaArray<Real> visflx_disp[3]; // gyro-viscous stress tensor
  AthenaArray<Real> nu; // viscosity array

  // conduction
  Real kappa_iso, kappa_aniso_neutral, kappa_aniso_para,
  kappa_aniso_perp, kappa_aniso_wedge; // thermal conduction coeffs
// Real kappa_aniso put this back if I break something
  AthenaArray<Real> cndflx[3]; // thermal stress tensor
  AthenaArray<Real> cndflx_disp[3]; // Right-Leduc stress tensor
  AthenaArray<Real> kappa; // conduction array

  // state fractions
  AthenaArray<Real> ion_frac; // ionised fraction
  Real ion_frac_const; // constant partial ionisation

  // array indices for hydro diffusion (conduction & viscosity) variants: directionality
  // should not be scoped (C++11) since enumerators are only used as "int" to index arrays
  enum DiffProcess {iso=0, kappa_neutral = 1, kappa_para = 2, kappa_perp = 3,
    kappa_wedge = 4, nu_neutral = 1, nu_para = 2, nu_perp1 = 3,
     nu_perp2 = 4, nu_wedge1 = 5, nu_wedge2 = 6, zeta = 7};

  // functions
  // iprim is primitives in the inertial system
  void CalcDiffusionFlux(const AthenaArray<Real> &prim, const AthenaArray<Real> &iprim,
                         const AthenaArray<Real> &bcc, bool calc_para, bool calc_perp);
  // TODO(felker): Rename+move out of this class. Confusing w/ Hydro::AddDiffusionFluxes()
  // See note in hydro_diffusion.cpp.
  void AddDiffusionFlux(AthenaArray<Real> *flx_src, AthenaArray<Real> *flx_des);
  void AddDiffusionEnergyFlux(AthenaArray<Real> *flux_src, AthenaArray<Real> *flux_des);
  void ClearFlux(AthenaArray<Real> *flx);
  void SetDiffusivity(const AthenaArray<Real> &w, const AthenaArray<Real> &bc);
  void NewDiffusionDt(Real &dt_vis, Real &dt_cnd, Real &dt_vis_disp, Real &dt_cnd_disp,
                      AthenaArray<Real> &inv_dt_cnd, AthenaArray<Real> &inv_dt_visc,
                      AthenaArray<Real> &inv_dt_cnd_disp,
                      AthenaArray<Real> &inv_dt_vis_disp);

  // viscosity
  void ViscousFluxIso(const AthenaArray<Real> &p, const AthenaArray<Real> &p_i,
                      AthenaArray<Real> *flx);
  void ViscousFluxAniso(const AthenaArray<Real> &p, const AthenaArray<Real> &p_i,
                        AthenaArray<Real> *flx, AthenaArray<Real> *flx_disp);
  void ViscousFluxBulk(const AthenaArray<Real> &p, const AthenaArray<Real> &p_i,
                        AthenaArray<Real> *flx);
  inline Real ViscAnisoEigen(const Real u, const Real HII, const Real nu_neu,
                      const Real nu_par, const Real nu_p1, const Real nu_p2,
                      const Real zr);
  Real ViscAnisoEigenMax(const Real HII, const Real nu_neu, const Real nu_par,
                         const Real nu_p1, const Real nu_p2, const Real zr);
  Real GyroViscEigenMax(const Real HII, const Real nu_w1, const Real nu_w2);

  // thermal conduction
  void ThermalFluxIso(const AthenaArray<Real> &p, AthenaArray<Real> *flx);
  void ThermalFluxAniso(const AthenaArray<Real> &p, AthenaArray<Real> *flx,
                        AthenaArray<Real> *flx_disp);

  AthenaArray<Real> inv_dt_vis_disp; // 1/dt_nu_wedge needed for STS hydro
  AthenaArray<Real> inv_dt_cnd_disp; // 1/dt_κ_wedge needed for TE + STS hydro
  AthenaArray<Real> inv_dt_cnd; // 1/dt_κ_par_perp + 1/dt_κ_wedge
  AthenaArray<Real> inv_dt_visc; // 1/dt_nu_par_perp + 1/dt_nu_wedge

 private:
  Hydro *pmy_hydro_;  // ptr to Hydro containing this HydroDiffusion
  MeshBlock *pmb_;    // ptr to meshblock containing this HydroDiffusion
  Coordinates *pco_;  // ptr to coordinates class
  AthenaArray<Real> div_vel_; // divergence of velocity
  AthenaArray<Real> x1area_, x2area_, x2area_p1_, x3area_, x3area_p1_;
  AthenaArray<Real> vol_;
  AthenaArray<Real> fx_, fy_, fz_;
  AthenaArray<Real> dx1_, dx2_, dx3_;
  AthenaArray<Real> nu_tot_, kappa_tot_, nu_tot_disp_, kappa_tot_disp_;

  // functions pointer to calculate spatial dependent coefficients
  ViscosityCoeffFunc CalcViscCoeff_;
  ConductionCoeffFunc CalcCondCoeff_;

  // auxiliary functions to calculate viscous flux
  void DivVelocity(const AthenaArray<Real> &prim, AthenaArray<Real> &divv);
  void FaceXdx(const int k, const int j, const int il, const int iu,
               const AthenaArray<Real> &prim, AthenaArray<Real> &len);
  void FaceXdy(const int k, const int j, const int il, const int iu,
               const AthenaArray<Real> &prim, AthenaArray<Real> &len);
  void FaceXdz(const int k, const int j, const int il, const int iu,
               const AthenaArray<Real> &prim, AthenaArray<Real> &len);
  void FaceYdx(const int k, const int j, const int il, const int iu,
               const AthenaArray<Real> &prim, AthenaArray<Real> &len);
  void FaceYdy(const int k, const int j, const int il, const int iu,
               const AthenaArray<Real> &prim, AthenaArray<Real> &len);
  void FaceYdz(const int k, const int j, const int il, const int iu,
               const AthenaArray<Real> &prim, AthenaArray<Real> &len);
  void FaceZdx(const int k, const int j, const int il, const int iu,
               const AthenaArray<Real> &prim, AthenaArray<Real> &len);
  void FaceZdy(const int k, const int j, const int il, const int iu,
               const AthenaArray<Real> &prim, AthenaArray<Real> &len);
  void FaceZdz(const int k, const int j, const int il, const int iu,
               const AthenaArray<Real> &prim, AthenaArray<Real> &len);
};
#endif // HYDRO_HYDRO_DIFFUSION_HYDRO_DIFFUSION_HPP_
