#ifndef FIELD_FIELD_DIFFUSION_FIELD_DIFFUSION_HPP_
#define FIELD_FIELD_DIFFUSION_FIELD_DIFFUSION_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file field_diffusion.hpp
//! \brief defines class FieldDiffusion

// C headers

// C++ headers

// Athena++ headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"

// Forward declarations
class Field;
class Hydro;
class ParameterInput;
class Coordinates;

class FieldDiffusion;

//! currently must be free function for compatibility with user-defined fn via fn pointers
void ConstDiffusivity(FieldDiffusion *pfdif, MeshBlock *pmb, const AthenaArray<Real> &w,
                      const AthenaArray<Real> &bmag,
                      const int is, const int ie, const int js, const int je,
                      const int ks, const int ke);

//! \class FieldDiffusion
//! \brief data and functions for physical diffusion processes in the field

class FieldDiffusion {
 public:
  FieldDiffusion(MeshBlock *pmb, ParameterInput *pin);

  // data
  MeshBlock* pmy_block;
  bool field_diffusion_defined = false; // Any non-ideal effect
  bool field_oa_defined = false; // ohmic, ambi, FT_wedge
  bool field_disp_defined = false; // hall, FT_para, FT_perp
  bool use_TE_para = false;
  bool use_TE_perp = false;
  bool use_TE_wedge = false;
  bool debug_therm_conduct_only = false;
  bool debug_therm_force_only = false;
  Real eta_ohm, eta_hall, eta_ad;
  Real kappa_aniso_TE_para, kappa_aniso_TE_perp, kappa_aniso_TE_wedge;
  AthenaArray<Real> etaB; // 4-dim array, covering O/H/A altogether
  AthenaArray<Real> thermKappa; // 4-dim array for thermoelectric coefficients
  EdgeField e_oa;       // edge-centered electric field for ohmic, ambi, FT_w
  EdgeField e_h;        // edge-centered electric field for Hall, FT_Para, FT_perp
  FaceField pflux;      // face-centered energy (Poynting) flux for ohmic, ambi, FT_w
  FaceField pflux_disp; // face-centered energy (Poynting) flux for Hall, FT_Para, FT_perp

  // Used in thermoelectric effects
  AthenaArray<Real> jfx, jfy, jfz; // interface current density
  AthenaArray<Real> jcc;     // cell-centered current density

  AthenaArray<Real> teflx[3]; // thermoelectric conduction flux wedge
  AthenaArray<Real> teflx_disp[3]; // thermoelectric conduction flux para+perp

  // For thermoelectric force
  AthenaArray<Real> Tcc;   // cell-centered T = p/rho
  AthenaArray<Real> Te1;   // T averaged to x1-edges (i, j-1/2, k-1/2)
  AthenaArray<Real> Te2;   // T averaged to x2-edges (i-1/2, j, k-1/2)
  AthenaArray<Real> Te3;   // T averaged to x3-edges (i-1/2, j-1/2, k)

  // array indices for magnetic diffusion types
  // should not be scoped (C++11) since enumerators are only used as "int" to index arrays
  enum DiffProcess {ohmic=0, hall=1, ambipolar=2};
  enum ThermElecProcess {kappa_TE_para = 0, kappa_TE_perp = 1,
                         kappa_TE_wedge = 2};
  // TODO(felker) Unlike HydroDiffusion::DiffProcess, not using optional "unscoped enum
  // name" qualifier when referencing the enumerators in other files. Be consistent

  // alternative to unscoped (possibly anonymous) for int constants:
  // static constexpr int n_ohmic = 0;
  // static constexpr int n_hall = 1;
  // static constexpr int n_ambi = 2;

  // functions
  void CalcDiffusionEMF(FaceField &bi, const AthenaArray<Real> &bc, EdgeField &e);
  void CalcEdgeTemperature(const AthenaArray<Real> &w, AthenaArray<Real> &Tcc,
                           AthenaArray<Real> &Te1, AthenaArray<Real> &Te2,
                           AthenaArray<Real> &Te3);
  void AddEMF(const EdgeField &e_src, EdgeField &e_des);
  void ClearEMF(EdgeField &e);
  void CalcCurrent(FaceField &b);
  void AddPoyntingFlux (FaceField &p_src);

  // functions to calculate diffusivities and timesteps
  void SetDiffusivity(const AthenaArray<Real> &w, const AthenaArray<Real> &bcc);
  void NewDiffusionDt(Real &dt_oa, Real &dt_h, Real &dt_OAH,
                                      Real &dt_TE_para, AthenaArray<Real> &inv_dt_OAH,
                                      AthenaArray<Real> &inv_dt_TE,
                                      AthenaArray<Real> &inv_dt_TE_hyp);
  Real TEEigenMax(const Real kJpara, const Real kJperp, const Real kJwedge,
                  const Real HII, const Real E_int);
  Real TEParaEigenMax(const Real gm1, const Real kappa_0, const Real d_kappa,
    const Real eta_oa, const Real kappa_Jw, const Real E_int);
  Real TEHypEigenMax(const Real etaH, const Real HII,
    const Real kJpara, const Real kJperp, const Real E_int);

  // non-ideal MHD EMFs
  void OhmicEMF(const FaceField &b, const AthenaArray<Real> &bc, EdgeField &e);
  void HallEMF(const FaceField &b, const AthenaArray<Real> &bc, EdgeField &e);
  void AmbipolarEMF(const FaceField &b, const AthenaArray<Real> &bc, EdgeField &e);
  void ThermoElectricConduction(MeshBlock *pmb, const FaceField &b,
                                const AthenaArray<Real> &bc,
                                AthenaArray<Real> *flx, AthenaArray<Real> *flx_disp);
  void ThermoElectricForce(MeshBlock *pmb, const FaceField &b,
                          const AthenaArray<Real> &bc,
                          EdgeField &e_para, EdgeField &e_disp);

  // functions for energy flux
  void PoyntingFlux(EdgeField &e, const AthenaArray<Real> &bc, FaceField &p_des);

  AthenaArray<Real> inv_dt_OAH; // 1/dt_OAH for TE unsplit dt
  AthenaArray<Real> inv_dt_TE;  // 1/dt_TE per cell for TE unsplit dt
  AthenaArray<Real> inv_dt_TE_hyp; // 1/dt_TE_hyp for hall + peltier/seebeck + wedge therm

 private:
  AthenaArray<Real> bmag_; // B field strength
  EdgeField jedge_;       // curl of B

  FieldDiffusionCoeffFunc CalcMagDiffCoeff_; // calculate magnetic diffusivities

  AthenaArray<Real> face_area_, face_area_p1_, edge_length_, edge_length_m1_;
  AthenaArray<Real>  cell_volume_;
  AthenaArray<Real> dx1_, dx2_, dx3_, len_;
  AthenaArray<Real> eta_tot_;
};
#endif // FIELD_FIELD_DIFFUSION_FIELD_DIFFUSION_HPP_
