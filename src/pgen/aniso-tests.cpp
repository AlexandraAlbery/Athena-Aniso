//======================================================================================
/* Athena++ astrophysical MHD code
 * Copyright (C) 2014 James M. Stone  <jmstone@princeton.edu>
 *
 * This program is free software: you can redistribute and/or modify it under the terms
 * of the GNU General Public License (GPL) as published by the Free Software Foundation,
 * either version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
 * PARTICULAR PURPOSE.  See the GNU General Public License for more details.
 *
 * You should have received a copy of GNU GPL in the file LICENSE included in the code
 * distribution.  If not see <http://www.gnu.org/licenses/>.
 *====================================================================================*/

// C Headers
#include <unistd.h>

// C++ headers
#include <algorithm>  // min, max
#include <cmath>    // sqrt()
#include <cstdio>   // fopen(), fprintf(), freopen()
#include <ios>
#include <iostream>   // endl
#include <random>
#include <sstream>
#include <stdexcept>  // runtime_error
#include <string>   // c_str()
#include <vector>   // vector type

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/bvals.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../fft/athena_fft.hpp"
#include "../fft/turbulence.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../outputs/io_wrapper.hpp"
#include "../outputs/outputs.hpp"
#include "../parameter_input.hpp"
#include "../reconstruct/reconstruction.hpp"
#include "../scalars/scalars.hpp"
#include "../utils/utils.hpp"

// INPUT FILE: ../../inputs/mhd/aniso-tests/athinput.aniso-tests

// Global variables
namespace{
  Real P_floor = 0; Real rho_floor = 0; Real nx1 = 0; Real nx2 = 0; Real nx3 = 0;
  Real xmin = 0; Real ymin = 0; Real zmin = 0.0;
  Real xmax = 0.0; Real ymax = 0.0; Real zmax = 0.0;
  Real CFL = 0; Real rho_0 = 0; Real P_0 = 0; Real Bx_0 = 0; Real By_0 = 0; Real Bz_0 = 0;
  Real vx_0 = 0; Real vy_0 = 0; Real vz_0 = 0; Real Gamma = 0; Real gm1 = 0;
  Real kappa_neu = 0; Real kappa_para = 0; Real kappa_perp = 0; Real kappa_wedge = 0;
  Real nu_neu = 0; Real nu_para = 0; Real nu_perp1 = 0; Real nu_perp2 = 0;
  Real nu_wedge1 = 0; Real nu_wedge2 = 0; Real zeta_bulk = 0; Real ion_frac= 0;
  Real eta_o = 0; Real eta_ad = 0; Real eta_h = 0;
  Real kappa_TE_para = 0; Real kappa_TE_perp = 0; Real kappa_TE_wedge = 0;
  bool debug_therm_conduct_only = false; bool debug_therm_force_only = false;
  int problem_ID = 0; // tracks which test to run. called in MeshBlock::UserMeshBlockData

  // case 0
  Real T0 = 0.0;
  int  nx  = 0; int  ny  = 0; int  nz  = 0;
  Real kx  = 0.0; Real ky  = 0.0; Real kz  = 0.0;

  // case 1 + 5
  Real v0   = 0.0;
  Real a1 = 0.0;   // k_hat amp
  Real a2 = 0.0;   // b_hat amp
  Real a3 = 0.0;   // k_hat x b_hat amp
  Real khat_x = 0.0;  Real khat_y = 0.0; Real khat_z = 0.0;
  Real bhat_x = 0.0;  Real bhat_y = 0.0; Real bhat_z = 0.0;
  Real ec_x = 0.0; Real ec_y = 0.0; Real ec_z = 0.0; // k_hat x b_hat
  Real B0_mag = 0.0;

  // case 0 + 3 + 4 + 6
  Real dT_amp;
  // case 2 + 3
  Real dB_amp = 0; // amplitude of the field perturbation

  // case 4
  Real kB = 0; // Beltrami wavenumber

  // Array index for hdf5 output
  const int out_T      = 0;  // temperature
  const int out_sound    = 1;  // adiabatic sound speed
  const int out_alfven   = 2;  // alfven speed
  const int out_normAlfven = 3;  // normal (x-face) alfven speed
  const int out_slowMag  = 4;  // slow magnetosonic speed
  const int out_fastMag  = 5;  // fast magnetosonic speed
  const int out_whist    = 6;  // hall effect whistler wave
  const int out_ioncyc   = 7;  // hall effect ion cyclotron speed
  const int out_TEwave   = 8;  // Peltier/Seebeck speed
  const int out_TEWwave  = 9;  // Peltier/Seebeck + Hall wavespeed
  const int out_GVwave   = 10; // gyro-viscous wave speed
  const int out_RLspeed  = 11; // righi-leduc advection speed

  /*
  const int out_dt_ideal   = 12; // ideal flow
  const int out_dt_vis_para  = 13; // parabolic viscosity
  const int out_dt_cnd_para  = 14; // parabolic thermal conduction
  const int out_dt_vis_hyp   = 15; // hyperbolic (gyro) viscosity
  const int out_dt_cnd_hyp   = 16; // hyperbolic (RL) thermal conduction
  const int out_dt_cnd_tot   = 17; // total thermal conduction
  const int out_dt_vis_tot   = 18; // total viscosity
  const int out_dt_oah     = 19; // ohmic + ambipolar + hall (no TE)
  const int out_dt_TE    = 20; // unsplit isolated TE (P/S + N/E)
  const int out_dt_TE_hyp  = 21; // Hall+P/S
  const int out_dt_total   = 22; // Actual timestep
  */
  const int out_WP       = 12;

  // check for blowups in unstable sims
  bool nonfinite_found = false;
  Real dt_largest = 0.0;
}; // namespace

// Initialise functions
void GetTemperature(MeshBlock *pmb);
void GetWaveSpeeds(MeshBlock *pmb);
//void GetEntropyProduction(MeshBlock *pmb);
void CalculateThermalKappa(MeshBlock *pmb);
void CalculateViscousNu(MeshBlock *pmb);
void GetTimeSteps(MeshBlock *pmb);
void GetBhat(MeshBlock *pmb, int k, int j, int i, Real &bx, Real &by, Real &bz);
void CalculateThermalKappa(HydroDiffusion *phdif, MeshBlock *pmb,
  const AthenaArray<Real> &prim,const AthenaArray<Real> &bcc,
  int is, int ie, int js, int je,int ks, int ke);
void CalculateViscousNu(HydroDiffusion *phdif, MeshBlock *pmb,
  const AthenaArray<Real> &prim,const AthenaArray<Real> &bcc,
  int is, int ie, int js, int je,int ks, int ke);
void CalculateMagDiff(FieldDiffusion *pfdif, MeshBlock *pmb,
  const AthenaArray<double> &w, const AthenaArray<double> &bmag,
  int is, int ie, int js, int je, int ks, int ke);
void CalculateWellPosedness(MeshBlock *pmb);
Real TETimeStep(MeshBlock *pmb);
Real TemperatureL1(MeshBlock *pmb, int iout);
Real TemperatureL2(MeshBlock *pmb, int iout);
Real TemperatureLInf(MeshBlock *pmb, int iout);
Real VelocityL1(MeshBlock *pmb, int iout);
Real VelocityL2(MeshBlock *pmb, int iout);
Real VelocityLInf(MeshBlock *pmb, int iout);
Real FieldL1(MeshBlock *pmb, int iout);
Real FieldL2(MeshBlock *pmb, int iout);
Real FieldLInf(MeshBlock *pmb, int iout);
Real DampedAmp(Real g, Real Om, Real t);
void DampedRotation(Real g, Real q, Real t, Real &C, Real &S);
void TemperatureErrors(MeshBlock *pmb, Real &l1, Real &l2sq, Real &linf);
void VelocityErrors(MeshBlock *pmb, Real &l1, Real &l2sq, Real &linf);
void FieldErrors(MeshBlock *pmb, Real &l1, Real &l2sq, Real &linf);
void PSModeAmplitudes(Real t, Real &AT, Real &ABy, Real &ABz);

void Mesh::InitUserMeshData(ParameterInput *pin) {
  if(!MAGNETIC_FIELDS_ENABLED) {
    std::stringstream msg;
    msg << "### FATAL ERROR in Problem Generator" << std::endl
      << "Magnetic Fields required for aniso tests" << std::endl;
    ATHENA_ERROR(msg);
  }
  // Get floor variables
  P_floor = pin -> GetReal("hydro","pfloor");
  rho_floor = pin -> GetReal("hydro","dfloor");

  // Get number of cells
  nx1 = pin -> GetReal("mesh","nx1"); // x
  nx2 = pin -> GetReal("mesh","nx2"); // y
  nx3 = pin -> GetReal("mesh","nx3"); // z

  // Get dimensions of grid
  xmin = pin -> GetReal("mesh","x1min"); // x min
  xmax = pin -> GetReal("mesh","x1max"); // x min
  ymin = pin -> GetReal("mesh","x2min"); // y min
  ymax = pin -> GetReal("mesh","x2max"); // y max
  zmin = pin -> GetReal("mesh","x3min"); // z min
  zmax = pin -> GetReal("mesh","x3max"); // z max

  // CFL number
  CFL = pin -> GetReal("time","cfl_number");

  // Adiabatic index
  Gamma = pin -> GetReal("hydro","gamma");
  gm1 = Gamma - 1.0;

  // Get initial density and pressure
  rho_0 = pin -> GetReal("problem","rho_0");
  P_0 = pin -> GetReal("problem","P_0");

  // Magnetic fields
  Bx_0 = pin -> GetReal("problem","Bx_0");
  By_0 = pin -> GetReal("problem","By_0");
  Bz_0 = pin -> GetReal("problem","Bz_0");

  // Non-ideal coeffs
  kappa_neu = pin -> GetReal("problem","kappa_aniso_neutral");
  kappa_para = pin -> GetReal("problem","kappa_aniso_para");
  kappa_perp = pin -> GetReal("problem","kappa_aniso_perp");
  kappa_wedge = pin -> GetReal("problem","kappa_aniso_wedge");
  nu_neu = pin -> GetReal("problem","nu_aniso_neutral");
  nu_para = pin -> GetReal("problem","nu_aniso_para");
  nu_perp1 = pin -> GetReal("problem","nu_aniso_perp1");
  nu_perp2 = pin -> GetReal("problem","nu_aniso_perp2");
  nu_wedge1 = pin -> GetReal("problem","nu_aniso_wedge1");
  nu_wedge2 = pin -> GetReal("problem","nu_aniso_wedge2");
  zeta_bulk = pin -> GetReal("problem","zeta_bulk");
  eta_o = pin -> GetReal("problem","eta_ohm");
  eta_ad = pin -> GetReal("problem","eta_ad");
  eta_h = pin -> GetReal("problem","eta_hall");
  kappa_TE_para = pin -> GetReal("problem","kappa_aniso_TE_para");
  kappa_TE_perp = pin -> GetReal("problem","kappa_aniso_TE_perp");
  kappa_TE_wedge = pin -> GetReal("problem","kappa_aniso_TE_wedge");
  ion_frac = pin -> GetReal("problem","ion_frac_const");
  debug_therm_conduct_only = pin -> GetBoolean("problem","debug_therm_conduct_only");
  debug_therm_force_only = pin -> GetBoolean("problem","debug_therm_force_only");

  // Get problem ID
  problem_ID = pin->GetInteger("problem", "problem_ID");

  //EnrollConductionCoefficient(CalculateThermalKappa);
  //EnrollViscosityCoefficient(CalculateViscousNu);
  //EnrollFieldDiffusivity(CalculateMagDiff);
  //EnrollUserTimeStepFunction(TETimeStep);

  // L2 columns hold sum(e^2 dV): take the sqrt in post-processing, since a per-block sqrt
  // does not survive the sum over MeshBlocks/ranks. L_inf columns are reduced with max.
  AllocateUserHistoryOutput(9);
  EnrollUserHistoryOutput(0, TemperatureL1,   "Temp_L1");
  EnrollUserHistoryOutput(1, TemperatureL2,   "Temp_L2sq");
  EnrollUserHistoryOutput(2, TemperatureLInf, "Temp_LInf", UserHistoryOperation::max);
  EnrollUserHistoryOutput(3, VelocityL1,    "Vel_L1");
  EnrollUserHistoryOutput(4, VelocityL2,    "Vel_L2sq");
  EnrollUserHistoryOutput(5, VelocityLInf,  "Vel_LInf", UserHistoryOperation::max);
  EnrollUserHistoryOutput(6, FieldL1,     "Field_L1");
  EnrollUserHistoryOutput(7, FieldL2,     "Field_L2sq");
  EnrollUserHistoryOutput(8, FieldLInf,     "Field_LInf", UserHistoryOperation::max);

  return;
}

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  AllocateUserOutputVariables(13);
  SetUserOutputVariableName(out_T,"Temperature");
  SetUserOutputVariableName(out_sound,"Sound speed");
  SetUserOutputVariableName(out_alfven,"Alfven speed");
  SetUserOutputVariableName(out_normAlfven,"Normal Alfven speed speed");
  SetUserOutputVariableName(out_slowMag,"Slow magnetosonic speed");
  SetUserOutputVariableName(out_fastMag,"Fast magnetosonic speed");
  SetUserOutputVariableName(out_whist,"Whistler speed");
  SetUserOutputVariableName(out_ioncyc,"Ion cyclotron speed");
  SetUserOutputVariableName(out_TEwave,"Peltier+Seebeck speed");
  SetUserOutputVariableName(out_TEWwave,"Peltier+Seebeck+Hall speed");
  SetUserOutputVariableName(out_GVwave,"Gyro-Viscous speed");
  SetUserOutputVariableName(out_RLspeed,"Righi-Leduc speed");
  SetUserOutputVariableName(out_WP, "Well-posed Criterion");

  /*
  SetUserOutputVariableName(out_dt_ideal,"Ideal dt");
  SetUserOutputVariableName(out_dt_vis_para ,"Viscoisty dt");
  SetUserOutputVariableName(out_dt_cnd_para,"Thermal dt");
  SetUserOutputVariableName(out_dt_vis_hyp,"Gyro-Viscous dt");
  SetUserOutputVariableName(out_dt_cnd_hyp,"Righi-Leduc dt");
  SetUserOutputVariableName(out_dt_cnd_tot,"Thermal+RL dt");
  SetUserOutputVariableName(out_dt_vis_tot,"visc+GV dt");
  SetUserOutputVariableName(out_dt_oah,"Magnetic dt ");
  SetUserOutputVariableName(out_dt_TE ,"Isolated TE dt");
  SetUserOutputVariableName(out_dt_TE_hyp,"Hall+Peltier+Seebeck dt");
  SetUserOutputVariableName(out_dt_total ,"Actual dt");
  */

  return;
}

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  if(gid == 0) {std::cout << "MeshBlock::ProblemGenerator called \n";}

  // used for T0 array in case 3
  AllocateRealUserMeshBlockDataField(1);
  ruser_meshblock_data[0].NewAthenaArray(ncells3, ncells2, ncells1);

  // Initial conditions
  for (int k=ks; k <= ke; k++ ) {
    for (int j=js; j <= je; j++ ) {
      for (int i=is; i <= ie; i++ ) {
        phydro->u(IDN,k,j,i) = rho_0;
        phydro->u(IM1,k,j,i) = 0;
        phydro->u(IM2,k,j,i) = 0;
        phydro->u(IM3,k,j,i) = 0;
        phydro->u(IEN,k,j,i) = P_0/(gm1);
      }
    }
  }

  // Set B fields
  Real B_energy = 0.5*(Bx_0*Bx_0 + By_0*By_0 + Bz_0*Bz_0);

  for (int k=ks; k <= ke+1; k++ ) {
    for (int j=js; j <= je+1; j++ ) {
      for (int i=is; i <= ie+1; i++ ) {
          pfield->b.x1f(k,j,i) = Bx_0;
          pfield->b.x2f(k,j,i) = By_0;
          pfield->b.x3f(k,j,i) = Bz_0;
          if(k <=ke && j<=je && i <= ie) {
            phydro->u(IEN,k,j,i) += B_energy;
          }
        }
      }
    }

  // integer to decide what test to run.
  // Make sure to comment/zero out unwanted diffusions in the input file
  //   case 0 Thermal Conduction: single Fourier mode in T at uniform P
  //   case 1 Viscosity: single velocity mode,
  //       amplitude visc_amp; k in x-y plane via mode_nx/ny;
  //       b tilted b_tilt degrees out of the plane from z-hat x k-hat (k perp b);
  //       velocity = visc_a1*k_hat + visc_a2*b_hat + visc_a3*(k_hat x b_hat)
  //   case 2: frozen-flow field mode: Ohmic + ambipolar + Hall
  //   case 3: frozen-flow TE mode: k perpendicular to b (Nernst-Ettingshausen)
  //   case 4: frozen-flow Righi-Leduc on a Beltrami field
  //   case 5: frozen-flow gyroviscosity: Pi3 across b, Pi4 along b
  //   case 6: frozen-flow Peltier-Seebeck (+ Hall)
  //   case 7: HLLE-mod tests
  //   case 8: timestep sweep

  switch (problem_ID) {
    case 0: { // ---- single thermal mode: conduction ----
      T0 = P_0/rho_0;
      dT_amp = pin->GetOrAddReal("problem","dT_amp", 1.0e-6);
      nx  = pin->GetOrAddInteger("problem","mode_nx", 1);
      ny  = pin->GetOrAddInteger("problem","mode_ny", 0);
      nz  = pin->GetOrAddInteger("problem","mode_nz", 0);
      kx  = 2.0*PI*static_cast<Real>(nx)/(xmax - xmin);
      ky  = 2.0*PI*static_cast<Real>(ny)/(ymax - ymin);
      kz  = 2.0*PI*static_cast<Real>(nz)/(zmax - zmin);
      Real B_energy = 0.5*(SQR(Bx_0) + SQR(By_0) + SQR(Bz_0));
      Real x = 0; Real y = 0; Real z = 0; Real T_local = 0;

            for (int k = ks; k <= ke; k++ ) {
                z = this->pcoord->x3v(k);
                for (int j = js; j <= je; j++ ) {
                    y = this->pcoord->x2v(j);
                    for (int i = is; i <= ie; i++ ) {
                        x = this->pcoord->x1v(i);
                        // single Fourier mode in T at uniform pressure:
                        T_local = T0*(1.0 + dT_amp*std::cos(kx*x + ky*y + kz*z));
                        phydro->u(IDN,k,j,i) = P_0/T_local;        // pressure equilibrium
                        phydro->u(IEN,k,j,i) = P_0/gm1 + B_energy; // uniform
                    } // i loop
                } // j loop
            } // k loop
            break;}
    case 1: { // ---- single velocity mode: viscosity ----
      v0   = pin->GetOrAddReal("problem","dv_amp", 1.0e-6);
      a1 = pin->GetOrAddReal("problem","visc_a1", 1.0);   // k_hat amp
      a2 = pin->GetOrAddReal("problem","visc_a2", 1.0);   // b_hat amp
      a3 = pin->GetOrAddReal("problem","visc_a3", 1.0);   // k_hat x b_hat amp
      Real b_tilt = PI/3.0;
      kx   = 2.0*PI/(xmax - xmin);
      Real x  = 0; Real y = 0; Real z = 0; Real vel = 0;

      khat_x = 1.0; khat_y = 0.0; khat_z = 0.0; // k_hat along x direction
      bhat_x = 0; bhat_y = std::cos(b_tilt); bhat_z = std::sin(b_tilt); // b_hat in y-z
      ec_x = 0.0; ec_y = -bhat_z; ec_z = bhat_y; // k_hat x b_hat

      // field along b_hat, velocity directions are the viscous eigenvectors
      B0_mag = std::sqrt(SQR(Bx_0) + SQR(By_0) + SQR(Bz_0));
      for (int k = ks; k <= ke+1; k++ ) {
        for (int j = js; j <= je+1; j++ ) {
          for (int i = is; i <= ie+1; i++ ) {
            pfield->b.x1f(k,j,i) = B0_mag*bhat_x;
            pfield->b.x2f(k,j,i) = B0_mag*bhat_y;
            pfield->b.x3f(k,j,i) = B0_mag*bhat_z;
          }
        }
      }

      for (int k = ks; k <= ke; k++ ) {
        for (int j = js; j <= je; j++ ) {
        z = this->pcoord->x3v(k);
          y = this->pcoord->x2v(j);
          for (int i = is; i <= ie; i++ ) {
            x = this->pcoord->x1v(i);
            vel = v0*std::sin(kx*x + ky*y + kz*z);
            phydro->u(IM1,k,j,i) = rho_0*vel*(a1*khat_x + a2*bhat_x + a3*ec_x);
            phydro->u(IM2,k,j,i) = rho_0*vel*(a1*khat_y + a2*bhat_y + a3*ec_y);
            phydro->u(IM3,k,j,i) = rho_0*vel*(a1*khat_z + a2*bhat_z + a3*ec_z);
            phydro->u(IEN,k,j,i) += 0.5*rho_0*vel*vel*(a1*a1 + a2*a2 + a3*a3);
          }
        }
      }
      break;}
    case 2: {
    // ---- frozen-flow field test: Ohmic, ambipolar and Hall, in any combination ----
    // Background B0 = B0_mag (cos theta, sin theta, 0), theta = 30 deg to k = K x-hat
    // perturbation dB cos(Kx) z-hat
    Real theta = PI/6.0;  // angle between B0 and k
    B0_mag = std::sqrt(SQR(Bx_0) + SQR(By_0) + SQR(Bz_0));
    bhat_x = std::cos(theta);  bhat_y = std::sin(theta);  bhat_z = 0.0;
    kx   = 2.0*PI/(xmax - xmin); // one wavelength in x
    dB_amp = pin->GetOrAddReal("problem","dB_amp", 1.0e-6);

    if (B0_mag == 0.0) {
      std::stringstream msg;
      msg << "### FATAL ERROR in ProblemGenerator case 2" << std::endl
        << "needs a nonzero background field (the input sets its magnitude)" << std::endl;
      ATHENA_ERROR(msg);
    }

        for (int k = ks; k <= ke; k++ ) {
            for (int j = js; j <= je; j++ ) {
                for (int i = is; i <= ie; i++ ) {
                    Real b3 = dB_amp*std::cos(kx*this->pcoord->x1v(i));
                    phydro->u(IDN,k,j,i) = rho_0;
                    phydro->u(IM1,k,j,i) = 0.0;
                    phydro->u(IM2,k,j,i) = 0.0;
                    phydro->u(IM3,k,j,i) = 0.0;
                    phydro->u(IEN,k,j,i) = P_0/gm1 + B_energy + 0.5*b3*b3;
                }
            }
        }
        for (int k = ks; k <= ke; k++ )
            for (int j = js; j <= je; j++ )
                for (int i = is; i <= ie+1; i++ )
                    pfield->b.x1f(k,j,i) = B0_mag*bhat_x;
        for (int k = ks; k <= ke; k++ )
            for (int j = js; j <= je+1; j++ )
                for (int i = is; i <= ie; i++ )
                    pfield->b.x2f(k,j,i) = B0_mag*bhat_y;
        for (int k = ks; k <= ke+1; k++ )
            for (int j = js; j <= je; j++ )
                for (int i = is; i <= ie; i++ )
                    pfield->b.x3f(k,j,i) = dB_amp*std::cos(kx*this->pcoord->x1v(i));
        break;}
        case 3: {
          // ---- thermoelectric mode: Nernst-Ettingshausen ----
          //                          +(neutral+perp conduction, Ohmic+ambipolar)
          // Background B0 = B0_mag y-hat, perpendicular to k = K x-hat
          // Perturbations: cos(Kx). dT, dB_y along b
          // kappa_par, Righi-Leduc, Hall and Peltier-Seebeck all drop out at
          // k perpendicular to b. well-posed when D_T D_B > c_TB c_BT.
          T0 = P_0/rho_0;
          B0_mag = std::sqrt(SQR(Bx_0) + SQR(By_0) + SQR(Bz_0));
          bhat_x = 0.0;  bhat_y = 1.0;  bhat_z = 0.0;
          kx     = 2.0*PI/(xmax - xmin);
          dT_amp = pin->GetOrAddReal("problem","dT_amp", 1.0e-6);
          dB_amp = pin->GetOrAddReal("problem","dB_amp", 1.0e-6);

      if (B0_mag == 0.0) {
        std::stringstream msg;
        msg << "### FATAL ERROR in ProblemGenerator case 3" << std::endl
            << "needs a nonzero background field (the input sets its magnitude)"
            << std::endl;
            ATHENA_ERROR(msg);
      }

          for (int kk = ks; kk <= ke; kk++ ) {
              for (int j = js; j <= je; j++ ) {
                  for (int i = is; i <= ie; i++ ) {
                      Real c  = std::cos(kx*this->pcoord->x1v(i));
                      Real Tc = T0 + dT_amp*c;
                      Real by = B0_mag + dB_amp*c;
                      phydro->u(IDN,kk,j,i) = rho_0; // uniform density
                      phydro->u(IM1,kk,j,i) = 0.0;
                      phydro->u(IM2,kk,j,i) = 0.0;
                      phydro->u(IM3,kk,j,i) = 0.0;
                      phydro->u(IEN,kk,j,i) = rho_0*Tc/gm1 + 0.5*SQR(by);
                  }
              }
          }
          for (int k = ks; k <= ke; k++ )
              for (int j = js; j <= je; j++ )
                  for (int i = is; i <= ie+1; i++ )
                      pfield->b.x1f(k,j,i) = 0.0;
          for (int k = ks; k <= ke; k++ )
              for (int j = js; j <= je+1; j++ )
                  for (int i = is; i <= ie; i++ )
                      pfield->b.x2f(k,j,i) = B0_mag
                                           + dB_amp*std::cos(kx*this->pcoord->x1v(i));
          for (int k = ks; k <= ke+1; k++ )
              for (int j = js; j <= je; j++ )
                  for (int i = is; i <= ie; i++ )
                      pfield->b.x3f(k,j,i) = 0.0;
          break;}
        case 4: {
          // ---- Righi-Leduc: Beltrami field, temperature mode across it ----
          // B = (cos kB z, sin kB z, 0) with |B| = B0_mag: curl B = -kB b,
          //T = T0 + dT cos(kx x).
          T0 = P_0/rho_0;
          B0_mag = std::sqrt(SQR(Bx_0) + SQR(By_0) + SQR(Bz_0));
          kx     = 2.0*PI/(xmax - xmin);
          kB     = 2.0*PI/(zmax - zmin);
          dT_amp = pin->GetOrAddReal("problem","dT_amp", 1.0e-6);

          if (B0_mag == 0.0) {
            std::stringstream msg;
            msg << "### FATAL ERROR in ProblemGenerator case 4" << std::endl
                << "needs a nonzero background field (the input sets its magnitude)"
                << std::endl;
            ATHENA_ERROR(msg);
          }

          for (int kk = ks; kk <= ke; kk++ ) {
              for (int j = js; j <= je; j++ ) {
                  for (int i = is; i <= ie; i++ ) {
                      Real Tc = T0 + dT_amp*std::cos(kx*this->pcoord->x1v(i));
                      phydro->u(IDN,kk,j,i) = rho_0; // uniform density
                      phydro->u(IM1,kk,j,i) = 0.0;
                      phydro->u(IM2,kk,j,i) = 0.0;
                      phydro->u(IM3,kk,j,i) = 0.0;
                      phydro->u(IEN,kk,j,i) = rho_0*Tc/gm1 + 0.5*SQR(B0_mag);
                  }
              }
          }
          for (int k = ks; k <= ke; k++ )
              for (int j = js; j <= je; j++ )
                  for (int i = is; i <= ie+1; i++ )
                      pfield->b.x1f(k,j,i) = B0_mag*std::cos(kB*this->pcoord->x3v(k));
          for (int k = ks; k <= ke; k++ )
              for (int j = js; j <= je+1; j++ )
                  for (int i = is; i <= ie; i++ )
                      pfield->b.x2f(k,j,i) = B0_mag*std::sin(kB*this->pcoord->x3v(k));
          for (int k = ks; k <= ke+1; k++ )
              for (int j = js; j <= je; j++ )
                  for (int i = is; i <= ie; i++ )
                      pfield->b.x3f(k,j,i) = 0.0;
          break;}
        case 5: {
          // ---- Gyro-viscosity: Pi3 across the field, Pi4 along it ----
          // B0 = B0_mag z-hat, rho = rho_0;
          // Two velocity modes, both perpendicular to b:
          //   across b (k = kx x-hat): v = (a_x, a_y, 0) sin(kx x)
          //   along b  (k = kz z-hat): v = (c_x, c_y, 0) sin(kz z)
          // Pi4 does nothing to the first mode and Pi3 nothing to the second.
          B0_mag = std::sqrt(SQR(Bx_0) + SQR(By_0) + SQR(Bz_0));
          kx = 2.0*PI/(xmax - xmin);                              // across-b wavenumber
          kz = 2.0*PI/(zmax - zmin);                              // along-b wavenumber
          v0 = pin->GetOrAddReal("problem","visc_amp", 1.0e-6);

          if (B0_mag == 0.0) {
            std::stringstream msg;
            msg << "### FATAL ERROR in ProblemGenerator case 5" << std::endl
                << "needs a nonzero background field (the input sets its magnitude)"
                << std::endl;
                ATHENA_ERROR(msg);
          }

          for (int kk = ks; kk <= ke; kk++ ) {
              for (int j = js; j <= je; j++ ) {
                  for (int i = is; i <= ie; i++ ) {
                      Real vx = v0*std::sin(kz*this->pcoord->x3v(kk)); // along-b: x
                      Real vy = v0*std::sin(kx*this->pcoord->x1v(i));  // across-b: y
                      phydro->u(IDN,kk,j,i) = rho_0;
                      phydro->u(IM1,kk,j,i) = rho_0*vx;
                      phydro->u(IM2,kk,j,i) = rho_0*vy;
                      phydro->u(IM3,kk,j,i) = 0.0;
                      phydro->u(IEN,kk,j,i) = P_0/gm1 + 0.5*SQR(B0_mag)
                                                      + 0.5*rho_0*(vx*vx + vy*vy);
                  }
              }
          }
          // uniform field along z: the discrete divergence is exactly zero
          for (int k = ks; k <= ke; k++ )
              for (int j = js; j <= je; j++ )
                  for (int i = is; i <= ie+1; i++ )
                      pfield->b.x1f(k,j,i) = 0.0;
          for (int k = ks; k <= ke; k++ )
              for (int j = js; j <= je+1; j++ )
                  for (int i = is; i <= ie; i++ )
                      pfield->b.x2f(k,j,i) = 0.0;
          for (int k = ks; k <= ke+1; k++ )
              for (int j = js; j <= je; j++ )
                  for (int i = is; i <= ie; i++ )
                      pfield->b.x3f(k,j,i) = B0_mag;
          break;}
        case 6: {
          // ---- Thermoelectric Mode: Peltier-Seebeck, (+ Hall) ----
          // B0 = B0_mag (cos theta, sin theta, 0), theta = 45 deg to k = kx x-hat;
          // T = T0 + dT cos(kx x)
          Real theta = PI/4.0; // sin cos is largest here
          T0 = P_0/rho_0;
          B0_mag = std::sqrt(SQR(Bx_0) + SQR(By_0) + SQR(Bz_0));
          bhat_x = std::cos(theta);  bhat_y = std::sin(theta);  bhat_z = 0.0;
          kx     = 2.0*PI/(xmax - xmin);
          dT_amp = pin->GetOrAddReal("problem","dT_amp", 1.0e-6);

          if (B0_mag == 0.0) {
            std::stringstream msg;
            msg << "### FATAL ERROR in ProblemGenerator case 6" << std::endl
                << "needs a nonzero background field (the input sets its magnitude)"
                << std::endl;
                ATHENA_ERROR(msg);
          }

          for (int kk = ks; kk <= ke; kk++ ) {
              for (int j = js; j <= je; j++ ) {
                  for (int i = is; i <= ie; i++ ) {
                      Real Tc = T0 + dT_amp*std::cos(kx*this->pcoord->x1v(i));
                      phydro->u(IDN,kk,j,i) = rho_0; // uniform density
                      phydro->u(IM1,kk,j,i) = 0.0;
                      phydro->u(IM2,kk,j,i) = 0.0;
                      phydro->u(IM3,kk,j,i) = 0.0;
                      phydro->u(IEN,kk,j,i) = rho_0*Tc/gm1 + 0.5*SQR(B0_mag);
                  }
              }
          }
          for (int k = ks; k <= ke; k++ )
              for (int j = js; j <= je; j++ )
                  for (int i = is; i <= ie+1; i++ )
                      pfield->b.x1f(k,j,i) = B0_mag*bhat_x;
          for (int k = ks; k <= ke; k++ )
              for (int j = js; j <= je+1; j++ )
                  for (int i = is; i <= ie; i++ )
                      pfield->b.x2f(k,j,i) = B0_mag*bhat_y;
          for (int k = ks; k <= ke+1; k++ )
              for (int j = js; j <= je; j++ )
                  for (int i = is; i <= ie; i++ )
                      pfield->b.x3f(k,j,i) = 0.0;
          break;}
        // ---- Temp / Vel / Field white noise (?) for HLLE-mod ----
        case 7: {
          // WIP
          break;}
        // ---- (T, rho, |B|) sweep for the timestep check ----
        case 8: {
          // x -> T,  y -> rho,  z -> |B|.  Log-spaced about the base state.
          // Run single-block with time/nlim=0.
          // Run with uniform cartesian grid
          Real dx = pcoord->dx1f(1);
          T0 = P_0/rho_0;
          Real B_0 = std::abs(Bx_0);

          int nT = ie - is + 1, nR = je - js + 1, nB = ke - ks + 1;
          AthenaArray<Real> Tg, Rg, Bg;
          Tg.NewAthenaArray(nT);  Rg.NewAthenaArray(nR);  Bg.NewAthenaArray(nB);
          for (int n = 0; n < nT; n++) Tg(n)   = T0*std::pow(10.0, dx*(n-0.5*(nT-1)));
          for (int n = 0; n < nR; n++) Rg(n)   = rho_0*std::pow(10.0, dx*(n-0.5*(nR-1)));
          for (int n = 0; n < nB; n++) Bg(n)   = B_0*std::pow(10.0, dx*(n-0.5*(nB-1)));

          for (int k = ks; k <= ke+1; k++) {
            Real Bl = Bg(std::min(k - ks, nB-1));
            for (int j = js; j <= je+1; j++) {
              for (int i = is; i <= ie+1; i++) {
                pfield->b.x1f(k,j,i) = Bl*Bx_0/B_0;
                pfield->b.x2f(k,j,i) = 0.0;
                pfield->b.x3f(k,j,i) = 0.0;
              }
            }
          }
          for (int k = ks; k <= ke; k++) {
            for (int j = js; j <= je; j++) {
              for (int i = is; i <= ie; i++) {
                phydro->u(IDN,k,j,i) = Rg(j-js);
                phydro->u(IM1,k,j,i) = 0.0;
                phydro->u(IM2,k,j,i) = 0.0;
                phydro->u(IM3,k,j,i) = 0.0;
                phydro->u(IEN,k,j,i) = Rg(j-js)*Tg(i-is)/gm1 + 0.5*SQR(Bg(k-ks));
              }
            }
          }
        break;}
  default:
    std::cout << "ERROR. Problem_ID can be: \n"
    << "0 - temperature mode (thermal conduction), \n "
    << "1 - velocity Fourier mode (viscoisty), \n"
    << "2 - MHD wave (Ohmic + ambi, Hall effect), \n"
    << "3 - TE eigenmode (Nernst-Ettingshausen) \n"
    << "4 - Righi-Leduc: Beltrami field + transverse T mode \n"
    << "5 - Gyroviscosity: Pi3 across b, Pi4 along b \n"
    << "6 - Peltier-Seebeck (+ Hall) \n"
    << "7 - Timestep checks \n";
    exit(-1);
  }

  // Output initial conditions
  if(Globals::my_rank ==0 && gid == 0) {
  std::cout << "Simulation Set Up: \n";
  std::cout << "\t Spatial Reconstruction: " << pin->GetString("time","xorder") << "\n";
  std::cout << "\t Riemann Solver: " << RIEMANN_SOLVER << "\n";
  std::cout << "\t Time integrator: " << pin->GetString("time","integrator") << "\n";
  std::cout << "\t Boundary -x: " << pin->GetString("mesh","ix1_bc") << "\n";
  std::cout << "\t Boundary x: " << pin->GetString("mesh","ox1_bc") << "\n";
  std::cout << "\t Boundary -y: " << pin->GetString("mesh","ix2_bc") << "\n";
  std::cout << "\t Boundary x: " << pin->GetString("mesh","ox2_bc") << "\n";
  std::cout << "\t Boundary -z: " << pin->GetString("mesh","ix3_bc") << "\n";
  std::cout << "\t Boundary z: " << pin->GetString("mesh","ox3_bc") << "\n";
  std::cout << "Parameter inputs have been read: \n";
  // Hydro, Thermo & magnetic Variables
  std::cout << "\t" << "Hydro and MHD Variables: \n";
  std::cout << "\t\t" << "Ambient Density = " << rho_0  << " M L^-3 \n";
  std::cout << "\t\t" << "Ambient Pressure = " << P_0  << " M L^-1 T^-2 \n";
  std::cout << "\t\t" << "x-Axis B field = "<< Bx_0 << " M^1/2 L^-1/2 T^-1 \n";
  std::cout << "\t\t" << "y-Axis B field = "<< By_0 << " M^1/2 L^-1/2 T^-1 \n";
  std::cout << "\t\t" << "z-Axis B field = "<< Bz_0 << " M^1/2 L^-1/2 T^-1 \n";
  // Eos Variables
  std::cout << "\t" << "Floor Variables: \n";
  std::cout << "\t\t" << "Pressure Floor = " << P_floor << " M L^-1 T^-2 \n";
  std::cout << "\t\t" << "Desnity Floor = " << rho_floor << " M L^-3 \n";
  // Time Variables
  std::cout << "\t" << "Time Variables: \n";
  std::cout << "\t\t" << "CLF number = " << CFL << " arb. u. \n";
  // Diffusion Variables
  std::cout << "\t" << "Diffusion on/off: \n";
  std::cout << "\t\t" << "Kappa Neutral: " << (kappa_neu > 0.0) << "\n";
  std::cout << "\t\t" << "Kappa Parallel: " << (kappa_para > 0.0) << "\n";
  std::cout << "\t\t" << "Kappa Perpendicular: " << (kappa_perp > 0.0) << "\n";
  std::cout << "\t\t" << "Kappa Wedge: " << (kappa_wedge > 0.0) << "\n";
  std::cout << "\t\t" << "Nu Neutral: " << (nu_neu > 0.0) << "\n";
  std::cout << "\t\t" << "Nu Parallel: " << (nu_para > 0.0) << "\n";
  std::cout << "\t\t" << "Nu Perpendicular 1: " << (nu_perp1 > 0.0) << "\n";
  std::cout << "\t\t" << "Nu Perpendicular 2: " << (nu_perp2 > 0.0) << "\n";
  std::cout << "\t\t" << "Nu Wedge 1: " << (nu_wedge1 > 0.0) << "\n";
  std::cout << "\t\t" << "Nu Wedge 2: " << (nu_wedge2 > 0.0) << "\n";
  std::cout << "\t\t" << "Zeta Bulk: " << (zeta_bulk > 0.0) << "\n";
  std::cout << "\t\t" << "Kappa J Parallel: " << (kappa_TE_para > 0.0) << "\n";
  std::cout << "\t\t" << "Kappa J Perpendicular: " << (kappa_TE_perp > 0.0) << "\n";
  std::cout << "\t\t" << "Kappa J Wedge: " << (kappa_TE_wedge > 0.0) << "\n";
  }
  return;
}

void MeshBlock::UserWorkInLoop() {
  if(gid == 0 && pmy_mesh->time == 0) {
    std::cout << "MeshBlock::UserWorkInLoop called \n";
  }

  GetTemperature(this);

    // flag a non-finite value and exit (dt=0 will run forever)
    const Real T0bg = P_0/rho_0;
    for (int k = ks; k <= ke; k++) {
      for (int j = js; j <= je; j++) {
        for (int i = is; i <= ie; i++) {
          for (int n = 0; n < NHYDRO; n++) {
            if (!std::isfinite(phydro->u(n,k,j,i))) nonfinite_found = true;
          }
          for (int n = IB1; n <= IB3; n++) {
            if (!std::isfinite(pfield->bcc(n,k,j,i))) nonfinite_found = true;
          }
          if(pmy_mesh->dt == 0.0 || pmy_mesh->dt_hyperbolic == 0.0
                                 || pmy_mesh->dt_parabolic == 0.0)
            nonfinite_found = true;
        }
      }
    }

  return;
}

void Mesh::UserWorkInLoop() {
  if(time == 0 && Globals::my_rank == 0) {
    std::cout << "Mesh::UserWorkInLoop called \n";
  }


  // in Mesh::UserWorkInLoop()
  // A step collapsed to a tiny fraction of the largest so far means a runaway (wave
  // speeds blowing up, or dt -> 0): the run would never reach tlim.
  dt_largest = std::max(dt_largest, dt);
  const bool last_step = (time + dt >= tlim);
  if (!last_step && !(dt > 1.0e-4*dt_largest)) {
    std::cout << "### stopping at cycle " << ncycle << ", t = " << time + dt
              << ": time step collapsed (dt = " << dt << ", largest " << dt_largest
              << ")" << std::endl;
    std::raise(SIGTERM);
  }

  if (nonfinite_found) {
    std::cout << "### stopping at cycle " << ncycle << ", t = " << time + dt
              << std::endl;
    std::raise(SIGTERM);
  }

  return;
}

void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin) {
  if(gid == 0 && pmy_mesh->time == 0) {
    std::cout << "MeshBlock::UserWorkBeforeOutput called \n";
  }

  GetTemperature(this);
  GetWaveSpeeds(this);
  if(problem_ID==3)CalculateWellPosedness(this);
  //if(problem_ID==7) GetTimeSteps(this);

  return;
}

void Mesh::UserWorkAfterLoop(ParameterInput *pin) {
  if(time == 0 && Globals::my_rank == 0) {
    std::cout << "Mesh::UserWorkAfterLoop called \n";
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void GetTemperature
//! \brief Fills MeshBlock::user_out_var[out_temperature]
void GetTemperature(MeshBlock *pmb) {
  // Get position indicies
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  // Get ghost cell indicies
  int il = is - NGHOST; int jl = js - NGHOST; int kl = ks - NGHOST;
  int iu = ie + NGHOST; int ju = je + NGHOST; int ku = ke + NGHOST;

  for (int k = kl; k<= ku; k++) {
    for (int j = jl; j<= ju; j++) {
      for (int i = il; i<= iu; i++) {
        pmb->user_out_var(out_T,k,j,i) = pmb->phydro->
                         w(IPR,k,j,i)/pmb->phydro->w(IDN,k,j,i);
        } // i loop
      } // j loop
    } // k loop
  return;
}

//----------------------------------------------------------------------------------------
//! \fn GetWaveSpeeds
//! \brief Calcualtes wavespeeds (sound, magnetic, whistler, ion-cyclotron, TE, GV)
void GetWaveSpeeds(MeshBlock *pmb) {
  // Get position indicies
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  Real P = 0.0; Real rho = 0.0;
  Real vx = 0.0; Real vy= 0.0; Real vz = 0.0;
  Real v_mag = 0.0; Real v_dot_B = 0.0;
  Real Bx = 0.0; Real By = 0.0; Real Bz = 0.0; Real B = 0.0; Real Bn = 0.0;
  Real bx = 0; Real by = 0; Real bz = 0;
  Real c = 0.0; Real ca= 0.0; Real can = 0.0; Real cs = 0.0; Real cf= 0.0;
  Real cw = 0.0; Real ci= 0.0; Real cTE = 0.0; Real cTEw = 0.0; Real cGV = 0.0;
  Real cRL = 0.0; Real eta_h = 0.0; Real kJpara = 0.0; Real kJperp = 0.0;
  Real ion_frac = 0.0; Real nuw1 = 0.0; Real nuw2 = 0.0;
  Real nyquest_wavenumber = 0.0;

  Real inv_root2 = 1.0/std::sqrt(2);
  Coordinates *pcoord_ = pmb->pcoord;
  HydroDiffusion &hdif_ = pmb->phydro->hdif;
  FieldDiffusion &fdif_ = pmb->pfield->fdif;

  for (int k = ks; k<= ke; k++) {
    for (int j = js; j<= je; j++) {
      for (int i = is; i<= ie; i++) {
        P = pmb->phydro->w(IPR,k,j,i);
        rho = pmb->phydro->w(IDN,k,j,i);

        vx = pmb->phydro->w(IVX,k,j,i);
        vy = pmb->phydro->w(IVY,k,j,i);
        vz = pmb->phydro->w(IVZ,k,j,i);

        Bx = pmb->pfield->bcc(IB1,k,j,i);
        By = pmb->pfield->bcc(IB2,k,j,i);
        Bz = pmb->pfield->bcc(IB3,k,j,i);
        Bn = Bx;
        B = std::sqrt(Bx*Bx + By*By + Bz*Bz);

        c = std::sqrt(Gamma*P/rho);
        ca = std::sqrt(Bx*Bx + By*By + Bz*Bz)/std::sqrt(rho);

        v_mag = std::sqrt(vx*vx + vy*vy + vz*vz);

        can = std::abs(Bn)/std::sqrt(rho); // Normal Alfvén: B along x

        Real disc = std::sqrt((c*c + ca*ca)*(c*c + ca*ca) - 4.0*c*c*can*can);
        cs = inv_root2 * std::sqrt(c*c + ca*ca - disc);
        cf = inv_root2 * std::sqrt(c*c + ca*ca + disc);

        nyquest_wavenumber = 1.0/std::min({pcoord_->dx1v(i),pcoord_->dx2v(j),
                           pcoord_->dx3v(k)});

        eta_h = (fdif_.eta_hall > 0.0)
            ? fdif_.etaB(FieldDiffusion::DiffProcess::hall, k,j,i)
            : 0.0;
        ion_frac = (hdif_.kappa_aniso_neutral > 0.0
              || hdif_.nu_aniso_neutral > 0.0)
            ?  hdif_.ion_frac(k,j,i)
            : 1.0;
        kJpara = (fdif_.kappa_aniso_TE_para > 0.0)
            ? fdif_.thermKappa(FieldDiffusion::ThermElecProcess::
                      kappa_TE_para,k,j,i)
            : 0.0;
        kJperp = (fdif_.kappa_aniso_TE_perp > 0.0)
              ? fdif_.thermKappa(FieldDiffusion::ThermElecProcess::
                     kappa_TE_perp,k,j,i)
            : 0.0;
        nuw1 = (hdif_.nu_aniso_wedge1 > 0.0)
             ? hdif_.nu(HydroDiffusion::DiffProcess::nu_wedge1,k,j,i)
             : 0.0;
        nuw2 = (hdif_.nu_aniso_wedge2 > 0.0)
            ? hdif_.nu(HydroDiffusion::DiffProcess::nu_wedge2,k,j,i)
            : 0.0;

        if(fdif_.eta_hall > 0.0) {
        cw = 0.5*(eta_h*nyquest_wavenumber)
           + std::sqrt(0.25*eta_h*eta_h*nyquest_wavenumber*nyquest_wavenumber
                 + can*can);
        ci = 0.5*(eta_h*nyquest_wavenumber)
           - std::sqrt(0.25*eta_h*eta_h*nyquest_wavenumber*nyquest_wavenumber
                 + can*can);
        } else {cw = 0; ci = 0;}

        if(fdif_.kappa_aniso_TE_para || fdif_.kappa_aniso_TE_perp >0.0) {
          Real dkJ = kJpara - kJperp;
          Real E_int = P/gm1;
          Real D = dkJ*dkJ/(std::sqrt(E_int)
               *nyquest_wavenumber*nyquest_wavenumber);
          cTE =  std::sqrt(cf*cf + D/4.0);
        } else {cTE = 0.0;}

        if((fdif_.kappa_aniso_TE_para > 0.0  || fdif_.kappa_aniso_TE_perp > 0.0)
                        && fdif_.eta_hall > 0.0) {
          Real dkJ = kJpara - kJperp;
          Real E_int = P/gm1;
          Real D = (dkJ*dkJ/E_int)
               *nyquest_wavenumber*nyquest_wavenumber;
          Real E = eta_h*eta_h*nyquest_wavenumber*nyquest_wavenumber;
          cTEw =  std::sqrt(std::max(cf,cw)*std::max(cf,cw)
                    + D/4.0 + std::sqrt(D*E)/2.0);
        } else {cTEw = 0.0;}

        if(hdif_.nu_aniso_wedge1 > 0.0 || hdif_.nu_aniso_wedge2 > 0.0) {
          Real gv1 = nuw1*ion_frac*nyquest_wavenumber;
          Real gv2 = 4.0*nuw2*ion_frac*nyquest_wavenumber;
          Real gv3 = 0.0;
          Real u_star = 0.0;
          Real denom = ion_frac*ion_frac*nuw1*(nuw1-4.0*nuw2);
          if(std::abs(denom) > TINY_NUMBER) {
            u_star = ion_frac*ion_frac*
                 (nuw1*nuw1 - 2.0*nuw1*nuw2 - 2.0*nuw2*nuw2)/denom;
          }
          if(u_star >= 0.0 && u_star <= 1.0) {
            gv3 = ion_frac*std::sqrt((4.0*nuw2*nuw2*nuw2*(-2.0*nuw1-nuw2))
                          /denom)*nyquest_wavenumber;
          }
          cGV = cf + std::max({gv1, gv2, gv3});
        }


        if (hdif_.kappa_aniso_wedge > 0.0) {
          Real kw = hdif_.kappa(HydroDiffusion::DiffProcess::kappa_wedge,k,j,i);

          Real bxip,byip,bzip, bxim,byim,bzim;
          Real bxjp,byjp,bzjp, bxjm,byjm,bzjm;
          Real bxkp,bykp,bzkp, bxkm,bykm,bzkm;
          GetBhat(pmb,k,j,i+1, bxip,byip,bzip);
          GetBhat(pmb,k,j,i-1, bxim,byim,bzim);

          Real inv_dx2 = 1.0/(pcoord_->dx1v(i-1) + pcoord_->dx1v(i));
          Real curlbx = 0.0, curlby = 0.0, curlbz = 0.0;

          if (pmb->pmy_mesh->f2) {
            GetBhat(pmb,k,j+1,i, bxjp,byjp,bzjp);
            GetBhat(pmb,k,j-1,i, bxjm,byjm,bzjm);
            Real inv_dy2 = 1.0/(pcoord_->h2v(i)
                      *(pcoord_->dx2v(j-1) + pcoord_->dx2v(j)));
            curlbx += (bzjp - bzjm)*inv_dy2;
            curlbz -= (bxjp - bxjm)*inv_dy2;
          }
          if (pmb->pmy_mesh->f3) {
            GetBhat(pmb,k+1,j,i, bxkp,bykp,bzkp);
            GetBhat(pmb,k-1,j,i, bxkm,bykm,bzkm);
            Real inv_dz2 = 1.0/((pcoord_->dx3v(k-1) + pcoord_->dx3v(k))
                      *pcoord_->h31v(i)*pcoord_->h32v(j));
            curlbx -= (bykp - bykm)*inv_dz2;
            curlby += (bxkp - bxkm)*inv_dz2;
          }
          curlby -= (bzip - bzim)*inv_dx2;
          curlbz += (byip - byim)*inv_dx2;

          cRL = gm1*ion_frac*kw*std::max({std::abs(curlbx),
                          std::abs(curlby),
                          std::abs(curlbz)});
        }

        pmb->user_out_var(out_sound,k,j,i) = c;
        pmb->user_out_var(out_alfven,k,j,i) = ca;
        pmb->user_out_var(out_normAlfven,k,j,i) = can;
        pmb->user_out_var(out_slowMag,k,j,i) = cs;
        pmb->user_out_var(out_fastMag,k,j,i) = cf;
        pmb->user_out_var(out_whist,k,j,i) = cw;
        pmb->user_out_var(out_ioncyc,k,j,i) = ci;
        pmb->user_out_var(out_TEwave,k,j,i) = cTE;
        pmb->user_out_var(out_TEWwave,k,j,i) = cTEw;
        pmb->user_out_var(out_GVwave,k,j,i) = cGV;
        pmb->user_out_var(out_RLspeed,k,j,i) = cRL;
      }
    }
  }
  return;
}

/*
//----------------------------------------------------------------------------------------
//! \fn void GetEntropyProduction
//! \brief Calculates the electron entropy production per unit volume due to TE.
//  SB1 eqn (4.51)
//  SB1: Braginskii, S., “Transport Processes in a Plasma”, Reviews of Plasma Physics,
//  Jan. 1965, vol. 1, pp. 205, URL:
//  https://static.ias.edu/pitp/2016/sites/pitp/files/braginskii_1965-1.pdf.
void GetEntropyProduction(MeshBlock *pmb) {

  // This needs a way to get TE fluxes and fields at runtime before it can be used.

  Coordinates *pco = pmb->pcoord;

   // Get position indicies
   int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
   int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

   // Get ghost cell indicies
   int il = is - NGHOST; int jl = js - NGHOST; int kl = ks - NGHOST;
   int iu = ie + NGHOST; int ju = je + NGHOST; int ku = ke + NGHOST;

   AthenaArray<Real> T;
   T.NewAthenaArray(pmb->ncells3, pmb->ncells2, pmb->ncells1);

   AthenaArray<Real> &jcc = pmb->pfield->fdif.jcc;

   // Fill temperature array
   for (int k = kl; k<= ku; k++) {
    for (int j = jl; j<= ju; j++) {
      for (int i = il; i<= iu; i++) {
        T(k,j,i) = pmb->user_out_var(out_T,k,j,i);
        }
      }
    }

  // Rebuild qJ and FT at cell centres
  for (int k = ks; k<= ke; k++) {
    for (int j = js; j<= je; j++) {
      for (int i = is; i<= ie; i++) {
        // grad(1/T) at cell centre
        Real dInvTdx = -(1.0/SQR(T(k,j,i))) * (T(k,j,i+1) - T(k,j,i-1))
        / (pco->dx1v(i-1) + pco->dx1v(i));
        Real dInvTdy = -(1.0/SQR(T(k,j,i))) * (T(k,j+1,i) - T(k,j-1,i))
                / (pco->h2v(i) * (pco->dx2f(j-1) + pco->dx2f(j)));
        Real dInvTdz = -(1.0/SQR(T(k,j,i))) * (T(k+1,j,i) - T(k-1,j,i))
                / ((pco->dx3f(k-1) + pco->dx3f(k)) * pco->h31v(i)
                  * pco->h32f(j));

        Real invT = 1/T(k,j,i);

// Interpolate face-centred qJ to cell centre
// Real qJx_cc = 0.5*(qJx(k,j,i) + qJx(k,j,i+1));
// Real qJy_cc = 0.5*(qJy(k,j,i) + qJy(k,j+1,i));
// Real qJz_cc = 0.5*(qJz(k,j,i) + qJz(k+1,j,i));

// Interpolate edge-centred FT to cell centre
// Real FTx_cc = 0.25*(FTx(k,j,i) + FTx(k,j+1,i)
//        + FTx(k+1,j,i) + FTx(k+1,j+1,i));
// Real FTy_cc = 0.25*(FTy(k,j,i) + FTy(k,j,i+1)
//        + FTy(k+1,j,i) + FTy(k+1,j,i+1));
// Real FTz_cc = 0.25*(FTz(k,j,i) + FTz(k,j+1,i)
//         + FTz(k,j,i+1) + FTz(k,j+1,i+1));

// Entropy production rate
// pmb->user_out_var(out_entropyProd,k,j,i) = (qJx_cc*dInvTdx + qJy_cc*dInvTdy
//                               + qJz_cc*dInvTdz)
//     + invT*(FTx_cc*jcc(IB1,k,j,i) + FTy_cc*jcc(IB2,k,j,i) + FTz_cc*jcc(IB3,k,j,i));
        }
      }
    }

  return;
}
*/
/*
//----------------------------------------------------------------------------------------
//! \fn GetTimesteps
//! \brief outputs timesteps to compare to Sympy/Numpy tests
void GetTimeSteps(MeshBlock *pmb) {
  HydroDiffusion &hdif = pmb->phydro->hdif;
  FieldDiffusion &fdif = pmb->pfield->fdif;

  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;


  for (int k = ks; k <= ke; k++) {
    for (int j = js; j <= je; j++) {
    for (int i = is; i <= ie; i++) {

      pmb->user_out_var(out_dt_vis_hyp,k,j,i) = 1.0/hdif.inv_dt_vis_disp(k,j,i);
      pmb->user_out_var(out_dt_vis_tot,k,j,i) = 1.0/hdif.inv_dt_visc(k,j,i);
      pmb->user_out_var(out_dt_vis_para,k,j,i) = 1.0/(hdif.inv_dt_visc(k,j,i)
                            - hdif.inv_dt_vis_disp(k,j,i));

      pmb->user_out_var(out_dt_cnd_hyp,k,j,i) = 1.0/hdif.inv_dt_cnd_disp(k,j,i);
      pmb->user_out_var(out_dt_cnd_tot,k,j,i) = 1.0/hdif.inv_dt_cnd(k,j,i);
      pmb->user_out_var(out_dt_cnd_para,k,j,i) = 1.0/(hdif.inv_dt_cnd(k,j,i)
                             -hdif.inv_dt_cnd_disp(k,j,i));

      pmb->user_out_var(out_dt_oah,k,j,i) = 1.0/fdif.inv_dt_OAH(k,j,i);
      pmb->user_out_var(out_dt_TE ,k,j,i) = 1.0/fdif.inv_dt_TE(k,j,i);
      pmb->user_out_var(out_dt_TE_hyp,k,j,i) = 1.0/fdif.inv_dt_TE_hyp(k,j,i);

     pmb->user_out_var(out_dt_total,k,j,i) = 1.0/(pmb->phydro->inv_dt_ideal(k,j,i)
                         + std::max<Real>(hdif.inv_dt_visc(k,j,i),
                         hdif.inv_dt_cnd(k,j,i)
                         + fdif.inv_dt_OAH(k,j,i)
                         + fdif.inv_dt_TE(k,j,i)));
    }
    }
  }
  return;
}
*/

//----------------------------------------------------------------------------------------
//! \fn void GetBhat
//! \brief Cell-centred unit magnetic field; zero where |B| is unresolved.
void GetBhat(MeshBlock *pmb, int k, int j, int i, Real &bx, Real &by, Real &bz) {
  Real Bx = pmb->pfield->bcc(IB1,k,j,i);
  Real By = pmb->pfield->bcc(IB2,k,j,i);
  Real Bz = pmb->pfield->bcc(IB3,k,j,i);
  Real B  = std::sqrt(Bx*Bx + By*By + Bz*Bz);
  if (B > TINY_NUMBER) {
    bx = Bx/B; by = By/B; bz = Bz/B;
  } else {
    bx = 0.0; by = 0.0; bz = 0.0;
  }
  return;
}

void CalculateThermalKappa(HydroDiffusion *phdif, MeshBlock *pmb,
                           const AthenaArray<Real> &prim,const AthenaArray<Real> &bcc,
                           int is, int ie, int js, int je,int ks, int ke) {
  Real rho = 0.0; Real P = 0.0; Real T = 0.0;
  Real Bx = 0.0; Real By = 0.0; Real Bz = 0.0;
  Real B = 0.0; Real k_neu = 0.0; Real k_para =0.0;
  Real k_perp = 0.0; Real k_wedge = 0.0; Real HII = 0.0;

  HydroDiffusion &hdif = pmb->phydro->hdif;
  FieldDiffusion &fdif = pmb->pfield->fdif;
  bool have_ion_frac = hdif.hydro_diffusion_defined && hdif.aniso_conduction
  && (hdif.kappa_aniso_neutral > 0.0);

  int il = pmb->is - NGHOST; int iu = pmb->ie + NGHOST;
  int jl = pmb->js - (pmb->pmy_mesh->f2 ? NGHOST : 0);
  int ju = pmb->je + (pmb->pmy_mesh->f2 ? NGHOST : 0);
  int kl = pmb->ks - (pmb->pmy_mesh->f3 ? NGHOST : 0);
  int ku = pmb->ke + (pmb->pmy_mesh->f3 ? NGHOST : 0);

  Real a0 = 1.0; // scale factor
  Real T_ion = 1e3;

  for (int k = kl; k <= ku; k++) {
    for (int j = jl; j <= ju; j++) {
      for (int i = il; i <= iu; i++) {
      rho = pmb->phydro->w(IDN,k,j,i);
      P = pmb->phydro->w(IPR,k,j,i);
      T = P/rho;
      Bx = pmb->pfield->bcc(IB1,k,j,i);
      By = pmb->pfield->bcc(IB2,k,j,i);
      Bz = pmb->pfield->bcc(IB3,k,j,i);
      B = std::sqrt(Bx*Bx + By*By + Bz*Bz);

      HII = (T/T_ion)/(1+(T/T_ion));
      k_neu = a0*std::pow(T,1.0/2.0);
      k_para = a0*std::pow(T,5.0/2.0);
      k_perp = a0*std::pow(T,-1.0/2.0)*std::pow(B,-2.0);
      k_wedge = a0*T/B;

      if(have_ion_frac) hdif.ion_frac(k,j,i) = HII;
      hdif.kappa(hdif.kappa_neutral,k,j,i) = k_neu;
      hdif.kappa(hdif.kappa_para,k,j,i) = k_para;
      hdif.kappa(hdif.kappa_perp,k,j,i) = k_perp;
      hdif.kappa(hdif.kappa_wedge,k,j,i) = k_wedge;
      }
    }
  }

  return;
}

void CalculateViscousNu(HydroDiffusion *phdif, MeshBlock *pmb,
                        const AthenaArray<Real> &prim,const AthenaArray<Real> &bcc,
                        int is, int ie, int js, int je,int ks, int ke) {
  Real rho = 0.0; Real P = 0.0; Real T = 0.0;
  Real Bx = 0.0; Real By = 0.0; Real Bz = 0.0;
  Real B = 0.0; Real n_neu = 0.0; Real n_para =0.0;
  Real n_perp = 0.0; Real n_wedge = 0.0; Real HII = 0.0;

  HydroDiffusion &hdif = pmb->phydro->hdif;
  FieldDiffusion &fdif = pmb->pfield->fdif;
  bool have_ion_frac = hdif.hydro_diffusion_defined && hdif.aniso_conduction
  && (hdif.kappa_aniso_neutral > 0.0);

  int il = pmb->is - NGHOST; int iu = pmb->ie + NGHOST;
  int jl = pmb->js - (pmb->pmy_mesh->f2 ? NGHOST : 0);
  int ju = pmb->je + (pmb->pmy_mesh->f2 ? NGHOST : 0);
  int kl = pmb->ks - (pmb->pmy_mesh->f3 ? NGHOST : 0);
  int ku = pmb->ke + (pmb->pmy_mesh->f3 ? NGHOST : 0);

  Real a1 = 0.3; // scale factor
  Real T_ion = 1e3;

  for (int k = kl; k <= ku; k++) {
    for (int j = jl; j <= ju; j++) {
      for (int i = il; i <= iu; i++) {
      rho = pmb->phydro->w(IDN,k,j,i);
      P = pmb->phydro->w(IPR,k,j,i);
      T = P/rho;
      Bx = pmb->pfield->bcc(IB1,k,j,i);
      By = pmb->pfield->bcc(IB2,k,j,i);
      Bz = pmb->pfield->bcc(IB3,k,j,i);
      B = std::sqrt(Bx*Bx + By*By + Bz*Bz);

      HII = (T/T_ion)/(1+(T/T_ion));
      n_neu = a1*std::pow(T,1.0/2.0);
      n_para = a1*std::pow(T,5.0/2.0);
      n_perp = a1*std::pow(T,-1.0/2.0)*std::pow(B,-2.0);
      n_wedge = a1*T/B;

      hdif.ion_frac(k,j,i) = HII;
      hdif.nu(hdif.nu_neutral,k,j,i) = n_neu;
      hdif.nu(hdif.nu_para,k,j,i) = n_para;
      hdif.nu(hdif.nu_perp1,k,j,i) = n_perp;
      hdif.nu(hdif.nu_perp2,k,j,i) = n_perp;
      hdif.nu(hdif.nu_wedge1,k,j,i) = n_wedge;
      hdif.nu(hdif.nu_wedge2,k,j,i) = n_wedge;
      }
    }
  }
  return;
}

void CalculateMagDiff(FieldDiffusion *pfdif, MeshBlock *pmb,
                      const AthenaArray<double> &w, const AthenaArray<double> &bmag,
                      int is, int ie, int js, int je, int ks, int ke) {
  Real rho = 0.0; Real P = 0.0; Real T = 0.0;
  Real Bx = 0.0; Real By = 0.0; Real Bz = 0.0;
  Real B = 0.0; Real HII = 0.0;
  Real kj_para =0.0; Real kj_perp = 0.0; Real kj_wedge = 0.0;
  Real eta_o = 0.0; Real eta_ad = 0.0; Real eta_h = 0.0;

  HydroDiffusion &hdif = pmb->phydro->hdif;
  FieldDiffusion &fdif = pmb->pfield->fdif;
  bool have_ion_frac = hdif.hydro_diffusion_defined && hdif.aniso_conduction
  && (hdif.kappa_aniso_neutral > 0.0);

  int il = pmb->is - NGHOST; int iu = pmb->ie + NGHOST;
  int jl = pmb->js - (pmb->pmy_mesh->f2 ? NGHOST : 0);
  int ju = pmb->je + (pmb->pmy_mesh->f2 ? NGHOST : 0);
  int kl = pmb->ks - (pmb->pmy_mesh->f3 ? NGHOST : 0);
  int ku = pmb->ke + (pmb->pmy_mesh->f3 ? NGHOST : 0);

  Real a2 = 0.1; // scale factor
  Real a3 = 0.2;
  Real T_ion = 1e3;

  for (int k = kl; k <= ku; k++) {
    for (int j = jl; j <= ju; j++) {
      for (int i = il; i <= iu; i++) {
      rho = pmb->phydro->w(IDN,k,j,i);
      P = pmb->phydro->w(IPR,k,j,i);
      T = P/rho;
      Bx = pmb->pfield->bcc(IB1,k,j,i);
      By = pmb->pfield->bcc(IB2,k,j,i);
      Bz = pmb->pfield->bcc(IB3,k,j,i);
      B = std::sqrt(Bx*Bx + By*By + Bz*Bz);

      eta_o = a2*std::pow(T,-3.0/2.0);
      eta_ad = a2*std::pow(rho,-2.0);
      eta_h = a2*B/rho;

      eta_ad *= B*B; // athena uses eta_ad*B^2 for coeff

      HII = (T/T_ion)/(1+(T/T_ion));
      kj_para = a3*T;
      kj_perp = a3*T*std::pow(B,-2.0);
      kj_wedge = a3*T/B;

      hdif.ion_frac(k,j,i) = HII;
      fdif.etaB(fdif.ohmic,k,j,i) = eta_o;
      fdif.etaB(fdif.ambipolar,k,j,i) = eta_ad;
      fdif.etaB(fdif.hall,k,j,i) = eta_h;
      fdif.thermKappa(fdif.kappa_TE_para,k,j,i) = kj_para;
      fdif.thermKappa(fdif.kappa_TE_perp,k,j,i) = kj_perp;
      fdif.thermKappa(fdif.kappa_TE_wedge,k,j,i) = kj_wedge;
      }
    }
  }
  return;
}

void CalculateWellPosedness(MeshBlock *pmb) {
  // Get position indicies
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  Real gm1 = pmb->peos->GetGamma() - 1.0;

  for (int k = ks; k <= ke; k++) {
    for (int j = js; j <= je; j++) {
      for (int i = is; i <= ie; i++) {
        Real HII = (pmb->phydro->hdif.kappa_aniso_neutral > 0.0)
                 ? pmb->phydro->hdif.ion_frac(k,j,i) : 1.0;
        Real kn = pmb->phydro->hdif.kappa(HydroDiffusion::kappa_neutral,k,j,i);
        Real kp = pmb->phydro->hdif.kappa(HydroDiffusion::kappa_perp,k,j,i);
        Real eta_o = pmb->pfield->fdif.etaB(FieldDiffusion::ohmic,k,j,i);
        Real eta_a = pmb->pfield->fdif.etaB(FieldDiffusion::ambipolar,k,j,i)
                     * (pmb->pfield->bcc(IB1,k,j,i) * pmb->pfield->bcc(IB1,k,j,i)
                      + pmb->pfield->bcc(IB2,k,j,i) * pmb->pfield->bcc(IB2,k,j,i)
                      + pmb->pfield->bcc(IB3,k,j,i) * pmb->pfield->bcc(IB3,k,j,i));
        Real kjw = pmb->pfield->fdif.thermKappa(FieldDiffusion::ThermElecProcess::
                            kappa_TE_wedge,k,j,i);

        Real eps_int = pmb->phydro->w(IPR,k,j,i)/gm1;
        Real cc = SQR(HII*kjw)/eps_int;                     // c_TB c_BT
        // D_T D_B/(c_TB c_BT): well-posed when > 1
        Real wp = (cc > 0.0) ? gm1*((1-HII)*kn + HII*kp)*(eta_o+eta_a)/cc : HUGE_NUMBER;
        pmb->user_out_var(out_WP,k,j,i) = wp;
      }
    }
  }

  return;
}

// show ill-posed dx scaling by holding dt constant until something breaks
Real TETimeStep(MeshBlock *pmb) {
  return 1e-4;
}

//----------------------------------------------------------------------------------------
//! \fn void TemperatureErrors
//! \brief Temperature errors e = |T_sim - T_ana| on this MeshBlock:
//!  l1 = sum e dV, l2sq = sum e^2 dV, linf = max e.
void TemperatureErrors(MeshBlock *pmb, Real &l1, Real &l2sq, Real &linf) {
  l1 = 0.0; l2sq = 0.0; linf = 0.0;

  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;
  Coordinates *pco = pmb->pcoord;
  Real t = pmb->pmy_mesh->time;

  Real amp = 0.0; // case 0: mode amplitude
  switch (problem_ID) {
    case 0: { // ---- single thermal mode: frozen-flow (isochoric) decay ----
      // T(r,t) = T0*(1 + dT*exp(-rate*t)*cos(k.x)), rate = (gamma-1)*kappa*k^2.
      Real k2 = kx*kx + ky*ky + kz*kz;
      Real B0 = std::sqrt(Bx_0*Bx_0 + By_0*By_0 + Bz_0*Bz_0);
      Real K_par = (B0 > 0.0) ? (kx*Bx_0 + ky*By_0 + kz*Bz_0)/B0 : 0.0; // k.b_hat
      Real HII = (kappa_neu > 0.0) ? ion_frac : 1.0;
      Real rate = gm1*((1.0 - HII)*kappa_neu*k2
            + HII*(kappa_para*SQR(K_par) + kappa_perp*(k2 - SQR(K_par))));
      if(pmb->phydro->hdif.kappa_iso > 0.0) { // stock athena
        rate = gm1*pmb->phydro->hdif.kappa_iso*k2;
      }
      amp = dT_amp*std::exp(-rate*t);

      for (int k = ks; k <= ke; k++) {
          for (int j = js; j <= je; j++) {
              for (int i = is; i <= ie; i++) {
                  Real T_sim = pmb->phydro->w(IPR,k,j,i)/pmb->phydro->w(IDN,k,j,i);
                  Real T_ana = T0*(1.0 + amp*std::cos(kx*pco->x1v(i) + ky*pco->x2v(j)
                            + kz*pco->x3v(k)));
                  Real err = std::abs(T_sim - T_ana);
                  Real dV = pco->GetCellVolume(k,j,i);
                  l1 += err*dV;
                  l2sq += err*err*dV;
                  linf = std::max(linf, err);
              }
            }
          }
      break;}
    case 3: {
      T0   = P_0/rho_0;
      Real HII  = (kappa_neu > 0.0) ? ion_frac : 1.0;
      Real DT   = gm1*((1.0 - HII)*kappa_neu + HII*kappa_perp); // kapa_eff
      Real DB   = eta_o + eta_ad*SQR(B0_mag);                   // eta_eff
      Real cTB  = gm1*HII*kappa_TE_wedge/rho_0;                 // Ettingshausen
      Real cBT  = HII*kappa_TE_wedge/T0;                        // Nernst
      Real abar = 0.5*(DT + DB);                                // mean rate
      Real del  = 0.5*(DT - DB);                                // half-difference
      Real D2   = del*del + cTB*cBT;                            // D^2 >= 0
      Real tau  = kx*kx*t;
      Real q    = D2*tau*tau;

      Real EC = 0.0; Real ES = 0.0; // e^{-a tau} cosh(D tau), e^{-a tau} sinh(D tau)/D
      if (q < 1.0e-4) {       // Taylor series in D^2: no 0/0 at D = 0
      Real decay = std::exp(-abar*tau);
      EC = decay*(1.0 + q/2.0 + q*q/24.0);
      ES = decay*tau*(1.0 + q/6.0 + q*q/120.0);
      } else {                   // D real, combine exponents first
      Real D  = std::sqrt(D2);
      Real ep = std::exp(-(abar - D)*tau); // grows when ill-posed (abar < D)
      Real em = std::exp(-(abar + D)*tau);
      EC = 0.5*(ep + em);
      ES = 0.5*(ep - em)/D;
      }
      Real AT  = dT_amp*(EC - del*ES) - dB_amp*cTB*ES;

        for (int k = ks; k <= ke; k++) {
          for (int j = js; j <= je; j++) {
            for (int i = is; i <= ie; i++) {
              Real cx  = std::cos(kx*pco->x1v(i));
              Real Tn  = pmb->phydro->w(IPR,k,j,i)/pmb->phydro->w(IDN,k,j,i);
              Real err = std::abs(Tn - (T0 + AT*cx));
              Real dV  = pco->GetCellVolume(k,j,i);
              l1   += err*dV;
              l2sq += err*err*dV;
              linf  = std::max(linf, err);
            }
          }
        }
      break;}
    case 4: { // ---- frozen-flow Righi-Leduc: each z-layer advects rigidly ----
      // T = T0 + dT0 cos(kx (x - U cos(kB z) t)),  U = (gamma-1) x kappa_wedge kB
      T0     = P_0/rho_0;
      Real HII    = (kappa_neu > 0.0) ? ion_frac : 1.0;
      Real U      = gm1*HII*kappa_wedge*kB; // Righi-Leduc advection speed
      for (int k = ks; k <= ke; k++) {
        Real bx = std::cos(kB*pco->x3v(k));
        for (int j = js; j <= je; j++) {
          for (int i = is; i <= ie; i++) {
            Real Ta  = T0 + dT_amp*std::cos(kx*(pco->x1v(i) - U*bx*t));
            Real Tn  = pmb->phydro->w(IPR,k,j,i)/pmb->phydro->w(IDN,k,j,i);
            Real err = std::abs(Tn - Ta);
            Real dV  = pco->GetCellVolume(k,j,i);
            l1   += err*dV;
            l2sq += err*err*dV;
            linf  = std::max(linf, err);
          }
        }
      }
      break;}
    case 6: { // ---- frozen-flow Peltier-Seebeck (+ Hall) ----
      Real AT = 0.0; Real ABy = 0.0; Real ABz = 0.0;
      T0 = P_0/rho_0;
      PSModeAmplitudes(t, AT, ABy, ABz);
      for (int k = ks; k <= ke; k++) {
        for (int j = js; j <= je; j++) {
          for (int i = is; i <= ie; i++) {
            Real cx  = std::cos(kx*pco->x1v(i));
            Real Tn  = pmb->phydro->w(IPR,k,j,i)/pmb->phydro->w(IDN,k,j,i);
            Real err = std::abs(Tn - (T0 + AT*cx));
            Real dV  = pco->GetCellVolume(k,j,i);
            l1   += err*dV;
            l2sq += err*err*dV;
            linf  = std::max(linf, err);
          }
        }
      }
      break;}
    default:
      break;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void VelocityErrors
//! \brief Velocity errors e = |v_sim - v_ana| on this MeshBlock:
//  l1 = sum e dV, l2sq = sum e^2 dV, linf = max e.
void VelocityErrors(MeshBlock *pmb, Real &l1, Real &l2sq, Real &linf) {
  l1 = 0.0; l2sq = 0.0; linf = 0.0;

  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;
  Coordinates *pco = pmb->pcoord;
  Real t = pmb->pmy_mesh->time;

  Real ev_x = 0.0; Real ev_y = 0.0; Real ev_z = 0.0; // case 1: damped mode vector
  switch (problem_ID) {
    //  v(r,0) = v0*sin(k.x)*exp(-k^2*nu*t)*(A1 k_hat + A2 b_hat + A3 k_hat x b_hat)
    //  vx(r,t) = v0*A1(0)exp(-(kx^2 t)*nu_comp)sin(kx x)
    //  vy(r,t) = v0*(a2(0)*exp(-(kx^2 t)*nu_sheary)*cos(theta)
    //        a3(0)*exp(-(kx^2 t)*nu_shearz)*sin(theta))*sin(kx x)
    //  vz(r,t) = v0*(a2(0)*exp(-(kx^2 t)*nu_sheary)*sin(theta)
    //        a3(0)*exp(-(kx^2 t)*nu_shearz)*cos(theta))*sin(kx x)
    // nu_comp = zeta_bulk/rho_0 + (4/3)*nu_neu*(1-HII) + HII*(nu_para/3 + nu_perp1)
    // nu_sheary = (1-HII)*nu_neu + 4*HII*nu_perp2
    // nu_shearz = (1-HII)*nu_neu +   HII*nu_perp1
    // Case 1 - Viscoisty
    case 1: {
      Real k2 = kx*kx;
      Real HII = (nu_neu > 0.0) ? ion_frac : 1.0;
      Real M1 = zeta_bulk/rho_0 + (4.0/3.0)*(1.0 - HII)*nu_neu
            + HII*(nu_para/3.0 + nu_perp1);
      Real M2 = (1.0 - HII)*nu_neu + 4.0*HII*nu_perp2;
      Real M3 = (1.0 - HII)*nu_neu + HII*nu_perp1;
      if(pmb->phydro->hdif.nu_iso > 0.0) { // stock athena
        M1 = zeta_bulk/rho_0 + (4.0/3.0)*pmb->phydro->hdif.nu_iso;
        M2 = pmb->phydro->hdif.nu_iso;
        M3 = pmb->phydro->hdif.nu_iso;
      }
      Real A1 = a1*std::exp(-k2*M1*t); // along k
      Real A2 = a2*std::exp(-k2*M2*t); // along b
      Real A3 = a3*std::exp(-k2*M3*t); // along k x b
      ev_x = A1*khat_x + A2*bhat_x + A3*ec_x;
      ev_y = A1*khat_y + A2*bhat_y + A3*ec_y;
      ev_z = A1*khat_z + A2*bhat_z + A3*ec_z;


      for (int k = ks; k <= ke; k++) {
        for (int j = js; j <= je; j++) {
          for (int i = is; i <= ie; i++) {
            Real vx_ana = 0.0; Real vy_ana = 0.0; Real vz_ana = 0.0;
            Real s = v0*std::sin(kx*pco->x1v(i));
            vx_ana = s*ev_x;
            vy_ana = s*ev_y;
            vz_ana = s*ev_z;
            Real dvx = pmb->phydro->w(IVX,k,j,i) - vx_ana;
            Real dvy = pmb->phydro->w(IVY,k,j,i) - vy_ana;
            Real dvz = pmb->phydro->w(IVZ,k,j,i) - vz_ana;
            Real err = std::sqrt(SQR(dvx) + SQR(dvy) + SQR(dvz));
            Real dV = pco->GetCellVolume(k,j,i);
            l1 += err*dV;
            l2sq += err*err*dV;
            linf = std::max(linf, err);
          }
        }
      }
      break;
    }
    // Case 5 - Gyro-viscoisty
    case 5: {
      Real HII = (nu_neu > 0.0) ? ion_frac : 1.0;
      Real W3  = HII*nu_wedge1*kx*kx; // Pi3 rotation rate, k across b
      Real W4  = 2.0*HII*nu_wedge2*kz*kz; // Pi4 rotation rate, k along b
      Real Ax = -v0*std::sin(W3*t), Ay = v0*std::cos(W3*t); // across-b mode
      Real Cx =  v0*std::cos(W4*t), Cy = v0*std::sin(W4*t); // along-b mode
      for (int k = ks; k <= ke; k++) {
        Real sz = std::sin(kz*pco->x3v(k));
        for (int j = js; j <= je; j++) {
          for (int i = is; i <= ie; i++) {
            Real sx = std::sin(kx*pco->x1v(i));
            Real dvx = std::abs(pmb->phydro->w(IVX,k,j,i) - (Ax*sx + Cx*sz));
            Real dvy = std::abs(pmb->phydro->w(IVY,k,j,i) - (Ay*sx + Cy*sz));
            Real dvz = std::abs(0.0 - pmb->phydro->w(IVZ,k,j,i));
            Real err = std::sqrt(SQR(dvx) + SQR(dvy) + SQR(dvz));
            Real dV  = pco->GetCellVolume(k,j,i);
            l1   += err*dV;
            l2sq += err*err*dV;
            linf  = std::max(linf, err);
          }
        }
      }
      break;
    }
    default:
      return;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void FieldErrors
//! \brief Magnetic field errors e = |B_sim - B_ana| (cell-centred) on this MeshBlock:
//!  l1 = sum e dV, l2sq = sum e^2 dV, linf = max e.
//----------------------------------------------------------------------------------------
void FieldErrors(MeshBlock *pmb, Real &l1, Real &l2sq, Real &linf) {
  Coordinates *pco = pmb->pcoord;
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;
  Real t = pmb->pmy_mesh->time;
  l1 = 0.0; l2sq = 0.0; linf = 0.0;

  // case 2 - ohmic+ambi+Hall
  if (problem_ID == 2) {
    Real eta_a = eta_ad*SQR(B0_mag); // ambipolar diffusion coefficient
    Real c  = bhat_x, s = bhat_y;
    Real abar = eta_o + 0.5*eta_a*(1.0 + c*c); // mean damping / K^2
    Real delA = 0.5*eta_a*s*s;         // ambipolar anisotropy / K^2
    Real beta = eta_h*c;               // Hall rotation / K^2 (sign of k.b)
    Real D2   = delA*delA - beta*beta; // D^2
    Real tau  = kx*kx*t;
    Real q  = D2*tau*tau;

    Real EC, ES; // e^{-abar tau} cosh(D tau), e^{-abar tau} sinh(D tau)/D
    if (std::abs(q) < 1.0e-4) { // Taylor series in D^2: no 0/0 at D = 0, limit is tau
      Real decay = std::exp(-abar*tau);
      EC = decay*(1.0 + q/2.0 + q*q/24.0);
      ES = decay*tau*(1.0 + q/6.0 + q*q/120.0);
    } else if (D2 > 0.0) { // ambipolar dominates: D real
      Real D  = std::sqrt(D2);
      Real ep = std::exp(-(abar - D)*tau);
      Real em = std::exp(-(abar + D)*tau);
      EC = 0.5*(ep + em);
      ES = 0.5*(ep - em)/D;
    } else {  // Hall dominates: D = i w
      Real w   = std::sqrt(-D2);
      Real decay = std::exp(-abar*tau);
      EC = decay*std::cos(w*tau);
      ES = decay*std::sin(w*tau)/w;
    }
    Real Ay = -dB_amp*beta*ES; // dBy
    Real Az =  dB_amp*(EC + delA*ES); // dBz

    for (int k = ks; k <= ke; k++) {
      for (int j = js; j <= je; j++) {
      for (int i = is; i <= ie; i++) {
        Real cx  = std::cos(kx*pco->x1v(i));
        Real dbx = std::abs(pmb->pfield->bcc(IB1,k,j,i) - B0_mag*bhat_x);
        Real dby = std::abs(pmb->pfield->bcc(IB2,k,j,i) - (B0_mag*bhat_y + Ay*cx));
        Real dbz = std::abs(pmb->pfield->bcc(IB3,k,j,i) - Az*cx);
        Real err = std::sqrt(SQR(dbx) + SQR(dby) + SQR(dbz));
        Real dV  = pco->GetCellVolume(k,j,i);
        l1   += err*dV;
        l2sq += err*err*dV;
        linf  = std::max(linf, err);
        }
      }
    }
  }

  // case 3 Nernst/Ettinghausen (needs perp conduction and ambi)
  if (problem_ID == 3) {
    T0   = P_0/rho_0;
    Real HII  = (kappa_neu > 0.0) ? ion_frac : 1.0;
    Real DT   = gm1*((1.0 - HII)*kappa_neu + HII*kappa_perp);
    Real DB   = eta_o + eta_ad*SQR(B0_mag);
    Real cTB  = gm1*HII*kappa_TE_wedge/rho_0;    // Ettingshausen
    Real cBT  = HII*kappa_TE_wedge/T0;           // Nernst
    Real abar = 0.5*(DT + DB);                   // mean rate / K^2
    Real del  = 0.5*(DT - DB);                   // half-difference / K^2
    Real D2   = del*del + cTB*cBT;               // D^2 >= 0
    Real tau  = kx*kx*t;
    Real q    = D2*tau*tau;                      // (D tau)^2

    Real EC = 0.0; Real ES = 0.0; // e^{-a tau} cosh(D tau), e^{-a tau} sinh(D tau)/D
    if (q < 1.0e-4) {  // Taylor series in D^2: no 0/0 at D = 0
      Real decay = std::exp(-abar*tau);
      EC = decay*(1.0 + q/2.0 + q*q/24.0);
      ES = decay*tau*(1.0 + q/6.0 + q*q/120.0);
    } else {                   // D real: combine exponents first
      Real D  = std::sqrt(D2);
      Real ep = std::exp(-(abar - D)*tau); // ill-posed (abar < D)
      Real em = std::exp(-(abar + D)*tau);
      EC = 0.5*(ep + em);
      ES = 0.5*(ep - em)/D;
    }

    Real ABy = dB_amp*(EC + del*ES) - dT_amp*cBT*ES;

    for (int k = ks; k <= ke; k++) {
      for (int j = js; j <= je; j++) {
        for (int i = is; i <= ie; i++) {
          Real cx  = std::cos(kx*pco->x1v(i));
          Real dbx = std::abs(0.0 - pmb->pfield->bcc(IB1,k,j,i));
          Real dby = std::abs(pmb->pfield->bcc(IB2,k,j,i) - (B0_mag + ABy*cx));
          Real dbz = std::abs(0.0-pmb->pfield->bcc(IB3,k,j,i));
          Real err = std::sqrt(SQR(dbx) + SQR(dby) + SQR(dbz));
          Real dV  = pco->GetCellVolume(k,j,i);
          l1   += err*dV;
          l2sq += err*err*dV;
          linf  = std::max(linf, err);
        }
      }
    }
  }

  // Case 6 - Pelteri/Seebeck (+ Hall)
  if (problem_ID == 6) {
    Real AT = 0.0; Real ABy = 0.0; Real ABz = 0.0;
    PSModeAmplitudes(t, AT, ABy, ABz);
    for (int k = ks; k <= ke; k++) {
      for (int j = js; j <= je; j++) {
        for (int i = is; i <= ie; i++) {
          Real cx  = std::cos(kx*pco->x1v(i));
          Real dbx = std::abs(pmb->pfield->bcc(IB1,k,j,i) - B0_mag*bhat_x);
          Real dby = std::abs(pmb->pfield->bcc(IB2,k,j,i) - (B0_mag*bhat_y + ABy*cx));
          Real dbz = std::abs(pmb->pfield->bcc(IB3,k,j,i) - ABz*cx);
          Real err = std::sqrt(SQR(dbx) + SQR(dby) + SQR(dbz));
          Real dV  = pco->GetCellVolume(k,j,i);
          l1   += err*dV;
          l2sq += err*err*dV;
          linf  = std::max(linf, err);
        }
      }
    }
  }
  return;
}

// L1 temperature error sum|T_sim-T_ana| dV for convergence tests
Real TemperatureL1(MeshBlock *pmb, int iout) {
  Real l1, l2sq, linf;
  TemperatureErrors(pmb, l1, l2sq, linf);
  return l1;
}

// Squared L2 temperature error sum(T_sim-T_ana)^2 dV
Real TemperatureL2(MeshBlock *pmb, int iout) {
  Real l1, l2sq, linf;
  TemperatureErrors(pmb, l1, l2sq, linf);
  return l2sq;
}

// L_inf temperature error max|T_sim-T_ana|
Real TemperatureLInf(MeshBlock *pmb, int iout) {
  Real l1, l2sq, linf;
  TemperatureErrors(pmb, l1, l2sq, linf);
  return linf;
}

// L1 velocity error sum|v_sim-v_ana| dV for convergence tests
Real VelocityL1(MeshBlock *pmb, int iout) {
  Real l1, l2sq, linf;
  VelocityErrors(pmb, l1, l2sq, linf);
  return l1;
}

// Squared L2 velocity error sum|v_sim-v_ana|^2 dV
Real VelocityL2(MeshBlock *pmb, int iout) {
  Real l1, l2sq, linf;
  VelocityErrors(pmb, l1, l2sq, linf);
  return l2sq;
}

// L_inf velocity error max|v_sim-v_ana|
Real VelocityLInf(MeshBlock *pmb, int iout) {
  Real l1, l2sq, linf;
  VelocityErrors(pmb, l1, l2sq, linf);
  return linf;
}

// L1 field error sum|B_sim-B_ana| dV for convergence tests
Real FieldL1(MeshBlock *pmb, int iout) {
  Real l1, l2sq, linf;
  FieldErrors(pmb, l1, l2sq, linf);
  return l1;
}

// Squared L2 field error sum|B_sim-B_ana|^2 dV
Real FieldL2(MeshBlock *pmb, int iout) {
  Real l1, l2sq, linf;
  FieldErrors(pmb, l1, l2sq, linf);
  return l2sq;
}

// L_inf field error max|B_sim-B_ana|
Real FieldLInf(MeshBlock *pmb, int iout) {
  Real l1, l2sq, linf;
  FieldErrors(pmb, l1, l2sq, linf);
  return linf;
}

void PSModeAmplitudes(Real t, Real &AT, Real &ABy, Real &ABz);
//----------------------------------------------------------------------------------------
//! \fn void PSModeAmplitudes
//! \brief Case 6 amplitudes from the sympy solution (frozen flow, Peltier-Seebeck with
//  optional Hall), tau = K^2 t:
//  dT(t)  = dT0 [1 - (c_TB c_BT/Omega^2)(1 - cos(Omega tau))]
//  dBy(t) = dT0 beta c_BT (1 - cos(Omega tau))/Omega^2
//  dBz(t) = -dT0 c_BT sin(Omega tau)/Omega
//  c_TB = (gamma-1) x (kJpara - kJperp) sin cos / rho_0,
//  c_BT = x (kJpara - kJperp) sin cos / T0, //  beta = eta_H cos,
//  Omega^2 = beta^2 + c_TB c_BT.  Only kJpara - kJperp enters.
void PSModeAmplitudes(Real t, Real &AT, Real &ABy, Real &ABz) {
  T0     = P_0/rho_0;
  Real HII  = (kappa_neu > 0.0) ? ion_frac : 1.0;  // ion weighting, as in the TE solvers
  Real c    = bhat_x; Real s = bhat_y;             // b direction
  Real dkJ  = HII*(kappa_TE_para - kappa_TE_perp); // only the anisotropy acts
  Real cTB  = gm1*dkJ*s*c/rho_0;                   // Peltier
  Real cBT  = dkJ*s*c/T0;                          // Seebeck
  Real beta = eta_h*c;                             // Hall
  Real W2   = beta*beta + cTB*cBT;                 // Omega^2 >= 0
  Real tau  = kx*kx*t;
  Real q    = W2*tau*tau;

  Real S1 = 0.0; Real C1 = 0.0; // sin(Omega tau)/Omega, (1 - cos(Omega tau))/Omega^2
  if (q < 1.0e-4) {             // Taylor series in Omega^2: no 0/0 at Omega = 0
    S1 = tau*(1.0 - q/6.0 + q*q/120.0);
    C1 = 0.5*tau*tau*(1.0 - q/12.0 + q*q/360.0);
  } else {
    Real W = std::sqrt(W2);
    Real h = std::sin(0.5*W*tau);
    S1 = std::sin(W*tau)/W;
    C1 = 2.0*h*h/W2;            // 1 - cos without cancellation
  }
  AT  = dT_amp*(1.0 - cTB*cBT*C1);
  ABy = dT_amp*beta*cBT*C1;
  ABz = -dT_amp*cBT*S1;
  return;
}
