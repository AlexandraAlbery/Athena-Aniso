// C++ headers
#include <algorithm>  // min, max
#include <cmath>      // sqrt()
#include <cstdio>     // fopen(), fprintf(), freopen()
#include <ios>
#include <iostream>   // endl
#include <random>
#include <stdexcept>  // runtime_error
#include <string>     // c_str()
#include <sstream>
#include <vector>     // vector type
#include <unistd.h>


// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../fft/athena_fft.hpp"
#include "../fft/turbulence.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../reconstruct/reconstruction.hpp"
#include "../scalars/scalars.hpp"
#include "../utils/utils.hpp"
#include "../bvals/bvals.hpp"
#include "../outputs/outputs.hpp"
#include "../outputs/io_wrapper.hpp"

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
int problem_ID = 0 ; // tracks which test to run. called in MeshBlock::UserMeshBlockData

// case 0
Real T_0 = 0; Real A_T = 0;
int  nx  = 0; int  ny  = 0; int  nz  = 0;
Real kx  = 0; Real ky  = 0; Real kz  = 0;

// Array index for hdf5 output
const int out_T          = 0;  // temperature
const int out_sound      = 1;  // adiabatic sound speed
const int out_alfven     = 2;  // alfven speed
const int out_normAlfven = 3;  // normal (x-face) alfven speed
const int out_slowMag    = 4;  // slow magnetosonic speed
const int out_fastMag    = 5;  // fast magnetosonic speed
const int out_whist      = 6;  // hall effect whistler wave
const int out_ioncyc     = 7;  // hall effect ion cyclotron speed
const int out_TEwave     = 8;  // Peltier/Seebeck speed
const int out_TEWwave    = 9;  // Peltier/Seebeck + Hall wavespeed
const int out_GVwave     = 10; // gyro-viscous wave speed
const int out_RLspeed    = 11; // righi-leduc advection speed

/*
const int out_dt_ideal     = 12; // ideal flow
const int out_dt_vis_para  = 13; // parabolic viscosity
const int out_dt_cnd_para  = 14; // parabolic thermal conduction
const int out_dt_vis_hyp   = 15; // hyperbolic (gyro) viscosity
const int out_dt_cnd_hyp   = 16; // hyperbolic (RL) thermal conduction
const int out_dt_cnd_tot   = 17; // total thermal conduction
const int out_dt_vis_tot   = 18; // total viscosity
const int out_dt_oah       = 19; // ohmic + ambipolar + hall (no TE)
const int out_dt_TE        = 20; // unsplit isolated TE (P/S + N/E)
const int out_dt_TE_hyp    = 21; // Hall+P/S
const int out_dt_total     = 22; // Actual timestep
*/
const int out_WP           = 12;
};

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
Real TemperatureIntrgral(MeshBlock *pmb, int iout);
Real VelocityIntegral(MeshBlock *pmb, int iout);
Real FieldIntegral(MeshBlock *pmb, int iout);

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

    AllocateUserHistoryOutput(3);
    EnrollUserHistoryOutput(0, TemperatureIntrgral, "Temp_int");
    EnrollUserHistoryOutput(1, VelocityIntegral, "vel_int");
    EnrollUserHistoryOutput(2, FieldIntegral, "B_int");

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
                    if(k <=ke && j<=je && i <= ie)
                    {
                        phydro->u(IEN,k,j,i) += B_energy;
                    }
                }
            }
        }

    // integer to decide what test to run.
    // Make sure to comment/zero out unwanted diffusions in the input file
    //   case 0: single Fourier mode in T at uniform P
    //   case 1: amplitude from input (default 1e-6); velocity along k, z-hat x k, or z-hat; tilt via mode_nx/ny
    //   case 2: Hall phase speed in the eigenvector
    //   case 3: TE eigenmode: k perpendicular to b (Nernst-Ettingshausen)
    //   case 4: Beltrami field for RL
    //   case 5: timestep sweep

    switch (problem_ID)
    {
        case 0:{ // ---- single thermal mode: conduction, Peltier-Seebeck (+ Hall) ----
            T_0 = P_0/rho_0;
            A_T = pin->GetOrAddReal("problem","mode_amp_T", 1.0e-6);
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
                        T_local = T_0*(1.0 + A_T*std::cos(kx*x + ky*y + kz*z));
                        phydro->u(IDN,k,j,i) = P_0/T_local;        // pressure equilibrium
                        phydro->u(IEN,k,j,i) = P_0/gm1 + B_energy; // uniform
                    } // i loop
                } // j loop
            } // k loop
            break;}
    case 1:{ // ---- single velocity mode: viscosity and gyroviscosity ----
    Real v0   = pin->GetOrAddReal("problem","visc_amp", 1.0e-6);
    int  vdir = pin->GetOrAddInteger("problem","visc_dir", 2); // 1: along k, 2: z-hat x k-hat (in-plane shear), 3: along z
    int  nx   = pin->GetOrAddInteger("problem","mode_nx", 1);
    int  ny   = pin->GetOrAddInteger("problem","mode_ny", 0);
    Real kx   = 2.0*PI*static_cast<Real>(nx)/(xmax - xmin);
    Real ky   = 2.0*PI*static_cast<Real>(ny)/(ymax - ymin);
    Real kmag = std::sqrt(SQR(kx) + SQR(ky));
    Real x  = 0; Real y = 0; Real vel = 0;

    if (kmag == 0.0 || vdir < 1 || vdir > 3) {
        std::stringstream msg;
        msg << "### FATAL ERROR in ProblemGenerator case 1" << std::endl
            << "need mode_nx or mode_ny nonzero, and visc_dir = 1, 2 or 3" << std::endl;
        ATHENA_ERROR(msg);
    }

    // unit vector of the velocity: k-hat (compressive), z-hat x k-hat (in-plane shear), or z-hat
    Real ex = (vdir == 1) ? kx/kmag : ((vdir == 2) ? -ky/kmag : 0.0);
    Real ey = (vdir == 1) ? ky/kmag : ((vdir == 2) ?  kx/kmag : 0.0);
    Real ez = (vdir == 3) ? 1.0 : 0.0;

    for (int k = ks; k <= ke; k++ ) {
        for (int j = js; j <= je; j++ ) {
            y = this->pcoord->x2v(j);
            for (int i = is; i <= ie; i++ ) {
                x = this->pcoord->x1v(i);
                vel = v0*std::sin(kx*x + ky*y);
                phydro->u(IDN,k,j,i) = rho_0;
                phydro->u(IM1,k,j,i) = rho_0*vel*ex;
                phydro->u(IM2,k,j,i) = rho_0*vel*ey;
                phydro->u(IM3,k,j,i) = rho_0*vel*ez;
                phydro->u(IEN,k,j,i) += 0.5*rho_0*vel*vel;
            }
        }
    }
        break;}
    case 2:{
        // ---- Alfven wave: Hall (circular) or Ohmic/ambipolar (linear along z) ----
        // pol = +1/-1: circularly polarised, needs By_0 = 0; exact Hall eigenmode travelling at vph.
        // pol = 0    : linearly polarised along z, any By_0 (field in the x-y plane); exact ideal
        //              eigenmode at vA = |Bx_0|/sqrt(rho_0), static (pure Ohmic decay) if Bx_0 = 0.
        Real L    = xmax - xmin;
        Real K    = 2.0*PI/L;     // one wavelength across box (periodic in x)
        Real dB   = 1.0e-6 * std::sqrt(Bx_0*Bx_0 + By_0*By_0 + Bz_0*Bz_0);
        Real pol  = pin->GetOrAddReal("problem","alfven_pol", 1.0); // polarisation: +1, -1 or 0
        Real vA   = std::sqrt((Bx_0*Bx_0)/rho_0); // Alfvén speed on the normal (x) field
        Real sgn  = (Bx_0 > 0.0) ? 1.0 : -1.0;
        Real etaH = eta_h; // Hall diffusivity on the background
        // forward root of v^2 + sgn*pol*etaH*K*v - vA^2 = 0; equals vA when etaH = 0
        Real vph  = (pol != 0.0) ? -0.5*sgn*pol*etaH*K + std::sqrt(SQR(0.5*etaH*K) + SQR(vA)) : vA;
        Real vfac = (vph != 0.0) ? -Bx_0/(rho_0*vph) : 0.0;
        Real x    = 0;

        if (Bz_0 != 0.0 || (pol != 0.0 && By_0 != 0.0)) {
            std::stringstream msg;
            msg << "### FATAL ERROR in ProblemGenerator case 2" << std::endl
                << "Alfven wave requires Bz_0 = 0, and By_0 = 0 unless alfven_pol = 0"
                << std::endl;
            ATHENA_ERROR(msg);
        }

        // eigenmode: v_perp = -(Bx_0/(rho_0*vph)) * b_perp  (+x propagation; = -(vA/Bx_0) b_perp when vph = vA)
        for (int kk = ks; kk <= ke; kk++ ) {
            for (int j = js; j <= je; j++ ) {
                for (int i = is; i <= ie; i++ ) {
                    x = this->pcoord->x1v(i);
                    Real By_pert = (pol != 0.0) ? dB*std::cos(K*x) : 0.0;
                    Real Bz_pert = (pol != 0.0) ? pol*dB*std::sin(K*x) : dB*std::cos(K*x);
                    Real vy = vfac*By_pert;
                    Real vz = vfac*Bz_pert;

                    phydro->u(IDN,kk,j,i) = rho_0;
                    phydro->u(IM1,kk,j,i) = 0.0;
                    phydro->u(IM2,kk,j,i) = rho_0*vy;
                    phydro->u(IM3,kk,j,i) = rho_0*vz;
                    phydro->u(IEN,kk,j,i) += + 0.5*rho_0*(vy*vy + vz*vz);
                }
            }
        }
        for (int k = ks; k <= ke+1; k++ ) {
            for (int j = js; j <= je+1; j++ ) {
                for (int i = is; i <= ie+1; i++ ) {
                    Real xf = this->pcoord->x1v(i);   // x at cell centre for x2f/x3f
                    Real b2 = (pol != 0.0) ? dB*std::cos(K*xf) : 0.0;
                    Real b3 = (pol != 0.0) ? pol*dB*std::sin(K*xf) : dB*std::cos(K*xf);
                    pfield->b.x1f(k,j,i) = Bx_0;       // uniform normal field
                    pfield->b.x2f(k,j,i) = By_0 + b2;
                    pfield->b.x3f(k,j,i) = b3;
                    if (k<=ke && j<=je && i<=ie) {
                        phydro->u(IEN,k,j,i) += 0.5*(b2*b2 + b3*b3);
                    }
                }
            }
        }
        break;}
        case 3:{// ---- TE eigenmode: k perpendicular to b (Nernst-Ettingshausen) ----
            Real T_0    = P_0/rho_0;
            Real L      = xmax - xmin;
            Real A      = pin->GetOrAddReal("problem","te_eig_amp",  1.0e-4);
            int  nx     = pin->GetOrAddInteger("problem","te_eig_nx", 1);
            int  nz     = pin->GetOrAddInteger("problem","te_eig_nz", 0);
            // branch: +1 = fast (always decaying), -1 = slow (crosses zero at threshold)
            int  branch = pin->GetOrAddInteger("problem","te_eig_branch", -1);

            if (Bx_0 != 0.0 || Bz_0 != 0.0 || By_0 == 0.0) {
                std::stringstream msg;
                msg << "### FATAL ERROR in ProblemGenerator case 3" << std::endl
                    << "TE eigenmode requires b along y: Bx_0 = Bz_0 = 0, By_0 != 0" << std::endl;
                ATHENA_ERROR(msg);
            }

            Real kx = 2.0*PI*static_cast<Real>(nx)/L;
            Real kz = 2.0*PI*static_cast<Real>(nz)/(zmax - zmin);
            Real k2 = SQR(kx) + SQR(kz);
            if (k2 == 0.0) {
                std::stringstream msg;
                msg << "### FATAL ERROR in ProblemGenerator case 3" << std::endl
                    << "te_eig_nx and te_eig_nz cannot both be zero" << std::endl;
                ATHENA_ERROR(msg);
            }

            // frozen-flow coefficient matrix (constant coefficients). With the hydro on this 2x2 is
            // exact only for the pressure-neutral mode printed below; otherwise the reference is e^{At}.
            Real kJw     = kappa_TE_wedge*ion_frac;
            Real kperp   = kappa_perp*ion_frac;
            Real eta_tot = eta_o + eta_ad*(SQR(Bx_0) + SQR(By_0) + SQR(Bz_0));

            Real D_T  = gm1*kperp/rho_0;
            Real D_B  = eta_tot;
            Real c_TB = gm1*kJw/rho_0;
            Real c_BT = kJw/T_0;

            Real half  = 0.5*(D_T + D_B);
            Real disc  = SQR(0.5*(D_T - D_B)) + c_TB*c_BT;
            Real mu    = half + static_cast<Real>(branch)*std::sqrt(disc);

            // eigenvector ratio dB/dT; c_TB -> 0 means the channels decouple
            Real ratio = (std::abs(c_TB) > 0.0) ? (mu - D_T)/c_TB : 0.0;

            // report the analytic rate so the run is self-documenting
            if (Globals::my_rank == 0 && gid == 0) {
                std::cout << "### case 3 inputs: gm1=" << gm1
                << "  kperp=" << kperp
                << "  kJw=" << kJw
                << "  eta_tot=" << eta_tot
                << std::endl;
                std::cout << "### well-posedness: D_T*D_B = " << D_T*D_B
                        << "  kJw^2/eps_int = "  << SQR(kJw)/(P_0/gm1)
                        << "  ratio = "          << D_T*D_B/(SQR(kJw)/(P_0/gm1)) << std::endl;
                // pressure-neutral mode: dB/dT = -rho_0/By_0 keeps d_rho = 0 and dP + By_0 dB = 0 for all t
                std::cout << "### frozen-flow 2x2 (not the reference unless pressure-neutral): mu = " << mu
                        << "  dB/dT = " << ratio << std::endl;
                std::cout << "### pressure-neutral mode needs eta_tot = "
                        << D_T + kJw*(SQR(By_0) - gm1*P_0)/(P_0*By_0)
                        << "  giving mu = " << D_T - gm1*kJw/By_0
                        << "  with dB/dT = " << -rho_0/By_0 << std::endl;
            }

            // ---- face-centred B first: B = (B_0 + dB) y-hat, div(B) = 0 identically ----
            for (int k = ks; k <= ke+1; k++) {
                Real z = pcoord->x3v((k <= ke) ? k : ke);
                for (int j = js; j <= je+1; j++) {
                for (int i = is; i <= ie+1; i++) {
                    Real x = pcoord->x1v((i <= ie) ? i : ie);
                    pfield->b.x1f(k,j,i) = 0.0;
                    pfield->b.x2f(k,j,i) = By_0 + A*ratio*std::cos(kx*x + kz*z);
                    pfield->b.x3f(k,j,i) = 0.0;
                }
                }
            }

            // ---- cell-centred state ----
            for (int k = ks; k <= ke; k++) {
                Real z = pcoord->x3v(k);
                for (int j = js; j <= je; j++) {
                for (int i = is; i <= ie; i++) {
                    Real x     = pcoord->x1v(i);
                    Real phase = std::cos(kx*x + kz*z);
                    Real dT    = A*phase;
                    Real dB    = A*ratio*phase;

                    // total-pressure balance: dP + B_0 dB = 0, with dP = rho_0 dT + T_0 d_rho
                    Real d_rho = -(By_0*dB + rho_0*dT)/T_0;
                    Real rho   = rho_0 + d_rho;
                    Real T     = T_0 + dT;
                    Real Bmag  = By_0 + dB;

                    phydro->u(IDN,k,j,i) = rho;
                    phydro->u(IM1,k,j,i) = 0.0;
                    phydro->u(IM2,k,j,i) = 0.0;
                    phydro->u(IM3,k,j,i) = 0.0;
                    phydro->u(IEN,k,j,i) = rho*T/gm1 + 0.5*SQR(Bmag); // constant total pressure

                    ruser_meshblock_data[0](k,j,i) = T;   // T_0 snapshot for the diagnostic
                }
                }
            }
            break;}
        case 4:{ // --- Righi-Leduc: Beltrami field + transverse T gradient ---
            // b = (cos kBz, sin kBz, 0): curl b = -kB b, torsion -kB, curvature 0.
            // Exact solution for the RL operator alone (rho, kappa_w const):
            // T(x,z,t) = T_init(x - (gamma-1)*kappa_w*kB*cos(kB z)*t, z)
            // div(curl b) = 0, advection is incompressible.
            // int T^2 dV is conserved.
            Real T_0    = P_0/rho_0;
            Real A_T    = pin->GetOrAddReal("problem","rl_amp_T", 1.0e-3);
            Real A_B    = pin->GetOrAddReal("problem","rl_amp_B", 0.0);
            Real v0     = pin->GetOrAddReal("problem","rl_amp_v", 0.0);
            Real nturn  = pin->GetOrAddReal("problem","rl_turns", 1.0);
            Real Lx     = xmax - xmin;
            Real Lz     = zmax - zmin;
            Real kB     = 2.0*PI*nturn/Lz;
            Real kT     = 2.0*PI/Lx;
            Real B0     = std::sqrt(SQR(Bx_0) + SQR(By_0) + SQR(Bz_0));

            if (A_B >= 1.0) {
                std::stringstream msg;
                msg << "### FATAL ERROR in ProblemGenerator case 4" << std::endl
                    << "rl_amp_B must be < 1 or the field reverses" << std::endl;
                ATHENA_ERROR(msg);
            }

            // Field: g depends on z only, so div(B) = 0 discretely for any g.
            for (int k = ks; k <= ke+1; k++) {
                Real z = this->pcoord->x3v(k);
                Real g = 1.0 - A_B*std::sin(kB*z);
                for (int j = js; j <= je+1; j++) {
                    for (int i = is; i <= ie+1; i++) {
                        pfield->b.x1f(k,j,i) = B0*g*std::cos(kB*z);
                        pfield->b.x2f(k,j,i) = B0*g*std::sin(kB*z);
                        pfield->b.x3f(k,j,i) = 0.0;
                    }
                }
            }
            for (int k = ks; k <= ke; k++) {
                Real z = this->pcoord->x3v(k);
                Real g = 1.0 - A_B*std::sin(kB*z);
                for (int j = js; j <= je; j++) {
                    for (int i = is; i <= ie; i++) {
                        Real x  = this->pcoord->x1v(i);
                        Real T_local = T_0*(1.0 + A_T*std::sin(kT*x));
                        Real vy = v0*std::sin(kT*x);
                        phydro->u(IDN,k,j,i) = P_0/T_local;
                        phydro->u(IM1,k,j,i) = 0.0;
                        phydro->u(IM2,k,j,i) = rho_0*vy;
                        phydro->u(IM3,k,j,i) = 0.0;
                        phydro->u(IEN,k,j,i) = P_0/gm1 + 0.5*rho_0*vy*vy + 0.5*SQR(B0*g);
                    }
                }
            }
            break;}
        case 5:{
                // ---- (T, rho, |B|) sweep for the timestep check ----
                // x -> T,  y -> rho,  z -> |B|.  Log-spaced about the base state.
                // Run single-block with time/nlim=0.
                // Run with uniform cartesian grid
                Real dx = pcoord->dx1f(1);
                Real T_0 = P_0/rho_0;
                Real B_0 = std::abs(Bx_0);

                int nT = ie - is + 1, nR = je - js + 1, nB = ke - ks + 1;
                AthenaArray<Real> Tg, Rg, Bg;
                Tg.NewAthenaArray(nT);  Rg.NewAthenaArray(nR);  Bg.NewAthenaArray(nB);
                for (int n = 0; n < nT; n++) Tg(n)   = T_0*std::pow(10.0, dx*(n - 0.5*(nT-1)));
                for (int n = 0; n < nR; n++) Rg(n)   = rho_0*std::pow(10.0, dx*(n - 0.5*(nR-1)));
                for (int n = 0; n < nB; n++) Bg(n)   = B_0*std::pow(10.0, dx*(n - 0.5*(nB-1)));

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
        << "0 - temperature gradient (thermal conduction, Peltier/Seebeck (+Hall)), \n "
        << "1 - velocity Fourier mode (viscoisty), \n"
        << "2 - MHD wave (Ohmic + ambi, Hall effect), \n"
        << "3 - TE eigenmode (Nernst-Ettingshausen) \n"
        << "4 - Righi-Leduc: Beltrami field + transverse T gradient \n"
        << "5 - Timestep checks \n";
        exit(-1);
    }

 // Output initial conditions
    if(Globals::my_rank ==0 && gid == 0){
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
    std::cout << "\t\t" << "CLF number = " << CFL << " arb. u. \n" ;
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
        std::cout << "MeshBlock::UserWorkInLoop called \n" ;
    }

    GetTemperature(this);

    return;
}

void Mesh::UserWorkInLoop() {

    if(time == 0 && Globals::my_rank == 0) {
        std::cout << "Mesh::UserWorkInLoop called \n" ;
    }

    return;
}

void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin) {

    if(gid == 0 && pmy_mesh->time == 0) {
        std::cout << "MeshBlock::UserWorkBeforeOutput called \n" ;
    }

    GetTemperature(this);
    GetWaveSpeeds(this);
    if(problem_ID==3)CalculateWellPosedness(this);
    //if(problem_ID==5) GetTimeSteps(this);

    return;
}

void Mesh::UserWorkAfterLoop(ParameterInput *pin) {

    if(time == 0 && Globals::my_rank == 0) {
        std::cout << "Mesh::UserWorkAfterLoop called \n" ;
    }
    return;
}

//----------------------------------------------------------------------------------------
//! \fn void GetTemperature
//! \brief Fills MeshBlock::user_out_var[out_temperature]
void GetTemperature(MeshBlock *pmb){

    // Get position indicies
    int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
    int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

    // Get ghost cell indicies
    int il = is - NGHOST; int jl = js - NGHOST; int kl = ks - NGHOST;
    int iu = ie + NGHOST; int ju = je + NGHOST; int ku = ke + NGHOST;

    for (int k = kl; k<= ku; k++){
        for (int j = jl; j<= ju; j++){
            for (int i = il; i<= iu; i++){
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

    for (int k = ks; k<= ke; k++){
        for (int j = js; j<= je; j++){
            for (int i = is; i<= ie; i++){
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
                } else{cw = 0; ci = 0;}

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
void GetEntropyProduction(MeshBlock *pmb){

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
     for (int k = kl; k<= ku; k++){
        for (int j = jl; j<= ju; j++){
            for (int i = il; i<= iu; i++){
                T(k,j,i) = pmb->user_out_var(out_T,k,j,i);
                }
            }
        }

    // Rebuild qJ and FT at cell centres
    for (int k = ks; k<= ke; k++){
        for (int j = js; j<= je; j++){
            for (int i = is; i<= ie; i++){
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
//                + FTx(k+1,j,i) + FTx(k+1,j+1,i));
// Real FTy_cc = 0.25*(FTy(k,j,i) + FTy(k,j,i+1)
//                + FTy(k+1,j,i) + FTy(k+1,j,i+1));
// Real FTz_cc = 0.25*(FTz(k,j,i) + FTz(k,j+1,i)
//                 + FTz(k,j,i+1) + FTz(k,j+1,i+1));

// Entropy production rate
// pmb->user_out_var(out_entropyProd,k,j,i) = (qJx_cc*dInvTdx + qJy_cc*dInvTdy
//                                                             + qJz_cc*dInvTdz)
//         + invT*(FTx_cc*jcc(IB1,k,j,i) + FTy_cc*jcc(IB2,k,j,i) + FTz_cc*jcc(IB3,k,j,i));
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
            fdif.thermKappa(fdif.kappa_TE_para,k,j,i) = kj_perp;
            fdif.thermKappa(fdif.kappa_TE_para,k,j,i) = kj_wedge;
            }
        }
    }


    return;
}

void CalculateWellPosedness(MeshBlock *pmb){

    // Get position indicies
    int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
    int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

    Real gm1 = pmb->peos->GetGamma() - 1.0;

    for (int k = ks; k <= ke; k++) {
        for (int j = js; j <= je; j++) {
            for (int i = is; i <= ie; i++) {
                Real HII = pmb->phydro->hdif.ion_frac(k,j,i);
                Real kn = pmb->phydro->hdif.kappa(HydroDiffusion::kappa_neutral,k,j,i);
                Real kp = pmb->phydro->hdif.kappa(HydroDiffusion::kappa_perp,k,j,i);
                Real eta_o = pmb->pfield->fdif.etaB(FieldDiffusion::ohmic,k,j,i);
                Real eta_ad = pmb->pfield->fdif.etaB(FieldDiffusion::ambipolar,k,j,i);
                Real kjw = pmb->pfield->fdif.thermKappa(FieldDiffusion::ThermElecProcess::
                    kappa_TE_wedge,k,j,i);

                Real wp = gm1*((1-HII)*kn + HII*kp)*(eta_o+eta_ad) /
                          kjw*kjw*HII*HII/pmb->phydro->w(IEN,k,j,i)/gm1;
                pmb->user_out_var(out_WP,k,j,i) = wp;

            }
        }
    }

    return;
}

// show ill-posed dx scaling by holding dt constant until something breaks
Real TETimeStep(MeshBlock *pmb){
    return 1e-4;
}

// Calculates the temperature integral (T-T0)^2 dV for TE convergence test
Real TemperatureIntrgral(MeshBlock *pmb, int iout) {

    int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
    int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;
    Real T_int = 0;
    Real T_sim = 0;
    Real T_ana = 0;
    Real t = 0;
    AthenaArray<Real> &T0 = pmb->ruser_meshblock_data[0];
    Real dV = 0;
    for (int k = ks; k <= ke; k++) {
        for (int j = js; j <= je; j++) {
            for (int i = is; i <= ie; i++) {
                dV = pmb->pcoord->GetCellVolume(k,j,i);
                T_sim = pmb->phydro->w(IPR,k,j,i)/pmb->phydro->w(IDN,k,j,i);
                T_ana = T0(k,j,i)*(1.0+std::exp(-gm1*kappa_neu*kx*kx*pmb->pmy_mesh->time)
                                *std::cos(kx*pmb->pcoord->x1v(i)));
                T_int += (T_sim-T_ana)*(T_sim-T_ana)*dV;
            }
        }
    }
    return T_int;
}

Real VelocityIntegral(MeshBlock *pmb, int iout) {
    Real v_int = 0.0;
    AthenaArray<Real> &w = pmb->phydro->w;
    for (int k = pmb->ks; k <= pmb->ke; k++)
      for (int j = pmb->js; j <= pmb->je; j++)
        for (int i = pmb->is; i <= pmb->ie; i++)
          v_int += (SQR(w(IVX,k,j,i)) + SQR(w(IVY,k,j,i)) + SQR(w(IVZ,k,j,i)))
               * pmb->pcoord->GetCellVolume(k,j,i);
    return v_int;
  }

  Real FieldIntegral(MeshBlock *pmb, int iout) {
    Real B_int = 0.0;
    AthenaArray<Real> &bcc = pmb->pfield->bcc;
    for (int k = pmb->ks; k <= pmb->ke; k++)
      for (int j = pmb->js; j <= pmb->je; j++)
        for (int i = pmb->is; i <= pmb->ie; i++)
          B_int += (SQR(bcc(IB1,k,j,i) - Bx_0) + SQR(bcc(IB2,k,j,i) - By_0)
              + SQR(bcc(IB3,k,j,i) - Bz_0)) * pmb->pcoord->GetCellVolume(k,j,i);
    return B_int;
  }