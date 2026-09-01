//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file hlle_mod.cpp
//! \brief HLLE Riemann solver for MHD. See the hydro version for details.

// C headers

// C++ headers
#include <algorithm>  // max(), min()
#include <cmath>      // sqrt()

// Athena++ headers
#include "../../../athena.hpp"
#include "../../../athena_arrays.hpp"
#include "../../../coordinates/coordinates.hpp"
#include "../../../eos/eos.hpp"
#include "../../../field/field.hpp"
#include "../../../field/field_diffusion/field_diffusion.hpp"
#include "../../hydro.hpp"
#include "../../hydro_diffusion/hydro_diffusion.hpp"

//----------------------------------------------------------------------------------------
//! \fn void Hydro::RiemannSolver
//! \brief HLLE with modified wave speeds for dispersive non-ideal terms

void Hydro::RiemannSolver(const int k, const int j, const int il, const int iu,
                          const int ivx, const AthenaArray<Real> &bx,
                          AthenaArray<Real> &wl, AthenaArray<Real> &wr,
                          AthenaArray<Real> &flx,
                          AthenaArray<Real> &ey, AthenaArray<Real> &ez,
                          AthenaArray<Real> &wct, const AthenaArray<Real> &dxw) {
  int ivy = IVX + ((ivx-IVX)+1)%3;
  int ivz = IVX + ((ivx-IVX)+2)%3;

  // cell offsets to the left cell across the face, per sweep direction
  int oi = (ivx == IVX) ? 1 : 0;
  int oj = (ivx == IVY) ? 1 : 0;
  int ok = (ivx == IVZ) ? 1 : 0;

  Real wli[(NWAVE)], wri[(NWAVE)], wroe[(NWAVE)];
  Real fl[(NWAVE)],fr[(NWAVE)],flxi[(NWAVE)];

  Real gm1 = pmy_block->peos->GetGamma() - 1.0;
  Real iso_cs = pmy_block->peos->GetIsoSoundSpeed();
  Real dt = pmy_block->pmy_mesh->dt;

  FieldDiffusion &fdif_ = pmy_block->pfield->fdif;
  HydroDiffusion &hdif_ = pmy_block->phydro->hdif;
  Coordinates *pco = pmy_block->pcoord;

  const bool f_on = fdif_.field_diffusion_defined;
  const bool h_on = hdif_.hydro_diffusion_defined;
  const bool i_on = hdif_.kappa_aniso_neutral > 0.0 || hdif_.nu_aniso_neutral > 0.0;


#pragma omp simd private(wli,wri,wroe,fl,fr,flxi)
  for (int i=il; i<=iu; ++i) {
    int kl = k-ok; int jl = j-oj; int il = i-oi;
    //--- Step 1. Load L/R states into local variables

    wli[IDN]=wl(IDN,i);
    wli[IVX]=wl(ivx,i);
    wli[IVY]=wl(ivy,i);
    wli[IVZ]=wl(ivz,i);
    if (NON_BAROTROPIC_EOS) wli[IPR]=wl(IPR,i);
    wli[IBY]=wl(IBY,i);
    wli[IBZ]=wl(IBZ,i);

    wri[IDN]=wr(IDN,i);
    wri[IVX]=wr(ivx,i);
    wri[IVY]=wr(ivy,i);
    wri[IVZ]=wr(ivz,i);
    if (NON_BAROTROPIC_EOS) wri[IPR]=wr(IPR,i);
    wri[IBY]=wr(IBY,i);
    wri[IBZ]=wr(IBZ,i);

    Real bxi = bx(k,j,i);

    //--- Step 2. Compute Roe-averaged state

    Real sqrtdl = std::sqrt(wli[IDN]);
    Real sqrtdr = std::sqrt(wri[IDN]);
    Real isdlpdr = 1.0/(sqrtdl + sqrtdr);

    wroe[IDN] = sqrtdl*sqrtdr;
    wroe[IVX] = (sqrtdl*wli[IVX] + sqrtdr*wri[IVX])*isdlpdr;
    wroe[IVY] = (sqrtdl*wli[IVY] + sqrtdr*wri[IVY])*isdlpdr;
    wroe[IVZ] = (sqrtdl*wli[IVZ] + sqrtdr*wri[IVZ])*isdlpdr;
    // Note Roe average of magnetic field is different
    wroe[IBY] = (sqrtdr*wli[IBY] + sqrtdl*wri[IBY])*isdlpdr;
    wroe[IBZ] = (sqrtdr*wli[IBZ] + sqrtdl*wri[IBZ])*isdlpdr;
    Real x = 0.5*(SQR(wli[IBY]-wri[IBY]) + SQR(wli[IBZ]-wri[IBZ]))/(SQR(sqrtdl+sqrtdr));
    Real y = 0.5*(wli[IDN] + wri[IDN])/wroe[IDN];

    // Following Roe(1981), the enthalpy H=(E+P)/d is averaged for adiabatic flows,
    // rather than E or P directly. sqrtdl*hl = sqrtdl*(el+pl)/dl = (el+pl)/sqrtdl
    Real pbl = 0.5*(bxi*bxi + SQR(wli[IBY]) + SQR(wli[IBZ]));
    Real pbr = 0.5*(bxi*bxi + SQR(wri[IBY]) + SQR(wri[IBZ]));
    Real el,er,hroe;
    if (NON_BAROTROPIC_EOS) {
      el = wli[IPR]/gm1 + 0.5*wli[IDN]*(SQR(wli[IVX])+SQR(wli[IVY])+SQR(wli[IVZ])) +pbl;
      er = wri[IPR]/gm1 + 0.5*wri[IDN]*(SQR(wri[IVX])+SQR(wri[IVY])+SQR(wri[IVZ])) +pbr;
      hroe = ((el + wli[IPR] + pbl)/sqrtdl + (er + wri[IPR] + pbr)/sqrtdr)*isdlpdr;
    }

    //--- Step 3. Compute fast magnetosonic speed in L,R, and Roe-averaged states

    Real cl = pmy_block->peos->FastMagnetosonicSpeed(wli,bxi);
    Real cr = pmy_block->peos->FastMagnetosonicSpeed(wri,bxi);
    Real cal = pmy_block->peos->AlfvenSpeed(wli,bxi);
    Real car = pmy_block->peos->AlfvenSpeed(wri,bxi);

    //--- Step 3a. Hall+Peltier/Seebeck coefficeints
    Real etaHl = f_on ? std::abs(fdif_.etaB(fdif_.hall,kl,jl,il)) : 0.0;
    Real etaHr = f_on ? std::abs(fdif_.etaB(fdif_.hall,k,j,i)) : 0.0;
    Real HIIl = i_on ? hdif_.ion_frac(kl,jl,il) : 1.0;
    Real HIIr = i_on ? hdif_.ion_frac(k,j,i) : 1.0;
    Real dkappaJl = f_on ? std::abs(fdif_.thermKappa(fdif_.kappa_TE_para,kl,jl,il)
                                   -fdif_.thermKappa(fdif_.kappa_TE_perp,kl,jl,il))
                                   *HIIl: 0.0;
    Real dkappaJr = f_on ? std::abs(fdif_.thermKappa(fdif_.kappa_TE_para,k,j,i)
                                   -fdif_.thermKappa(fdif_.kappa_TE_perp,k,j,i))
                                   *HIIr : 0.0;
    Real dxl = 0; Real dxr = 0;
    if (ivx == IVX) {
      dxl = pco->dx1f(i-1);
      dxr = pco->dx1f(i);
    } else if (ivx == IVY) {
      dxl = pco->dx2f(j-1);
      dxr = pco->dx2f(j);
    } else {
      dxl = pco->dx3f(k-1);
      dxr = pco->dx3f(k);
    }
    Real dxf = 0.5*(dxl + dxr);

    //--- Step 3b. Gyro-viscous coefficeints
    Real nw1l = h_on ? hdif_.nu(hdif_.nu_wedge1,kl,jl,il)*HIIl : 0.0;
    Real nw1r = h_on ? hdif_.nu(hdif_.nu_wedge1,k,j,i)*HIIr : 0.0;
    Real nw2l = h_on ? hdif_.nu(hdif_.nu_wedge2,kl,jl,il)*HIIl : 0.0;
    Real nw2r = h_on ? hdif_.nu(hdif_.nu_wedge2,k,j,i)*HIIr : 0.0;

    //--- Step 3c. Thermal Hall (right-Leduc) coefficeints
    Real dta = 0.0; Real dtb = 0.0;
    if (ivx == IVX) {
      dta = pco->dx2f(j);
      dtb = pco->dx3f(k);
    } else if (ivx == IVY) {
      dta = pco->dx3f(k);
      dtb = pco->dx1f(i);
    } else {
      dta = pco->dx1f(i);
      dtb = pco->dx2f(j);
    }
    Real curlb = 2.0/dta + 2.0/dtb;
    Real kappawl = h_on ? hdif_.kappa(hdif_.kappa_wedge,kl,jl,il)
                                     *hdif_.ion_frac(kl,jl,il) : 0.0;
    Real kappawr = h_on ? hdif_.kappa(hdif_.kappa_wedge,k,j,i)
                                     *hdif_.ion_frac(k,j,i) : 0.0;

    // Compute fast-magnetosonic speed using eq. B18 (adiabatic) or B39 (isothermal)
    Real btsq = SQR(wroe[IBY]) + SQR(wroe[IBZ]);
    Real vaxsq = bxi*bxi/wroe[IDN];
    Real bt_starsq, twid_asq;
    if (NON_BAROTROPIC_EOS) {
      bt_starsq = (gm1 - (gm1 - 1.0)*y)*btsq;
      Real hp = hroe - (vaxsq + btsq/wroe[IDN]);
      Real vsq = SQR(wroe[IVX]) + SQR(wroe[IVY]) + SQR(wroe[IVZ]);
      twid_asq = std::max((gm1*(hp-0.5*vsq)-(gm1-1.0)*x), 0.0);
    } else {
      bt_starsq = btsq*y;
      twid_asq = iso_cs*iso_cs + x;
    }
    Real ct2 = bt_starsq/wroe[IDN];
    Real tsum = vaxsq + ct2 + twid_asq;
    Real tdif = vaxsq + ct2 - twid_asq;
    Real cf2_cs2 = std::sqrt(tdif*tdif + 4.0*twid_asq*ct2);
    Real cfsq = 0.5*(tsum + cf2_cs2);
    Real a = std::sqrt(cfsq);

    //--- Step 3d. Roe-averaged non-ideal coefficients (face values)
    Real etaHroe = 0.5*(etaHl + etaHr);
    Real dkJroe  = 0.5*(dkappaJl + dkappaJr);
    Real nw1roe  = 0.5*(nw1l + nw1r);
    Real nw2roe  = 0.5*(nw2l + nw2r);
    Real kapwroe = 0.5*(kappawl + kappawr);
    Real eintroe = NON_BAROTROPIC_EOS
                  ? 0.5*(wli[IPR] + wri[IPR])/gm1
                  : 0.5*(wli[IDN] + wri[IDN])*iso_cs*iso_cs/gm1;
    Real Droe = NON_BAROTROPIC_EOS
                ? dkJroe*dkJroe/(eintroe*dxf*dxf)
                : 0.0;
    Real Eroe = etaHroe*etaHroe/(dxf*dxf);
    Real aTw = gm1*curlb*kapwroe;
    Real aw = 0.5*(etaHroe/dxf + std::sqrt(etaHroe*etaHroe/(dxf*dxf) + 4.0*vaxsq));
    Real ae = std::sqrt(a*a + Droe/4.0) + aTw;
    Real af = std::sqrt(std::max(a,aw)*std::max(a,aw) + Droe/4.0
                                                      + std::sqrt(Droe*Eroe)/2.0);

    Real Groe = std::max(nw1roe*nw1roe, 4.0*nw2roe*nw2roe);
    Real denomroe = nw1roe*(nw1roe - 4.0*nw2roe);
    if(std::abs(denomroe) > TINY_NUMBER) {
      Real u_starroe = (nw1roe*nw1roe - 2.0*nw1roe*nw2roe - 2.0*nw2roe*nw2roe)/denomroe;
      if(u_starroe > 0.0 && u_starroe < 1.0) {
        Real Gsroe = -4.0*nw2roe*nw2roe*nw2roe*(2.0*nw1roe + nw2roe)/denomroe;
        Groe = std::max(Groe, Gsroe);
      }
    }

    Real av = a + sqrt(Groe)/dxf;

    // L/R dispersive waves

    // Whistler wave (Field)
    Real cwl = 0.5*(etaHl/dxl + std::sqrt(etaHl*etaHl/(dxl*dxl) + 4.0*cal*cal));
    Real cwr = 0.5*(etaHr/dxr + std::sqrt(etaHr*etaHr/(dxr*dxr) + 4.0*car*car));

    // Peltier Seebeck wave (Internal energy)
    Real Dl = 0;
    Real Dr = 0;
    if(NON_BAROTROPIC_EOS) {
    Dl = dkappaJl*dkappaJl/((wli[IPR]/gm1)*dxl*dxl);
    Dr = dkappaJr*dkappaJr/((wri[IPR]/gm1)*dxr*dxr);
    }
    Real ctel = std::sqrt(cl*cl + Dl/4.0);
    Real cter = std::sqrt(cr*cr + Dr/4.0);

    // Hall + Peltier/Seebeck wave (field)
    Real El = etaHl*etaHl/(dxl*dxl);
    Real Er = etaHr*etaHr/(dxr*dxr);
    Real ctewl = std::sqrt(std::max(cl,cwl)*std::max(cl,cwl)
                         + Dl/4.0 + std::sqrt(Dl*El)/2.0);
    Real ctewr = std::sqrt(std::max(cr,cwr)*std::max(cr,cwr)
                        + Dr/4.0 + std::sqrt(Dr*Er)/2.0);

    // Gyro-viscous Wave (Velocity)
    Real Gl = std::max(nw1l*nw1l, 4.0*nw2l*nw2l);
    Real denoml = nw1l*(nw1l - 4.0*nw2l);
    if(std::abs(denoml) > TINY_NUMBER) {
      Real u_starl = (nw1l*nw1l - 2.0*nw1l*nw2l - 2.0*nw2l*nw2l)/denoml;
      if(u_starl > 0.0 && u_starl < 1.0) {
        Real Gsl = -4.0*nw2l*nw2l*nw2l*(2.0*nw1l + nw2l)/denoml;
        Gl = std::max(Gl, Gsl);
      }
    }

    Real Gr = std::max(nw1r*nw1r, 4.0*nw2r*nw2r);
    Real denomr = nw1r*(nw1r - 4.0*nw2r);
    if(std::abs(denomr) > TINY_NUMBER) {
      Real u_starr = (nw1r*nw1r - 2.0*nw1r*nw2r - 2.0*nw2r*nw2r)/denomr;
      if(u_starr > 0.0 && u_starr < 1.0) {
        Real Gsr = -4.0*nw2r*nw2r*nw2r*(2.0*nw1r + nw2r)/denomr;
        Gr = std::max(Gr, Gsr);
      }
    }

    Real cgvl = cl + std::sqrt(Gl)/dxl;
    Real cgvr = cr + std::sqrt(Gr)/dxr;

    // Thermal hall (right_Leduc) velocity (Internal Energy)
    Real cTwl = gm1*curlb*kappawl;
    Real cTwr = gm1*curlb*kappawr;

    //--- Step 4. Compute the max/min wave speeds based on L/R and Roe-averaged values

    // density speeds
    Real al = std::min((wroe[IVX] - a),(wli[IVX] - cl));
    Real ar = std::max((wroe[IVX] + a),(wri[IVX] + cr));
    Real bp = ar > 0.0 ? ar : 0.0;
    Real bm = al < 0.0 ? al : 0.0;

    // velocity speed
    Real avl = std::min((wroe[IVX] - av),(wli[IVX] - cgvl));
    Real avr = std::max((wroe[IVX] + av),(wri[IVX] + cgvr));
    Real bvp = avr > 0.0 ? avr : 0.0;
    Real bvm = avl < 0.0 ? avl : 0.0;

    // energy speeds
    Real ael = std::min((wroe[IVX] - ae),(wli[IVX] - ctel - cTwl));
    Real aer = std::max((wroe[IVX] + ae),(wri[IVX] + cter + cTwr));
    Real bep = aer > 0.0 ? aer : 0.0;
    Real bem = ael < 0.0 ? ael : 0.0;

    // field speeds
    Real afl = std::min((wroe[IVX] - af),(wli[IVX] - ctewl));
    Real afr = std::max((wroe[IVX] + af),(wri[IVX] + ctewr));
    Real bfp = afr > 0.0 ? afr : 0.0;
    Real bfm = afl < 0.0 ? afl : 0.0;

    //--- Step 5. Compute L/R fluxes along the lines bm/bp: F_L - (S_L)U_L; F_R - (S_R)U_R

    Real vxl = wli[IVX] - bm;
    Real vxr = wri[IVX] - bp;
    Real vxvl = wli[IVX] - bvm;
    Real vxvr = wri[IVX] - bvp;
    Real vxel = wli[IVX] - bem;
    Real vxer = wri[IVX] - bep;
    Real vxfl = wli[IVX] - bfm;
    Real vxfr = wri[IVX] - bfp;

    // --- Split energy into interal+magnetic+kinetic sub-fluxes for each signal speed
    Real ekl = 0.0; Real ekr = 0.0;
    Real fl_int = 0.0; Real fr_int = 0.0; Real fl_kin = 0.0; Real fr_kin = 0.0;
    Real fl_mag = 0.0; Real fr_mag = 0.0;

    if(NON_BAROTROPIC_EOS) {
      ekl = 0.5*wli[IDN]*(SQR(wli[IVX])+SQR(wli[IVY])+SQR(wli[IVZ]));
      ekr = 0.5*wri[IDN]*(SQR(wri[IVX])+SQR(wri[IVY])+SQR(wri[IVZ]));

      fl_int = (wli[IPR]/gm1)*vxel + wli[IPR]*wli[IVX];
      fr_int = (wri[IPR]/gm1)*vxer + wri[IPR]*wri[IVX];

      fl_kin = ekl*vxvl;
      fr_kin = ekr*vxvr;

      fl_mag = pbl*vxfl + (pbl - bxi*bxi)*wli[IVX]
                  - bxi*(wli[IBY]*wli[IVY] + wli[IBZ]*wli[IVZ]);
      fr_mag = pbr*vxfr + (pbr - bxi*bxi)*wri[IVX]
                  - bxi*(wri[IBY]*wri[IVY] + wri[IBZ]*wri[IVZ]);
    }

    fl[IDN] = wli[IDN]*vxl;
    fr[IDN] = wri[IDN]*vxr;

    fl[IVX] = wli[IDN]*wli[IVX]*vxvl + pbl - SQR(bxi);
    fr[IVX] = wri[IDN]*wri[IVX]*vxvr + pbr - SQR(bxi);

    fl[IVY] = wli[IDN]*wli[IVY]*vxvl - bxi*wli[IBY];
    fr[IVY] = wri[IDN]*wri[IVY]*vxvr - bxi*wri[IBY];

    fl[IVZ] = wli[IDN]*wli[IVZ]*vxvl - bxi*wli[IBZ];
    fr[IVZ] = wri[IDN]*wri[IVZ]*vxvr - bxi*wri[IBZ];

    if (NON_BAROTROPIC_EOS) {
      fl[IVX] += wli[IPR];
      fr[IVX] += wri[IPR];
    } else {
      fl[IVX] += (iso_cs*iso_cs)*wli[IDN];
      fr[IVX] += (iso_cs*iso_cs)*wri[IDN];
    }

    fl[IBY] = wli[IBY]*vxfl - bxi*wli[IVY];
    fr[IBY] = wri[IBY]*vxfr - bxi*wri[IVY];

    fl[IBZ] = wli[IBZ]*vxfl - bxi*wli[IVZ];
    fr[IBZ] = wri[IBZ]*vxfr - bxi*wri[IVZ];

    //--- Step 6. Compute the HLLE flux at interface.

    Real tmp=0.0;
    Real tmpv=0.0;
    Real tmpe=0.0;
    Real tmpf=0.0;
    if (bp != bm) tmp = 0.5*(bp + bm)/(bp - bm);
    if (bvp != bvm) tmpv = 0.5*(bvp + bvm)/(bvp - bvm);
    if (bep != bem) tmpe = 0.5*(bep + bem)/(bep - bem);
    if (bfp != bfm) tmpf = 0.5*(bfp + bfm)/(bfp - bfm);

    flxi[IDN] = 0.5*(fl[IDN]+fr[IDN]) + (fl[IDN]-fr[IDN])*tmp;
    flxi[IVX] = 0.5*(fl[IVX]+fr[IVX]) + (fl[IVX]-fr[IVX])*tmpv;
    flxi[IVY] = 0.5*(fl[IVY]+fr[IVY]) + (fl[IVY]-fr[IVY])*tmpv;
    flxi[IVZ] = 0.5*(fl[IVZ]+fr[IVZ]) + (fl[IVZ]-fr[IVZ])*tmpv;
    if (NON_BAROTROPIC_EOS) flxi[IEN] = 0.5*(fl_int+fr_int) + (fl_int-fr_int)*tmpe
                                      + 0.5*(fl_kin+fr_kin) + (fl_kin-fr_kin)*tmpv
                                      + 0.5*(fl_mag+fr_mag) + (fl_mag-fr_mag)*tmpf;
    flxi[IBY] = 0.5*(fl[IBY]+fr[IBY]) + (fl[IBY]-fr[IBY])*tmpf;
    flxi[IBZ] = 0.5*(fl[IBZ]+fr[IBZ]) + (fl[IBZ]-fr[IBZ])*tmpf;

    flx(IDN,k,j,i) = flxi[IDN];
    flx(ivx,k,j,i) = flxi[IVX];
    flx(ivy,k,j,i) = flxi[IVY];
    flx(ivz,k,j,i) = flxi[IVZ];
    if (NON_BAROTROPIC_EOS) flx(IEN,k,j,i) = flxi[IEN];
    ey(k,j,i) = -flxi[IBY];
    ez(k,j,i) =  flxi[IBZ];

    wct(k,j,i)=GetWeightForCT(flxi[IDN], wli[IDN], wri[IDN], dxw(i), dt);
  }
  return;
}
