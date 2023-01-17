#define USE_DOUBLE 1

#include <lib99ocl/core.c>

// Debugging 0 [0,1,2,3,>3]
#define DEBUG 0
#define DEBUG_EVT 0 // the event that is being debugged

#define FAST_INTEGRAL 1 // fast integral, means avoinding faddeva

// Time resolution parameters
#define SIGMA_T 0.15
#define SIGMA_THRESHOLD 5.0

// Time acceptance parameters
#define NKNOTS 4
#define SPL_BINS 4
#define NTIMEBINS 5
const CONSTANT_MEM ftype KNOTS[5] = {0.3, 0.91, 1.96, 9.0};

// PDF parameters
#define MKNOTS 2
const CONSTANT_MEM ftype MHH[MKNOTS] = {990, 1050};

#include <lib99ocl/complex.c>
// #include <lib99ocl/cspecial.c>
// #include <lib99ocl/random.c>
#include <lib99ocl/special.c>

#include "decay_time_acceptance.c"
#include "decay_time_acceptance.h"
#include "tagging.c"
#include "tagging.h"

#define USE_TIMEACC 0
#define USE_TIMERES 0

KERNEL
void calibrated_mistag(ftype *data, ftype *out, const ftype p0, const ftype dp0,
                       const ftype p1, const ftype dp1, const ftype p2,
                       const ftype dp2, const ftype eta_bar) {
  int row = get_global_id(0);
  if (row < 1) {
    printf("%+.4f  %+.4f %+.4f %+.4f %+.4f %+.4f %+.4f\n", p0, dp0, p1, dp1, p2,
           dp2, eta_bar);
  }
  const ftype eta = data[1 + row * 3];
  const ftype q_true = data[2 + row * 3] / fabs(data[2 + row * 3]);
  const ftype om = get_omega(eta, q_true, p0, p1, p2, dp0, dp1, dp2, eta_bar);
  out[row] = om;
}

KERNEL
void tagging_calibration_ost(GLOBAL_MEM const ftype *data,
                             GLOBAL_MEM ftype *out, const ftype p0,
                             const ftype dp0, const ftype p1, const ftype dp1,
                             const ftype p2, const ftype dp2,
                             const ftype eta_bar, const int Nevt) {
  // Loading the data
  int row = get_global_id(0);

  if (row >= Nevt) {
    return;
  }

  const ftype q = data[0 + row * 3];
  const ftype eta = data[1 + row * 3];
  const ftype id = data[2 + row * 3];

  // if (row < 10) {
  //   printf("row = %+.4f    %+.4f    %+.4f\n", q, eta, BID);
  // }
  // if (row < 1) {
  //   printf("%+.4f  %+.4f %+.4f %+.4f %+.4f %+.4f %+.4f\n", p0, dp0, p1, dp1,
  //   p2,
  //          dp2, eta_bar);
  // }

  ftype well_predicted = 0.;
  const ftype q_true = id / fabs(id);
  if (q_true == q) {
    well_predicted = 1.;
  }

  const ftype om = get_omega(eta, q_true, p0, p1, p2, dp0, dp1, dp2, eta_bar);
  const ftype ans = (1 - well_predicted) * om + well_predicted * (1 - om);
  out[row] = ans;
}

KERNEL
void tagging_calibration_sst(GLOBAL_MEM const ftype *data,
                             GLOBAL_MEM ftype *out, GLOBAL_MEM ftype *coeffs,
                             const ftype G, const ftype DG, const ftype DM,
                             const ftype p0, const ftype dp0, const ftype p1,
                             const ftype dp1, const ftype p2, const ftype dp2,
                             const ftype eta_bar, const ftype tLL,
                             const ftype tUL, const int Nevt) {
  // Loading the data
  int row = get_global_id(0);

  if (row >= Nevt) {
    return;
  }

  // get data {{{

  const ftype q = data[0 + row * 3];
  const ftype eta = data[1 + row * 3];
  const ftype id = data[2 + row * 3];
  const ftype time = data[3 + row * 4];

  if (row < 1) {
    printf("row = %+.4f    %+.4f    %+.4f    %+.4f\n", q, eta, id, time);
  }
  if (row < 1) {
    printf("%+.4f  %+.4f %+.4f %+.4f %+.4f %+.4f %+.4f\n", p0, dp0, p1, dp1, p2,
           dp2, eta_bar);
  }

  // }}}

  // Time-dependent part {{{

  const ftype t_offset = 0;
  const ftype delta_t = 0.04;

  // to store exponential covolutions
  ctype exp_p = C(0, 0);
  ctype exp_m = C(0, 0);
  ctype exp_i = C(0, 0);

  exp_p = expconv_simon(time - t_offset, G + 0.5 * DG, 0., delta_t);
  exp_m = expconv_simon(time - t_offset, G - 0.5 * DG, 0., delta_t);
  exp_i = expconv_simon(time - t_offset, G, DM, delta_t);

  ftype ta = 0.5 * (exp_m.x + exp_p.x);
  ftype tc = exp_i.x;

  // to store integral
  ftype integrals[4] = {0., 0., 0., 0.};
  intgTimeAcceptance(integrals, delta_t, G, DG, DM, coeffs, t_offset, tLL, tUL);

  const ftype int_ta = integrals[0];
  const ftype int_tb = integrals[1];
  const ftype int_tc = integrals[2];
  const ftype int_td = integrals[3];

#if DEBUG
  if (DEBUG >= 3 && get_global_id(0) == DEBUG_EVT) {
    printf("\nTIME TERMS         : ta=%.8f  tc=%.8f\n", ta, tc);
    printf(
        "                   : exp_m=%.8f  exp_p=%.8f  exp_i=%.8f  exp_i=%.8f\n",
        sqrt(2 * M_PI) * exp_m.x, sqrt(2 * M_PI) * exp_p.x,
        sqrt(2 * M_PI) * exp_i.x, exp_i.y);
  }
#endif

  // }}}

  // Decay-time acceptance {{{
  //     To get rid of decay-time acceptance set USE_TIMEACC to False. If True
  //     then time_efficiency locates the time bin of the event and returns
  //     the value of the cubic spline.

  ftype dta = 1.0;
  if (USE_TIMEACC) {
    dta = time_efficiency(time, coeffs, tLL, tUL);
  }

  // }}}

  // Does it have oscillate {{{

  ftype q_mix = -1.;
  const ftype q_true = id / fabs(id);
  if (q_true == q) {
    q_mix = 1.;
  }

  // }}}

  const ftype om = get_omega(eta, q_true, p0, p1, p2, dp0, dp1, dp2, eta_bar);
  const ftype num = dta * (ta + q_mix * (1 - 2 * om) * tc);
  const ftype den = (int_ta + q_mix * (1 - 2 * om) * int_tc);
  out[row] = num / den;
}

// vim: fdm=marker
