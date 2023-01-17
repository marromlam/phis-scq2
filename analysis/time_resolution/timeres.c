#define USE_DOUBLE 1
#include <exposed/kernels.ocl>

WITHIN_KERNEL
ftype wrong_pv_component(const ftype x, const ftype tau1, const ftype tau2,
                         const ftype share, const ftype xmin,
                         const ftype xmax) {
  ftype num = 0.0;
  ftype den = 1.0;

  ftype exp1 = exp(-fabs(x) / tau1) /
               (2 - exp(-(xmax / tau1)) - exp(xmin / tau1)) / tau1;
  ftype exp2 = exp(-fabs(x) / tau2) /
               (2 - exp(-(xmax / tau2)) - exp(xmin / tau2)) / tau2;
  num = exp1 + (1 - share) * exp2 / share;
  den = 1 + (1 - share) / share;

  return num / den;
}

KERNEL
void kernel_wrong_pv_component(GLOBAL_MEM ftype *prob,
                               GLOBAL_MEM const ftype *time, const ftype tau1,
                               const ftype tau2, const ftype share,
                               const ftype tLL, const ftype tUL) {
  const int idx = get_global_id(0);
  prob[idx] = wrong_pv_component(time[idx], tau1, tau2, share, tLL, tUL);
}

WITHIN_KERNEL
ftype mygauss(const ftype x, const ftype mu, const ftype sigma, const ftype xLL,
              const ftype xUL) {
  const ftype num = exp(-0.5 * pow(mu - x, 2) / rpow(sigma, 2));
  ftype den = 0;
  den += erf((mu - xLL) / (sqrt(2.) * sigma));
  den -= erf((mu - xUL) / (sqrt(2.) * sigma));
  den *= sqrt(M_PI / 2.) * sigma;
  return num / den;
}

#define Pi M_PI
WITHIN_KERNEL
ftype myexp(const ftype x, const ftype tau, const ftype mu, const ftype sigma,
            const ftype xLL, const ftype xUL) {

  const ftype num =
      exp((rpow(sigma, 2) + 2 * mu * tau - 2 * tau * x) / (2. * rpow(tau, 2))) *
      sqrt(M_PI / 2.) * sigma *
      erfc((rpow(sigma, 2) + tau * (mu - x)) / (sqrt(2.) * sigma * tau));
  const ftype den =
      sqrt(Pi / 2.) * sigma * tau *
      (-1 + erf((-mu + xUL) / (sqrt(2.) * sigma)) +
       erfc((-mu + xLL) / (sqrt(2.) * sigma)) +
       exp((pow(sigma, 2) + 2 * tau * (mu - xLL - xUL)) / (2. * pow(tau, 2))) *
           (exp(xUL / tau) * erfc((pow(sigma, 2) + mu * tau - tau * xLL) /
                                  (sqrt(2.) * sigma * tau)) -
            exp(xLL / tau) * erfc((pow(sigma, 2) + mu * tau - tau * xUL) /
                                  (sqrt(2.) * sigma * tau))));
  return num / den;
}

WITHIN_KERNEL
ftype time_fit(const ftype time, GLOBAL_MEM const ftype *fres, const ftype mu,
               GLOBAL_MEM const ftype *sigma, const ftype fprompt,
               const ftype fll, const ftype fsl, const ftype fwpv,
               const ftype taul, const ftype taus, const ftype tau1,
               const ftype tau2, const ftype share_wpv, const ftype tLL,
               const ftype tUL) {
  // if (get_global_id(0)==5000)
  // {
  //    printf("fprompt=%f, fll=%f, fsl=%f, fwpw=%f, fres[0]=%f, fres[1]=%f\\n",
  //    fprompt, fll, fsl, fwpv, fres[0], fres[1]); printf("mu=%f, s1=%f, s2=%f,
  //    fres1=%f, fres2=%f\\n", mu, sigma[0], sigma[1], fres[0], fres[1]);
  //    printf("tLL=%f, tUL=%f\\n", tLL, tUL);
  //    printf("time = %f -> %f / %f\\n", time, exp(rpow(-mu +
  //    time,2.)/(2*rpow(sigma[0],2.))) ,0    ); printf("prompt = %f\\n",
  //    prompt); printf("long_live = %f\\n", long_live);
  // }

  ftype prompt = 0;
  ftype long_live = 0;

  for (int i = 0; i < 2; ++i) {
    prompt += fres[i] * mygauss(time, mu, sigma[i], tLL, tUL);
    long_live += fsl * fres[i] * myexp(time, taus, mu, sigma[i], tLL, tUL);
    long_live +=
        (1 - fsl) * fres[i] * myexp(time, taul, mu, sigma[i], tLL, tUL);
  }

  const ftype wpv = wrong_pv_component(time, tau1, tau2, share_wpv, tLL, tUL);

  const ftype num = fprompt * prompt + fll * long_live + fwpv * wpv;
  const ftype den = fprompt + fll + fwpv; //(fprompt + fwpv)/fprompt;
  // if (get_global_id(0)==5000)
  // {
  //    printf("den from CUDA = %f\\n", den);
  // }
  return num / den;
}

KERNEL
void kernel_time_fit(GLOBAL_MEM ftype *prob, GLOBAL_MEM const ftype *time,
                     GLOBAL_MEM const ftype *fres, const ftype mu,
                     GLOBAL_MEM const ftype *sigma, const ftype fprompt,
                     const ftype fll, const ftype fsl, const ftype fwpv,
                     const ftype taul, const ftype taus, const ftype tau1,
                     const ftype tau2, const ftype share, const ftype tLL,
                     const ftype tUL) {
  const int idx = get_global_id(0);
  prob[idx] = time_fit(time[idx], fres, mu, sigma, fprompt, fll, fsl, fwpv,
                       taul, taus, tau1, tau2, share, tLL, tUL);
}

WITHIN_KERNEL
ftype time_fit_bis(const ftype time, GLOBAL_MEM const ftype *fres,
                   const ftype mu, GLOBAL_MEM const ftype *sigma,
                   const ftype fprompt, const ftype fll, const ftype fsl,
                   const ftype fwpv, const ftype taul, const ftype taus,
                   const ftype tau1, const ftype tau2, const ftype share_wpv,
                   const ftype tLL, const ftype tUL) {
  if (get_global_id(0) == 5000) {
    //    printf("fprompt=%f, fll=%f, fsl=%f, fwpw=%f, fres[0]=%f,
    //    fres[1]=%f\\n", fprompt, fll, fsl, fwpv, fres[0], fres[1]);
    printf("mu=%f, s1=%f, s2=%f, s3=%f, fres1=%f, fres2=%f, fres3=%f\n", mu,
           sigma[0], sigma[1], sigma[2], fres[0], fres[1], fres[2]);
    //    printf("tLL=%f, tUL=%f\\n", tLL, tUL);
    //    printf("time = %f -> %f / %f\\n", time, exp(rpow(-mu +
    //    time,2.)/(2*rpow(sigma[0],2.))) ,0    ); printf("prompt = %f\\n",
    //    prompt); printf("long_live = %f\\n", long_live);
  }

  ftype prompt = 0;
  ftype llive = 0;

  for (int i = 0; i < 3; ++i) {
    prompt += fres[i] * mygauss(time, mu, sigma[i], tLL, tUL);
    llive += fsl * fres[i] * myexp(time, taus, mu, sigma[i], tLL, tUL);
    llive += (1 - fsl) * fres[i] * myexp(time, taul, mu, sigma[i], tLL, tUL);
  }

  const ftype wpv = wrong_pv_component(time, tau1, tau2, share_wpv, tLL, tUL);

  const ftype num = fprompt * prompt + fll * llive + fwpv * wpv;
  const ftype den = fprompt + fll + fwpv; //(fprompt + fwpv)/fprompt;
  // if (get_global_id(0)==5000)
  // {
  //    printf("den from CUDA = %f\\n", den);
  // }
  return num / den;
}

KERNEL
void kernel_time_fit_bis(GLOBAL_MEM ftype *prob, GLOBAL_MEM const ftype *time,
                         GLOBAL_MEM const ftype *fres, const ftype mu,
                         GLOBAL_MEM const ftype *sigma, const ftype fprompt,
                         const ftype fll, const ftype fsl, const ftype fwpv,
                         const ftype taul, const ftype taus, const ftype tau1,
                         const ftype tau2, const ftype share, const ftype tLL,
                         const ftype tUL) {
  const int idx = get_global_id(0);
  prob[idx] = time_fit_bis(time[idx], fres, mu, sigma, fprompt, fll, fsl, fwpv,
                           taul, taus, tau1, tau2, share, tLL, tUL);
}

// vim: fdm=marker
