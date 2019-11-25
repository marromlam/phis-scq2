////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//                       CUDA decay rate Bs -> mumuKK                         //
//                                                                            //
//  Created: 2019-01-25                                                       //
//                                                                            //
//                                                                            //
//                                                                            //
//                                                                            //
//                                                                            //
//                                                                            //
//                                                                            //
//                                                                            //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
// Inlude headers //////////////////////////////////////////////////////////////

#include <stdio.h>
#include <math.h>

extern "C"

////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
{


	__device__ double getNcoeffs(double APlon,
	   double ASlon,
	   double APpar,
	   double APper,
	   double CSP,
	   int k)
	{
	  double nk;
	  switch(k) {
	  case 1:  nk = APlon*APlon;
	   break;
	  case 2:  nk = APpar*APpar;
	   break;
	  case 3:  nk = APper*APper;
	   break;
	  case 4:  nk = APper*APpar;
	   break;
	  case 5:  nk = APlon*APpar;
	   break;
	  case 6:  nk = APlon*APper;
	   break;
	  case 7:  nk = ASlon*ASlon;
	   break;
	  case 8:  nk = CSP*ASlon*APpar;
	   break;
	  case 9:  nk = CSP*ASlon*APper;
	   break;
	  case 10: nk = CSP*ASlon*APlon;
	   break;
	  default: printf("Wrong k index in nk, please check code %d\\n", k);
	   return 0.;
	  }
	  return nk;
	}

	__device__ double getFcoeffs(double cosK,
	   double cosL,
	   double hphi,
	   int k)
	{
	  double helsinthetaK = sqrt(1. - cosK*cosK);
	  double helsinthetaL = sqrt(1. - cosL*cosL);
	//     hphi -= M_PI;
	//     double helsinphi = sin(-hphi);
	//     double helcosphi = cos(-hphi);
	  double helsinphi = sin(hphi);
	  double helcosphi = cos(hphi);

	  double fk;
	  switch(k) {
	  case 1:  fk = cosK*cosK*helsinthetaL*helsinthetaL;
	   break;
	  case 2:  fk = 0.5*helsinthetaK*helsinthetaK*(1.-helcosphi*helcosphi*helsinthetaL*helsinthetaL);
	   break;
	  case 3:  fk = 0.5*helsinthetaK*helsinthetaK*(1.-helsinphi*helsinphi*helsinthetaL*helsinthetaL);
	   break;
	  case 4:  fk = helsinthetaK*helsinthetaK*helsinthetaL*helsinthetaL*helsinphi*helcosphi;
	   break;
	  case 5:  fk = sqrt(2.)*helsinthetaK*cosK*helsinthetaL*cosL*helcosphi;
	   break;
	  case 6:  fk = -sqrt(2.)*helsinthetaK*cosK*helsinthetaL*cosL*helsinphi;
	   break;
	  case 7:  fk = helsinthetaL*helsinthetaL/3.;
	   break;
	  case 8:  fk = 2.*helsinthetaK*helsinthetaL*cosL*helcosphi/sqrt(6.);
	   break;
	  case 9:  fk = -2.*helsinthetaK*helsinthetaL*cosL*helsinphi/sqrt(6.);
	   break;
	  case 10: fk = 2.*cosK*helsinthetaL*helsinthetaL/sqrt(3.);
	   break;
	  default: printf("Wrong k index in fk, please check code %d\\n", k);
	   return 0.;
	  }
	  return fk;
	}

	__device__ double getAcoeffs(double phisPlon,
	   double phisSlon,
	   double phisPpar,
	   double phisPper,
	   double deltaPlon,
	   double deltaSlon,
	   double deltaPpar,
	   double deltaPper,
	   double lambdaPlon,
	   double lambdaSlon,
	   double lambdaPpar,
	   double lambdaPper,
	   int k)
	{
	  double ak;
	  switch(k) {
	  case 1:  ak = 0.5*(1.+lambdaPlon*lambdaPlon);
	   break;
	  case 2:  ak = 0.5*(1.+lambdaPpar*lambdaPpar);
	   break;
	  case 3:  ak = 0.5*(1.+lambdaPper*lambdaPper);
	   break;
	  case 4:  ak = 0.5*(sin(deltaPper-deltaPpar) - lambdaPper*lambdaPpar*sin(deltaPper-deltaPpar-phisPper+phisPpar));
	   break;
	  case 5:  ak = 0.5*(cos(deltaPlon-deltaPpar) + lambdaPlon*lambdaPpar*cos(deltaPlon-deltaPpar-phisPlon+phisPpar));
	   break;
	  case 6:  ak = -0.5*(sin(deltaPlon-deltaPper) - lambdaPlon*lambdaPper*sin(deltaPlon-deltaPper-phisPlon+phisPper));
	   break;
	  case 7:  ak = 0.5*(1.+lambdaSlon*lambdaSlon);
	   break;
	  case 8:  ak = 0.5*(cos(deltaSlon-deltaPpar) - lambdaSlon*lambdaPpar*cos(deltaSlon-deltaPpar-phisSlon+phisPpar));
	   break;
	  case 9:  ak = -0.5*(sin(deltaSlon-deltaPper) + lambdaSlon*lambdaPper*sin(deltaSlon-deltaPper-phisSlon+phisPper));
	   break;
	  case 10: ak = 0.5*(cos(deltaSlon-deltaPlon) - lambdaSlon*lambdaPlon*cos(deltaSlon-deltaPlon-phisSlon+phisPlon));
	   break;
	  default: printf("Wrong k index in ak, please check code %d\\n", k);
	   return 0.;
	  }
	  return ak;
	}

	__device__ double getBcoeffs(double phisPlon,
	   double phisSlon,
	   double phisPpar,
	   double phisPper,
	   double deltaPlon,
	   double deltaSlon,
	   double deltaPpar,
	   double deltaPper,
	   double lambdaPlon,
	   double lambdaSlon,
	   double lambdaPpar,
	   double lambdaPper,
	   int k)
	{
	  double bk;
	  switch(k) {
	  case 1:  bk = -lambdaPlon*cos(phisPlon);
	   break;
	  case 2:  bk = -lambdaPpar*cos(phisPpar);
	   break;
	  case 3:  bk = lambdaPper*cos(phisPper);
	   break;
	  case 4:  bk = 0.5*(lambdaPper*sin(deltaPper-deltaPpar-phisPper) + lambdaPpar*sin(deltaPpar-deltaPper-phisPpar));
	   break;
	  case 5:  bk = -0.5*(lambdaPlon*cos(deltaPlon-deltaPpar-phisPlon) + lambdaPpar*cos(deltaPpar-deltaPlon-phisPpar));
	   break;
	  case 6:  bk = 0.5*(lambdaPlon*sin(deltaPlon-deltaPper-phisPlon) + lambdaPper*sin(deltaPper-deltaPlon-phisPper));
	   break;
	  case 7:  bk = lambdaSlon*cos(phisSlon);
	   break;
	  case 8:  bk = 0.5*(lambdaSlon*cos(deltaSlon-deltaPpar-phisSlon) - lambdaPpar*cos(deltaPpar-deltaSlon-phisPpar));
	   break;
	  case 9:  bk = -0.5*(lambdaSlon*sin(deltaSlon-deltaPper-phisSlon) - lambdaPper*sin(deltaPper-deltaSlon-phisPper));
	   break;
	  case 10: bk = 0.5*(lambdaSlon*cos(deltaSlon-deltaPlon-phisSlon) - lambdaPlon*cos(deltaPlon-deltaSlon-phisPlon));
	   break;
	  default: printf("Wrong k index in bk, please check code %d\\n", k);
	   return 0.;
	  }
	  return bk;
	}

	__device__ double getCcoeffs(double phisPlon,
	   double phisSlon,
	   double phisPpar,
	   double phisPper,
	   double deltaPlon,
	   double deltaSlon,
	   double deltaPpar,
	   double deltaPper,
	   double lambdaPlon,
	   double lambdaSlon,
	   double lambdaPpar,
	   double lambdaPper,
	   int k)
	{

	  double ck;
	  switch(k) {
	  case 1:  ck = 0.5*(1.-lambdaPlon*lambdaPlon);
	   break;
	  case 2:  ck = 0.5*(1.-lambdaPpar*lambdaPpar);
	   break;
	  case 3:  ck = 0.5*(1.-lambdaPper*lambdaPper);
	   break;
	  case 4:  ck = 0.5*(sin(deltaPper-deltaPpar) + lambdaPper*lambdaPpar*sin(deltaPper-deltaPpar-phisPper+phisPpar));
	   break;
	  case 5:  ck = 0.5*(cos(deltaPlon-deltaPpar) - lambdaPlon*lambdaPpar*cos(deltaPlon-deltaPpar-phisPlon+phisPpar));
	   break;
	  case 6:  ck = -0.5*(sin(deltaPlon-deltaPper) + lambdaPlon*lambdaPper*sin(deltaPlon-deltaPper-phisPlon+phisPper));
	   break;
	  case 7:  ck = 0.5*(1.-lambdaSlon*lambdaSlon);
	   break;
	  case 8:  ck = 0.5*(cos(deltaSlon-deltaPpar) + lambdaSlon*lambdaPpar*cos(deltaSlon-deltaPpar-phisSlon+phisPpar));
	   break;
	  case 9:  ck = -0.5*(sin(deltaSlon-deltaPper) - lambdaSlon*lambdaPper*sin(deltaSlon-deltaPper-phisSlon+phisPper));
	   break;
	  case 10: ck = 0.5*(cos(deltaSlon-deltaPlon) + lambdaSlon*lambdaPlon*cos(deltaSlon-deltaPlon-phisSlon+phisPlon));
	   break;
	  default: printf("Wrong k index in ck, please check code %d\\n", k);
	   return 0.;
	  }
	  return ck;
	}

	__device__ double getDcoeffs(double phisPlon,
	   double phisSlon,
	   double phisPpar,
	   double phisPper,
	   double deltaPlon,
	   double deltaSlon,
	   double deltaPpar,
	   double deltaPper,
	   double lambdaPlon,
	   double lambdaSlon,
	   double lambdaPpar,
	   double lambdaPper,
	   int k)
	{

	  double dk;
	  switch(k) {
	  case 1:  dk = lambdaPlon*sin(phisPlon);
	   break;
	  case 2:  dk = lambdaPpar*sin(phisPpar);
	   break;
	  case 3:  dk = -lambdaPper*sin(phisPper);
	   break;
	  case 4:  dk = -0.5*(lambdaPper*cos(deltaPper-deltaPpar-phisPper) + lambdaPpar*cos(deltaPpar-deltaPper-phisPpar));
	   break;
	  case 5:  dk = -0.5*(lambdaPlon*sin(deltaPlon-deltaPpar-phisPlon) + lambdaPpar*sin(deltaPpar-deltaPlon-phisPpar));
	   break;
	  case 6:  dk = -0.5*(lambdaPlon*cos(deltaPlon-deltaPper-phisPlon) + lambdaPper*cos(deltaPper-deltaPlon-phisPper));
	   break;
	  case 7:  dk = -lambdaSlon*sin(phisSlon);
	   break;
	  case 8:  dk = 0.5*(lambdaSlon*sin(deltaSlon-deltaPpar-phisSlon) - lambdaPpar*sin(deltaPpar-deltaSlon-phisPpar));
	   break;
	  case 9:  dk = -0.5*(-lambdaSlon*cos(deltaSlon-deltaPper-phisSlon) + lambdaPper*cos(deltaPper-deltaSlon-phisPper));
	   break;
	  case 10: dk = 0.5*(lambdaSlon*sin(deltaSlon-deltaPlon-phisSlon) - lambdaPlon*sin(deltaPlon-deltaSlon-phisPlon));
	   break;
	  default: printf("Wrong k index in dk, please check code %d\\n", k);

	   return 0.;
	  }
	  return dk;
	}




















}
