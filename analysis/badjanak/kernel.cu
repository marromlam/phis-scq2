////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//                      KERNEL FOR THE PHIS-SCQ ANALYSIS                      //
//                                                                            //
//   Created: 2019-11-18                                                      //
//    Author: Marcos Romero Lamas (mromerol@cern.ch)                          //
//                                                                            //
//    This file is part of phis-scq packages, Santiago's framework for the    //
//                     phi_s analysis in Bs -> Jpsi K+ K-                     //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
// Include headers /////////////////////////////////////////////////////////////

#define USE_DOUBLE ${USE_DOUBLE}
#include <lib99ocl/core.c>

// Debugging 0 [0,1,2,3,>3]
#define DEBUG ${DEBUG}
#define DEBUG_EVT ${DEBUG_EVT}               // the event that is being debugged

#define FAST_INTEGRAL ${FAST_INTEGRAL} // fast integral, means avoinding faddeva

// Time resolution parameters
#define SIGMA_T ${SIGMA_T}
#define SIGMA_THRESHOLD 5.0

// Time acceptance parameters
#define NKNOTS ${NKNOTS}
#define SPL_BINS ${SPL_BINS}
#define NTIMEBINS ${NTIMEBINS}
const CONSTANT_MEM ftype KNOTS[NKNOTS] = ${KNOTS};

// PDF parameters
#define NTERMS ${NTERMS}
#define MKNOTS ${NMASSKNOTS}
const CONSTANT_MEM ftype MHH[MKNOTS] = ${MHH};
const CONSTANT_MEM ftype TRISTAN[NTERMS] = ${TRISTAN};

// Include ipanema
#include <lib99ocl/complex.c>
#include <lib99ocl/special.c>
#include <lib99ocl/cspecial.c>
#include <lib99ocl/random.c>

// Include analysis parts implementation
#include "fk_helpers.c"
#include "tagging.c"
#include "time_angular_distribution.c"
#include "decay_time_acceptance.c"
#include "cross_rate_bs.c"
#include "cross_rate_bd.c"
#include "angular_acceptance.c"

// Include exposed kernels
#ifdef CUDA //fk_helpers.cu is not openCL compilant (why?)
#include "fk_helpers.cu"
#endif
#include "time_angular_distribution.cu"
#include "decay_time_acceptance.cu"
#include "cross_rate_bs.cu"
#include "cross_rate_bd.cu"
#include "angular_acceptance.cu"
#ifdef CUDA
#include "toy.cu"
#endif


// that's all folks
