//this methods are helpful for the translation of cjlm (coefficeints of Legendre)
//to the normweights
WITHIN_KERNEL
ftype integral_ijk_f1(const ftype cosKa, const ftype cosKb,
                      const ftype cosLa, const ftype cosLb,
                      const ftype phia, const ftype phib,
                      const ftype k, const ftype i, const ftype j)
{
  ftype c0 = 9.0/(32.0*M_PI);
  return
    2.0*c0
    *(pow(cosKb,k+3.)-pow(cosKa,k+3.))/(k+3.) //cosK
    *(pow(cosLb,i+1.)/(i+1.)-pow(cosLb,i+3.)/(i+3.)-pow(cosLa,i+1)/(i+1)+pow(cosLa,i+3.)/(i+3.)) //cosL
    *(pow(phib,j+1.)-pow(phia,j+1.))/(j+1.); //phi
}

WITHIN_KERNEL
ftype integral_ijk_f2(const ftype cosKa, const ftype cosKb,
                      const ftype cosLa, const ftype cosLb,
                      const ftype phia, const ftype phib,
                      const ftype k, const ftype i, const ftype j)
{
  ftype c0 = 9.0/(32.0*M_PI);
  return
   c0
   *(pow(cosKb,k+1.)/(k+1.)-pow(cosKb,k+3.)/(k+3.)-pow(cosKa,k+1.)/(k+1.)+pow(cosKa,k+3.)/(k+3.)) //cosK
   *(
     (pow(phib,j+1.)-pow(phia,j+1.))/(j+1.)//phi1
     *(pow(cosLb,i+1.)-pow(cosLa,i+1.))/(i+1.)//cosK1
     -(pow(cosLb,i+1.)/(i+1.)-pow(cosLb,i+3.)/(i+3.)-pow(cosLa,i+1.)/(i+1.)+pow(cosLa,i+3.)/(i+3.)) //cosK2
     *corzo(j, phia, phib)//phi2
     );
}

WITHIN_KERNEL
ftype integral_ijk_f3(const ftype cosKa, const ftype cosKb,
                      const ftype cosLa, const ftype cosLb,
                      const ftype phia, const ftype phib,
                      const ftype k, const ftype i, const ftype j)
{
  ftype c0 = 9.0/(32.0*M_PI);
  return
    c0
    *(pow(cosKb,k+1.)/(k+1.)-pow(cosKb,k+3.)/(k+3.)-pow(cosKa,k+1.)/(k+1.)+pow(cosKa,k+3.)/(k+3.)) //cosL
    *(
      (pow(phib,j+1.)-pow(phia,j+1.))/(j+1.)//phi1
      *(pow(cosLb,i+1.)-pow(cosLa,i+1))/(i+1.)//cosK1
      -(pow(cosLb,i+1.)/(i+1.)-pow(cosLb,i+3.)/(i+3.)-pow(cosLa,i+1.)/(i+1.)+pow(cosLa,i+3.)/(i+3.)) //cosK2
      *maycar(j, phia, phib)//phi2
      );
}

WITHIN_KERNEL
ftype integral_ijk_f4(const ftype cosKa, const ftype cosKb,
                      const ftype cosLa, const ftype cosLb,
                      const ftype phia, const ftype phib,
                      const ftype k, const ftype i, const ftype j)
{
  ftype c0 = 9.0/(32.0*M_PI);
  return
    c0
    *(pow(cosKb,k+1.)/(k+1.)-pow(cosKb,k+3.)/(k+3.)-pow(cosKa,k+1.)/(k+1.)+pow(cosKa,k+3.)/(k+3.)) //cosL
    *(pow(cosLb,i+1.)/(i+1.)-pow(cosLb,i+3.)/(i+3.)-pow(cosLa,i+1.)/(i+1.)+pow(cosLa,i+3.)/(i+3.)) //cosK
    *pozo(2.0, j, phia, phib);//phi
}

WITHIN_KERNEL
ftype integral_ijk_f5(const ftype cosKa, const ftype cosKb,
                      const ftype cosLa, const ftype cosLb,
                      const ftype phia, const ftype phib,
                      const ftype k, const ftype i, const ftype j)
{
  //assert(i>=0);
  //assert(j>=0);
  //assert(k>=0);
  ftype c0 = 9.0/(32.0*M_PI);
  return
    c0/sqrt(2.0)
    *2.0*tarasca(k+1.)//cosL
    *2.0*tarasca(i+1.)//cosK
    *curruncho(1.0, j, phia, phib);//phi
}
WITHIN_KERNEL
ftype integral_ijk_f6(const ftype cosKa, const ftype cosKb,
                      const ftype cosLa, const ftype cosLb,
                      const ftype phia, const ftype phib,
                      const ftype k, const ftype i, const ftype j)
{
  //assert(i>=0);
  //assert(j>=0);
  //assert(k>=0);
  ftype c0 = 9.0/(32.0*M_PI);
  return
    -c0/sqrt(2.0)
    *2.0*tarasca(k+1.)//cosL
    *2.0*tarasca(i+1.)//cosK
    *pozo( 1.0, j,  phia, phib);//phi
}
WITHIN_KERNEL
ftype integral_ijk_f7(const ftype cosKa, const ftype cosKb,
                      const ftype cosLa, const ftype cosLb,
                      const ftype phia, const ftype phib,
                      const ftype k, const ftype i, const ftype j)
{
  //assert(i>=0);
  //assert(j>=0);
  //assert(k>=0);
  const ftype c0 = +3.0/(32.0*M_PI);
  return
    c0*2.0
    *(pow(cosLb,i+1.)/(i+1)-pow(cosLb,i+3.)/(i+3.)-pow(cosLa,i+1.)/(i+1.)+pow(cosLa,i+3.)/(i+3.)) //cosK
    *(pow(phib,j+1.)-pow(phia,j+1.))/(j+1.) //phi
    *(pow(cosKb,k+1.)-pow(cosKa,k+1.))/(k+1.); //cosL
}
WITHIN_KERNEL
ftype integral_ijk_f8(const ftype cosKa, const ftype cosKb,
                      const ftype cosLa, const ftype cosLb,
                      const ftype phia, const ftype phib,
                      const ftype k, const ftype i, const ftype j)
{
  const ftype c0 = +3.0/(32.0*M_PI);
  return
    c0*sqrt(6.0)
    *tarasca(k)//cosL
    *curruncho( 1.0, j, phia, phib)//phi
    *2.0*tarasca(i+1.);//cosK
}


WITHIN_KERNEL
ftype integral_ijk_f9(const ftype cosKa, const ftype cosKb,
                      const ftype cosLa, const ftype cosLb,
                      const ftype phia, const ftype phib,
                      const ftype k, const ftype i, const ftype j)
{
  const ftype c0 = +3.0/(32.0*M_PI);
  return
    -c0*sqrt(6.0)
    *tarasca(k)//cosL
    *(pozo(1.0, j, phia, phib))//phi
    *2.0*tarasca(i+1.);//cosK
}


WITHIN_KERNEL
ftype integral_ijk_f10(const ftype cosKa, const ftype cosKb,
                      const ftype cosLa, const ftype cosLb,
                      const ftype phia, const ftype phib,
                      const ftype k, const ftype i, const ftype j)
{
  const ftype c0 = +3.0/(32.0*M_PI);
  return
    c0*4.0*sqrt(3.0)
    *(pow(cosKb,k+2.)-pow(cosKa,k+2.))/(k+2.) //cosL
    *(pow(phib,j+1.)-pow(phia,j+1.))/(j+1.) //phi
    *(pow(cosLb,i+1.)/(i+1.)-pow(cosLb,i+3.)/(i+3.)-pow(cosLa,i+1.)/(i+1.)+pow(cosLa,i+3.)/(i+3.)); //cosK
}
