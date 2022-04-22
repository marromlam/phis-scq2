std::vector<double> beam_shift(const ROOT::VecOps::RVec<Float_t>& PVX, const ROOT::VecOps::RVec<Float_t>& PVY, int nPVs)
{
    double sum_of_PVX = 0.0;
    for (auto& nx : PVX)
        sum_of_PVX += nx;

    double sum_of_PVY = 0.0;
    for (auto& ny : PVY)
        sum_of_PVY += ny;

    std::vector<double> beamPosition;

    beamPosition.push_back(sum_of_PVX / nPVs);
    beamPosition.push_back(sum_of_PVY / nPVs);

    return beamPosition;
};

double doca_z(double beam_shift_X, double beam_shift_Y,double B_ENDVERTEX_X, double B_ENDVERTEX_Y, const ROOT::VecOps::RVec<Float_t>& PX, const ROOT::VecOps::RVec<Float_t>& PY)
{

    double PT = sqrt(PX[0]*PX[0] + PY[0]*PY[0]);

    double DOCAz = std::abs(1./PT*((B_ENDVERTEX_Y-beam_shift_X)*PX[0] - (B_ENDVERTEX_X-beam_shift_Y)*PY[0]));

    return DOCAz;
}