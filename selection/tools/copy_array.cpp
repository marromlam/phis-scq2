std::vector <float> copy_array(ROOT::VecOps::RVec<Float_t> input_vector)
{
    std::vector<float> output_vector;
    for(auto &i : input_vector){
        output_vector.push_back(i);
        }

    return output_vector;
}
