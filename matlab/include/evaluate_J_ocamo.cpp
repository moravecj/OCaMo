#include <mex.h>
#include <iostream>
#include <vector>
#include <cmath>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) 
{ 
    double * dist = mxGetPr(prhs[0]);
    
    mwSize numOfRows = mxGetM(prhs[0]);
    mwSize numOfCols = mxGetN(prhs[0]);
        
    double * idxs = mxGetPr(prhs[1]);
    
    mwSize numOfPtsRows = mxGetM(prhs[1]);
    mwSize numOfPtsCols = mxGetN(prhs[1]);
    
    int num_of_params = mxGetPr(prhs[2])[0];
    double sigma = mxGetPr(prhs[3])[0];
    double edge_cnt = mxGetPr(prhs[4])[0];
    
    plhs[0] = mxCreateNumericMatrix(num_of_params, 1, mxDOUBLE_CLASS, mxREAL);
    double * J_val = mxGetPr(plhs[0]);
    double * J_cnt = new double[num_of_params];
    for(int i = 0; i < num_of_params; i++){
        J_val[i] = 0;
        J_cnt[i] = 0;
    }
    for (int j = 0; j < numOfRows; j++){
        for (int i = 0; i < numOfCols; i++){
            J_val[int(idxs[j]) - 1] += (std::exp(-(dist[i * numOfRows + j] * dist[i * numOfRows + j]) / (2.0 * sigma * sigma)));
        }
        J_cnt[int(idxs[j]) - 1] += 1;
    }
    
    for(int i = 0; i < num_of_params; i++)
        J_val[i] = - J_val[i] / (J_cnt[i] * edge_cnt);
    
    delete[] J_cnt;
}