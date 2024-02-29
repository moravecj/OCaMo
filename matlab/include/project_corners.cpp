#include <mex.h>
#include <iostream>
#include <vector>
#include <cmath>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) 
{ 
    double * pts = mxGetPr(prhs[0]);
    
    mwSize numOfPtsRows = mxGetM(prhs[0]);
    mwSize numOfPtsCols = mxGetN(prhs[0]);

    double * TLid2Cam = mxGetPr(prhs[1]);
    double * K = mxGetPr(prhs[2]);
    
    double r_bnd = mxGetPr(prhs[3])[0];
    double r_diff = mxGetPr(prhs[4])[0];
    double t_bnd = mxGetPr(prhs[5])[0];
    double t_diff = mxGetPr(prhs[6])[0];
    int num_of_params = mxGetPr(prhs[7])[0];
    
    plhs[0] = mxCreateNumericMatrix(numOfPtsCols, num_of_params * numOfPtsRows, mxDOUBLE_CLASS, mxREAL);
    plhs[1] = mxCreateNumericMatrix(1, num_of_params * numOfPtsRows, mxDOUBLE_CLASS, mxREAL);
    double * pts_ts = mxGetPr(plhs[0]);
    double * pts_ts_idx = mxGetPr(plhs[1]);
    
    int cnt = 0;
    
    for(double r1=-r_bnd; r1 <= r_bnd + 0.0001; r1+= r_diff){
    for(double r2=-r_bnd; r2 <= r_bnd + 0.0001; r2+= r_diff){
    for(double r3=-r_bnd; r3 <= r_bnd + 0.0001; r3+= r_diff){
    for(double t1=-t_bnd; t1 <= t_bnd + 0.0001; t1+= t_diff){
    for(double t2=-t_bnd; t2 <= t_bnd + 0.0001; t2+= t_diff){
    for(double t3=-t_bnd; t3 <= t_bnd + 0.0001; t3+= t_diff){
        double R11 = std::cos(r2) * std::cos(r3);
        double R12 = -std::cos(r2) * std::sin(r3);
        double R13 = std::sin(r2);
        double R21 = std::sin(r1) * std::sin(r2) * std::cos(r3) + std::cos(r1) * std::sin(r3);
        double R22 = -std::sin(r1) * std::sin(r2) * std::sin(r3) + std::cos(r1) * std::cos(r3);
        double R23 = -std::sin(r1) * std::cos(r2);
        double R31 = -std::cos(r1) * std::sin(r2) * std::cos(r3) + std::sin(r1) * std::sin(r3);
        double R32 = std::cos(r1) * std::sin(r2) * std::sin(r3) + std::sin(r1) * std::cos(r3);
        double R33 = std::cos(r1) * std::cos(r2);
        
        int param_offset = numOfPtsCols * numOfPtsRows * cnt;
        for(int i = 0; i < numOfPtsRows; i++){
            double x = pts[i];
            double y = pts[i + numOfPtsRows];
            double z = pts[i + 2 * numOfPtsRows];
            double x_new, y_new, z_new;
            x_new = R11 * x + R12 * y + R13 * z + t1;
            y_new = R21 * x + R22 * y + R23 * z + t2;
            z_new = R31 * x + R32 * y + R33 * z + t3;

            x = x_new;
            y = y_new;
            z = z_new;

            x_new = TLid2Cam[0*4 + 0] * x + TLid2Cam[1*4 + 0] * y + TLid2Cam[2*4 + 0] * z + TLid2Cam[3*4 + 0];
            y_new = TLid2Cam[0*4 + 1] * x + TLid2Cam[1*4 + 1] * y + TLid2Cam[2*4 + 1] * z + TLid2Cam[3*4 + 1];
            z_new = TLid2Cam[0*4 + 2] * x + TLid2Cam[1*4 + 2] * y + TLid2Cam[2*4 + 2] * z + TLid2Cam[3*4 + 2];

            x = x_new;
            y = y_new;
            z = z_new;

            x_new = K[0*3 + 0] * x + K[1*3 + 0] * y + K[2*3 + 0] * z;
            y_new = K[0*3 + 1] * x + K[1*3 + 1] * y + K[2*3 + 1] * z;
            z_new = K[0*3 + 2] * x + K[1*3 + 2] * y + K[2*3 + 2] * z;

            if(z_new > 1){
                x_new = x_new / z_new;
                y_new = y_new / z_new;
                z_new = z_new / z_new;
                if(x_new < 0 || x_new > 1919 || y_new < 0 || y_new > 1279){
                    x_new = -1;
                    y_new = -1;
                    z_new = -1;
                }
            }else{
                x_new = -1;
                y_new = -1;
                z_new = -1;
            }
            pts_ts[3*i + 0 + param_offset] = x_new;
            pts_ts[3*i + 1 + param_offset] = y_new;
            pts_ts[3*i + 2 + param_offset] = z_new;
            pts_ts_idx[i + numOfPtsRows * cnt] = cnt + 1;
        }
        cnt += 1;
    }
    }
    }
    }
    }
    }
}