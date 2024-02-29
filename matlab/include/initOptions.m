function opt = initOptions()
    opt.TLiDARDistance = 0.01;
    opt.TLiDARIntensity = 0.05;
    opt.TLiDARAzimuth = 0.1;
    opt.sigma = 9;
    opt.k = 10;
    opt.sequential = 1;
    x = -5:1:5;
    opt.mask = -x.*exp(-(x.^2)/2);
    opt.NMSLiDARDistance = 2;
    opt.NMSLiDARIntensity = 3;
end
