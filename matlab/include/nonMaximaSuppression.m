function sup = nonMaximaSuppression(c,w,thr)
    c = abs(c);
    sup = zeros(size(c));
    sup(1+w:end-w) = c(1+w:end-w) > thr;
    for i=-w:w
        sup(1+w:end-w) = sup(1+w:end-w) .* (c(1+w:end-w) >= c(1+w+i:end-w+i));
    end
end
