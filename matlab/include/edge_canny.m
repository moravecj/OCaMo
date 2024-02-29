function [eI, edgesIndices] = edge_canny(image, ~)
    eI = edge(image, 'canny');
    % Remove the top one third of edges if not needed.
    if nargin == 2
        eI(1:round(size(eI, 1)/3), :) = 0;
    end
    [row, col] = find(eI);
    edgesIndices = [col row];
end