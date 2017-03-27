function [ tcc, found ] = deapth_search_tcc( case_name, tmin, tmax, epsilon )
%DEAPTH_SEARCH_TCC Performs dynamic simulation and searches for a tcc
%   Detailed explanation goes here

load_system('dyn_model');

found = false;

while ~found
    stable_tmax = simulate(tmax);
    stable_tmin = simulate(tmin);
    
    if ~stable_tmax && stable_tmin
        thalf = (tmax-tmin)/2+tmin;
        stable_thalf = simulate(thalf);
        if stable_thalf
            tmin = thalf;
        else
            tmax = thalf;
        end
    else
        tcc = 0;
        disp('Please specify different tmax and tmin');
        break;
    end
    if tmax - tmin < epsilon
        found = true;
        tcc = (tmax-tmin)/2+tmin;
    end

end

end

