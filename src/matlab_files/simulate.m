function [ stable ] = simulate( dt )
%SIMULATE Summary of this function goes here
%   Detailed explanation goes here
load_system('dyn_model');
    set_param('dyn_model/F1','SwitchTimes',...
            ['[',num2str(1),' ',num2str(1+dt),']']);
        simout = sim('dyn_model');
        stop = simout.get('stop');
        stable = ~any(stop.Data);
end

