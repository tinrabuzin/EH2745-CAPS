% Load the Simulink dynamic model
load_system('dyn_model');

%%
tcc_dataset = table(); % Initialize the table

% Iterating through the operating points
for i=1:size(set,1)
        disp('============================================')
        disp(['Comencing estimation for case number ', num2str(i)]);
        disp(['Pgen: ', num2str(set(i,1)), ' MW', ', Qgen: ', num2str(set(i,2)), 'Mvar']);
        
        % Change active and reactive powers of G2
        set_param('dyn_model','FastRestart','off')
        set_param('dyn_model/G2','Pref', num2str(set(i,1)*1e6));
        set_param('dyn_model/G2','Qref', num2str(set(i,2)*1e6));
        
        % Perform power flow and update the model
        % FastRestart has to be disabled before PF calculations
        LF = power_loadflow('-v2', 'dyn_model','solve');
        set_param('dyn_model','FastRestart','on')
        
        % Search algorithm for tcc
        [tcc, found] = deapth_search_tcc('dyn_model',0.001,7,0.01);
        if ~found 
            disp(['tcc not found for case - ',num2str(i)]);
            continue;
        end
        disp(['---> tcc = ', num2str(tcc)])
        
        % Create variables from the power flow results
        Vmag = abs([LF.bus.Vbus]);
        Vang = rad2deg(angle([LF.bus.Vbus]));
        Pgen = real([LF.bus.Sgen]*100);
        Qgen = imag([LF.bus.Sgen]*100); 
        case_nr = {num2str(i)};
        ID = {LF.bus.ID};
        
        %Create a table containing power flow results and estimated tcc
        if isempty(tcc_dataset)
            tcc_dataset = table(tcc, Vmag,Vang,Pgen,Qgen,ID,case_nr);
        else
            tcc_dataset = [tcc_dataset; table(tcc,Vmag,Vang,...
                Pgen,Qgen,ID, case_nr)]; 
        end
        save('temp.mat', 'tcc_dataset');
end

