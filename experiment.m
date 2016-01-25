function experiment(game, horizon, numArms, N, policy, mode, fname)
    % Run a number of experiments and save the results subsampling in time
    % Use: experiment(game, n, N, policy, mode, fname),
    % n is the horizon and N the number of played games.
    % numArms is the number of best Arms we want to obtain
    % for mode = 'budget', n is an int
    % for mode = 'confidence', n is a couple [epsilon, delta]
    
    if nargin<6, fname = 'results/exp'; end % Defaut file name
    
    % Recommendations
    recommendations = zeros(1, N);
    times = zeros(1, N);
    
    fprintf('%s %d: ', policy.getName(), N);
    H1 = game.H1;
    policyName = policy.getName();
    % Expectations of the arms
    mu = game.means;
    hr = num2str(horizon(1));
    if numel(horizon)>1
        hr = [hr '_' num2str(horizon(2))];
    end
    foutput = [fname '_' mode '_h_' hr '_N_' num2str(N) '_' policy.getId() '.mat'];
    if exist(foutput, 'file')
        return
    end
        for j = 1:N
        fprintf('|%d', j);
        [r, t] = game.play(policy, mode, horizon, numArms);
        if isempty(r)
            r = 0;
        end
        recommendations(j) = r;
        fprintf(':%d', r(1));
        times(j) = t;
        % Once every N/50 runs, display something and save current state
        % of variables
        if (rem(j, floor(N/50))==0) || (j == N)
            save(foutput,...
              'mu', 'horizon', 'N', 'recommendations', 'times', 'H1', 'policyName');
        end
    end
    fprintf('\n');   
end
