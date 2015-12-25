function experiment(game, horizon, N, policy, mode, fname)
    % Run a number of experiments and save the results subsampling in time
    % Use: experiment(game, n, N, policy, mode, fname),
    % n is the horizon and N the number of played games.
    % for mode = 'budget', n is an int
    % for mode = 'confidence', n is a couple [epsilon, delta]
    
    if nargin<6, fname = 'results/exp'; end % Defaut file name
    
    % Recommendations
    recommendations = zeros(1, N);
    times = zeros(1, N);
    
    fprintf('%s %d:', class(policy), N);
    for j = 1:N, 
        [r, t] = game.play(policy, mode, horizon);
        recommendations(j) = r;
        times(j) = t;
        % Once every N/50 runs, display something and save current state
        % of variables
        if (rem(j, floor(N/50))==0) || (j == N)
            fprintf(' %d', j);
            % Expectations of the arms
            mu = game.means;
            hr = num2str(horizon(1));
            if numel(horizon)>1
                hr = [hr '_' num2str(horizon(2))];
            end
            save([fname '_' mode '_h_' hr '_N_' num2str(N) '_' class(policy) '.mat'],...
              'mu', 'horizon', 'N', 'recommendations', 'times');
        end
    end
    fprintf('\n');   
end
