function experiment(game, n, N, policy, fname)
    % Run a number of experiments and save the results subsampling in time
    % Use: experiment(game, n, N, policy, tsave, fname),
    % n is the horizon and N the number of played games.
    %
    % authors: Olivier Cappé and Aurélien Garivier

    % $Id: experiment.m,v 1.17 2012-06-06 09:42:04 cappe Exp $
    
    if nargin<5, fname = 'results/exp'; end % Defaut file name
    
    % Recommendations
    recommendations = zeros(1, N);
    
    fprintf('%s %d:', class(policy), N);
    for j = 1:N, 
        recommendations(j) = recommendations(j) + game.play(policy, n);
        % Once every N/50 runs, display something and save current state
        % of variables
        if (rem(j, floor(N/50))==0) || (j == N)
            fprintf(' %d', j);
            % Expectations of the arms
            mu = game.means;
            save([fname '_n_' num2str(n) '_N_' num2str(N) '_' class(policy)],...
              'mu', 'n', 'N', 'recommendations');
        end
    end
    fprintf('\n');   
end
