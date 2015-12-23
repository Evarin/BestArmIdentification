function plotResults(game, n, N, policies, fname)
    % Plot the results of different algorithms in a given scenario.
    % When there are more than 4 algoprithms the figure will look
    % best when viewed in full screen.
    %
    % Use: plotResults(game, n, N, policies, fname)
    
    l = length(policies);
    % Find best arm (assuming there is one only...)
    mu = game.means;
    [~, iBest] = max(mu);
    if (length(iBest) > 1)
        warning('Found %d best arms, plots will be incorrect', length(iBest));
    end
    
    clf;
    hold on;
    % Common y-axis values
    ar = -Inf;
    ad = -Inf;
    policynames = {};
    colors = {'r', 'g', 'b', 'k', 'y', 'p'};
    for i = 1:l
        % Load saved results and compute regret
        load([fname '_n_' num2str(n) '_N_' num2str(N) '_' class(policies{i})]);
        policyName = class(policies{i}); policyName = policyName(7:end);
        policynames{i} = policyName;
        perror = sum(recommendations((1:length(mu)) ~= iBest))/sum(recommendations)
        bar(i, perror);
    end
    legend(policynames);
end