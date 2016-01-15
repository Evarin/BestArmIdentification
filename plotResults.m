function plotResults(game, horizon, N, policies, mode, fname)
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
    
    % reading files
    
    % Common y-axis values
    policynames = {};
    perror = zeros(1, l);
    times = zeros(1, l);
    colors = {'r', 'y', 'g', 'c', 'b', 'm', 'k', [0.7 0 0], [0.7 0.7 0], [0 0.7 0], [0 0.7 0.7], [0 0 0.7], [0.7 0 0.7], 'w'};
    for i = 1:l
        % Load saved results and compute regret
        hr = num2str(horizon(1));
        if numel(horizon)>1
            hr = [hr '_' num2str(horizon(2))];
        end
        load([fname '_' mode '_h_' hr '_N_' num2str(N) '_' policies{i}.getId() '.mat']);
        policynames{i} = policyName;
        perror(i) = sum(recommendations ~= iBest)/length(recommendations);
        ptimes(i) = sum(times)/length(times);
    end
    
    % plotting error
    figure(1);
    clf;
    hold on;
    title('Probability of error');
    for i=1:l
        bar(i, perror(i), 'FaceColor', colors{i});
    end
    legend(policynames);
    
    % plotting time
    figure(2);
    clf;
    hold on;
    title('Time to achieve confidence');
    % Common y-axis values
    set(gca,'YScale','log')
    for i=1:l
        bar(i, ptimes(i), 'FaceColor', colors{i});
    end
    axis([0 (l+2) (min(ptimes)/100) max(ptimes)]);
    legend(policynames);
end
