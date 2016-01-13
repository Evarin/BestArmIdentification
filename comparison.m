% Compare different strategies

%%% First scenario: Bernoulli bandits
disp('--- First scenario: Bernoulli bandits');

MAB = {armBernoulli(0.2), armBernoulli(0.1), armBernoulli(0.3)};
game = ExpGame(MAB); fname = 'results/test';

%%% Fixed budgets
disp('-- Fixed budget algorithms');
% Choice of policies to be run
%policies = {policyUCBE, policySR, policyOptMAI};
%policies = {policyUGapE};
policies = {};

% horizon is length of play, N is number of plays 
horizon = [50, 0.1]; N = 20;

% Run everything one policy after each other
defaultStream = RandStream.getGlobalStream; 
savedState = defaultStream.State;
for k = 1:length(policies)
    defaultStream.State = savedState;
    tic; experiment(game, horizon, 1, N, policies{k}, 'budget', fname); toc 
end
plotResults(game, horizon, N, policies, 'budget', fname);

pause;
%%% Fixed budgets
disp('-- Fixed confidence algorithms');
% Choice of policies to be run
policies = {policyNaive, policyLUCB};
policies = {policyLilUCB(0.01, 1, 9)};
policies = {policyExpGap};

% horizon is [eps, delta, m], N is number of plays 
horizon = [0.5, 0.5]; N = 20;

% Run everything one policy after each other
defaultStream = RandStream.getGlobalStream; 
savedState = defaultStream.State;
for k = 1:length(policies)
    defaultStream.State = savedState;
    tic; experiment(game, horizon, 1, N, policies{k}, 'confidence', fname); toc 
end
plotResults(game, horizon, N, policies, 'confidence', fname);