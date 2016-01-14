% Compare different strategies

%%% First scenario: Bernoulli bandits
disp('--- First scenario: Bernoulli bandits, From Best Arm Identification');

MAB = {};
for i=1:20
    MAB{i} = armBernoulli(0.4);
end
MAB{7} = armBernoulli(0.5);
game = ExpGame(MAB); fname = 'results/BAI_exp1';

%%% Fixed budgets
disp('-- Fixed budget algorithms');
% Choice of policies to be run
policies = {policyNaive, policyUCB, policyUCBE(2000/game.H1), policyAUCBE, policySR, policyUGapE, policySH};%, policyOptMAI};
%policies = {policySR};

% horizon is length of play, N is number of plays 
horizon = [2000, 0.1]; N = 100;

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
policies = {policyNaive, policyME, policySE, policyLUCB, policyLilUCB};
%policies = {policyExpGap};

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