% Compare different strategies

%%% First scenario: Bernoulli bandits
disp('--- First scenario: Bernoulli bandits');

MAB = {armBernoulli(0.2), armBernoulli(0.1), armBernoulli(0.3)};
game = ExpGame(MAB); fname = 'results/test';

% Choice of policies to be run
policies = {policyUCBE, policySR};

% n is length of play, N is number of plays 
n = 500; N = 20;

% Time indices at which the results will be saved
tsave = round(linspace(1,n,200));

% Run everything one policy after each other
defaultStream = RandStream.getGlobalStream; 
savedState = defaultStream.State;
for k = 1:length(policies)
    defaultStream.State = savedState;
    tic; experiment(game, n, N, policies{k}, fname); toc 
end
plotResults(game, n, N, policies, fname);
