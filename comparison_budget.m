% Compare different strategies
% Fixed budgets
disp('### Fixed budget algorithms');

%% First scenario: Bernoulli bandits    BAI 1
disp('--- First scenario: Bernoulli bandits, One group of bad arms');

% Onde group of bad arms
MAB = {};
for i=1:20
    MAB{i} = armBernoulli(0.4);
end
MAB{7} = armBernoulli(0.5);
game = ExpGame(MAB); fname = 'results/BAI_exp1';

% Choice of policies to be run
policies = {policyNaive, policyUCB, policyUCBE(2000/game.H1, ' a=n/H'), ...
    policyUCBE(2000/game.H1, ' a=4n/H'), policyAUCBE(0.5), ...
    policyAUCBE(1), policySR, policyUGapE, policySH, policyOptMAI};

% horizon is length of play, N is number of plays 
horizon = [2000, 0.1]; N = 50;

% Run everything one policy after each other
defaultStream = RandStream.getGlobalStream; 
savedState = defaultStream.State;
for k = 1:length(policies)
    defaultStream.State = savedState;
    tic; experiment(game, horizon, 1, N, policies{k}, 'budget', fname); toc 
end
plotResults(game, horizon, N, policies, 'budget', fname);

pause;



%% Second scenario: Bernoulli bandits    BAI 2
disp('--- Second scenario: Bernoulli bandits, Two groups of bad arms');

MAB = {};
for i=1:5
    MAB{i} = armBernoulli(0.42);
end
for i=6:20
    MAB{i} = armBernoulli(0.38);
end
MAB{14} = armBernoulli(0.5);
game = ExpGame(MAB); fname = 'results/BAI_exp2';

% Choice of policies to be run
policies = {policyNaive, policyUCB, policyUCBE(2000/game.H1, ' a=n/H'), ...
    policyUCBE(2000/game.H1, ' a=4n/H'), policyAUCBE(0.5), ...
    policyAUCBE(1), policySR, policyUGapE, policySH, policyOptMAI};

% horizon is length of play, N is number of plays 
horizon = [2000, 0.1]; N = 50;

% Run everything one policy after each other
defaultStream = RandStream.getGlobalStream; 
savedState = defaultStream.State;
for k = 1:length(policies)
    defaultStream.State = savedState;
    tic; experiment(game, horizon, 1, N, policies{k}, 'budget', fname); toc 
end
plotResults(game, horizon, N, policies, 'budget', fname);

pause;



%% Third scenario: Bernoulli bandits    BAI 3
disp('--- Third scenario: Bernoulli bandits, Geometric progression');

MAB = {};
MAB{1} = armBernoulli(0.5);
for i=2:4
    MAB{i} = armBernoulli(0.5-0.37^i);
end
game = ExpGame(MAB); fname = 'results/BAI_exp3';

% Choice of policies to be run
policies = {policyNaive, policyUCB, policyUCBE(2000/game.H1, ' a=n/H'), ...
    policyUCBE(2000/game.H1, ' a=4n/H'), policyAUCBE(0.5), ...
    policyAUCBE(1), policySR, policyUGapE, policySH, policyOptMAI};

% horizon is length of play, N is number of plays 
horizon = [2000, 0.1]; N = 50;

% Run everything one policy after each other
defaultStream = RandStream.getGlobalStream; 
savedState = defaultStream.State;
for k = 1:length(policies)
    defaultStream.State = savedState;
    tic; experiment(game, horizon, 1, N, policies{k}, 'budget', fname); toc 
end
plotResults(game, horizon, N, policies, 'budget', fname);

pause;




%% Fourth scenario: Bernoulli bandits    BAI 6
disp('--- Fourth scenario: Bernoulli bandits, Two good arms and a large group of bad arms');

MAB = {};
MAB{1} = armBernoulli(0.5);
MAB{2} = armBernoulli(0.48);
for i=3:20
    MAB{i} = armBernoulli(0.37);
end
game = ExpGame(MAB); fname = 'results/BAI_exp6';

% Choice of policies to be run
policies = {policyNaive, policyUCB, policyUCBE(2000/game.H1, ' a=n/H'), ...
    policyUCBE(2000/game.H1, ' a=4n/H'), policyAUCBE(0.5), ...
    policyAUCBE(1), policySR, policyUGapE, policySH, policyOptMAI};

% horizon is length of play, N is number of plays 
horizon = [6000, 0.1]; N = 50;

% Run everything one policy after each other
defaultStream = RandStream.getGlobalStream; 
savedState = defaultStream.State;
for k = 1:length(policies)
    defaultStream.State = savedState;
    tic; experiment(game, horizon, 1, N, policies{k}, 'budget', fname); toc 
end
plotResults(game, horizon, N, policies, 'budget', fname);

pause;


%% Fifth scenario: Truncated Exponential bandits
disp('--- Fifth scenario: Exponential bandit vs Bernouilli bad arms');

MAB = {};
MAB{1} = armExp(0.5);
for i=1:20
    MAB{i} = armBernoulli(0.4);
end
game = ExpGame(MAB); fname = 'results/expArms1';

% Choice of policies to be run
policies = {policyNaive, policyUCB, policyUCBE(2000/game.H1, ' a=n/H'), ...
    policyUCBE(2000/game.H1, ' a=4n/H'), policyAUCBE(0.5), ...
    policyAUCBE(1), policySR, policyUGapE, policySH, policyOptMAI};

% horizon is length of play, N is number of plays 
horizon = [6000, 0.1]; N = 50;

% Run everything one policy after each other
defaultStream = RandStream.getGlobalStream; 
savedState = defaultStream.State;
for k = 1:length(policies)
    defaultStream.State = savedState;
    tic; experiment(game, horizon, 1, N, policies{k}, 'budget', fname); toc 
end
plotResults(game, horizon, N, policies, 'budget', fname);

pause;


%% First scenario: Truncated Exponential bandits
disp('--- Sixth scenario: Exponential bandits, One group of bad arms');

MAB = {};
MAB{1} = armExp(0.5);
for i=1:20
    MAB{i} = armExp(0.3);
end
game = ExpGame(MAB); fname = 'results/expexpArms1';

% Choice of policies to be run
policies = {policyNaive, policyUCB, policyUCBE(2000/game.H1, ' a=n/H'), ...
    policyUCBE(2000/game.H1, ' a=4n/H'), policyAUCBE(0.5), ...
    policyAUCBE(1), policySR, policyUGapE, policySH, policyOptMAI};

% horizon is length of play, N is number of plays 
horizon = [6000, 0.1]; N = 50;

% Run everything one policy after each other
defaultStream = RandStream.getGlobalStream; 
savedState = defaultStream.State;
for k = 1:length(policies)
    defaultStream.State = savedState;
    tic; experiment(game, horizon, 1, N, policies{k}, 'budget', fname); toc 
end
plotResults(game, horizon, N, policies, 'budget', fname);

pause;


%% Seventh scenario: Truncated Exponential bandits
disp('--- Seventh scenario: Exponential bandits, One group of bad arms, T=12000');

MAB = {};
MAB{1} = armExp(0.5);
for i=1:20
    MAB{i} = armExp(0.3);
end
game = ExpGame(MAB); fname = 'results/expArms3';

% Choice of policies to be run
policies = {policyNaive, policyUCB, policyUCBE(2000/game.H1, ' a=n/H'), ...
    policyUCBE(2000/game.H1, ' a=4n/H'), policyAUCBE(0.5), ...
    policyAUCBE(1), policySR, policyUGapE, policySH, policyOptMAI};

% horizon is length of play, N is number of plays 
horizon = [12000, 0.1]; N = 50;

% Run everything one policy after each other
defaultStream = RandStream.getGlobalStream; 
savedState = defaultStream.State;
for k = 1:length(policies)
    defaultStream.State = savedState;
    tic; experiment(game, horizon, 1, N, policies{k}, 'budget', fname); toc 
end
plotResults(game, horizon, N, policies, 'budget', fname);

pause;