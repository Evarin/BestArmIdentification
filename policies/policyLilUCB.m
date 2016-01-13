classdef policyLilUCB < ExpPolicy
    % lil'UCB for all bandits
    %
    % From lil’ UCB : An Optimal Exploration Algorithm for Multi-Armed
    % Bandits, Jamieson, Malloy, Nowak, Bubeck, 2014
    
    properties
        t % Number of the round
        N % Number of times each action has been chosen
        S % Cumulated reward with each action
        lastAction % Stores the last action played
        delta
        eps = 0.01
        beta = 1
        lambda = 9
        m % Useless
        sigma % Parameter (subgaussian variance)
        round % Current round
        stopping % stopping criterion reached
    end
    
    methods
        function self = policyLilUCB(eps, beta, lambda)
            self.sigma = 0.5; % a<X<b -> sigma = |b-a|/2
            if nargin >= 1
                self.eps = eps; % [0.01] or [0]
            end
            if nargin >= 2
                self.beta = beta; % [1] or [0.5]
            end
            if nargin >= 3
                self.lambda = lambda; % [((2+beta)/beta)^2] or [1+10/n]
            end
        end
        
        function init(self, nbActions, mode, horizon, numArms)
            if ~strcmp(mode, 'confidence')
                throw(MException('EXPPOLICY:BadParameter', ...
                    'LilUCB can only be used for fixed confidence'));
            end
            if numArms > 1
                throw(MException('EXPPOLICY:BadParameter', ...
                    'LilUCB can only find the best arm'));
            end
            self.eps = horizon(1);
            self.delta = horizon(2);
            self.m = 1;
            self.t = 1;
            self.round = 1;
            self.N = zeros(1, nbActions);
            self.S = zeros(1, nbActions);
            self.stopping = 0;
        end
        
        function action = decision(self)
            if self.t <= length(self.N)
                action = self.t;
            else
                mu = self.S./self.N;
                rac = 2*self.sigma^2*(1+self.eps)*log(log((1+self.eps).*self.N)/self.delta)./self.N;
                rac(rac<0) = +Inf;
                fun = mu + (1+self.beta)*(1+sqrt(self.eps))*sqrt(rac);
                [~, action] = max(fun);
            end
            self.lastAction = action;
        end
        
        function getReward(self, reward)
            self.N(self.lastAction) = self.N(self.lastAction) + 1; 
            self.S(self.lastAction) = self.S(self.lastAction) + reward;
            self.t = self.t + 1;
        end
        
        function J = getRecommendation(self)
            [~, J] = max(self.N);
        end
        
        function b = fbeta(self, u, t)
            b = sqrt(0.5*log(1.25*length(self.N)*t^4/self.delta)./u);
        end
        
        function r = isConfident(self)
            r = self.t>length(self.N) && any(self.N >= 1 + self.lambda.*(sum(self.N)-self.N));
        end
        
    end
end