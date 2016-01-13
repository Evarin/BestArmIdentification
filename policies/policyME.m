classdef policyME < ExpPolicy
    % Median Elimination for all bandits
    %
    % From Action Elimination and Stopping Conditions for the
    % Multi-Armed Bandit and Reinforcement Learning Problems
    % Even-Dar, Mannor, Mansour
    
    properties
        t % Number of the round
        N % Number of times each action has been chosen
        S % Cumulated reward with each action
        lastAction % Stores the last action played
        delta
        eps
        A
        l
        lastA
        numsamples
    end
    
    methods
        function self = policyME()
        end
        
        function init(self, nbActions, mode, horizon, numArms)
            if ~strcmp(mode, 'confidence')
                throw(MException('EXPPOLICY:BadParameter', ...
                    'Median Elimination can only be used for fixed confidence'));
            end
            if numArms > 1
                throw(MException('EXPPOLICY:BadParameter', ...
                    'Median Elimination can only find the best arm'));
            end
            self.eps = horizon(1)/4;
            self.delta = horizon(2)/2;
            self.t = 1;
            self.l = 1;
            self.A = 1:nbActions;
            self.N = zeros(1, nbActions);
            self.S = zeros(1, nbActions);
            self.lastA = 1;
            self.numsamples = round(1/(self.eps/2)^2*log(3/self.delta));
        end
        
        function action = decision(self)
            action = mod(self.t, length(self.A)) + 1;
            self.lastAction = action;
        end
        
        function getReward(self, reward)
            self.N(self.lastAction) = self.N(self.lastAction) + 1; 
            self.S(self.lastAction) = self.S(self.lastAction) + reward;
            if self.t - self.lastA == length(self.A) * self.numsamples
                p = self.S ./ self.N;
                sp = sort(p(self.A));
                med = sp(ceil(length(p)/2));
                self.A = self.A(p(self.A)>=med);
                self.lastA = self.t;
                self.l = self.l + 1;
                self.eps = self.eps * 0.75;
                self.delta = self.delta / 2;
                self.numsamples = round(1/(self.eps/2)^2*log(3/self.delta));
            end
            self.t = self.t + 1;
        end
        
        function J = getRecommendation(self)
            J = self.A;
        end
        
        function r = isConfident(self)
            r = length(self.A) == 1;
        end
        
    end
end