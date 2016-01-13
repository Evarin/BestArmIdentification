classdef policyNaive < ExpPolicy
    % Naive policy for fixed confidence and any bandit
    
    properties
        lastAction % Stores the last action played
        N % Number of times each action has been chosen
        S % Cumulated reward with each action
        eps % destination confidence
        delta
        m
        l % number of attempts
    end
    
    methods
        function self = policyNaive()
        end
        
        function init(self, nbActions, mode, horizon)
            if ~strcmp(mode, 'confidence')
                throw(MException('EXPPOLICY:BadParameter', ...
                    'Naive policy can only be used for fixed confidence'));
            end
            self.eps = horizon(1);
            self.delta = horizon(2);
            self.m = 1;
            self.l = ceil(4/self.eps^2 * log(2*nbActions/self.delta));
            self.N = zeros(1, nbActions);
            self.S = zeros(1, nbActions);
        end
        
        function action = decision(self)
            action = find(self.N<self.l, 1);
            self.lastAction = action;
        end
        
        function getReward(self, reward)
            self.N(self.lastAction) = self.N(self.lastAction) + 1; 
            self.S(self.lastAction) = self.S(self.lastAction) + reward;
        end
        
        function J = getRecommendation(self)
            [~, J] = max(self.S./self.N);
        end
        
        function r = isConfident(self)
            r = ~any(self.N<self.l);
        end
        
    end
end