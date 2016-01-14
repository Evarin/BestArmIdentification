classdef policyNaive < ExpPolicy
    % Naive policy for multiple almost best arms identification, both fixed
    % confidence and fixed budget
    %
    % Fixed confidence algorithm from 
    
    properties
        l % number of attempts (for fixed confidence)
        isBudget % is the algorithm run in fixed budget or fixed confidence
    end
    
    methods
        function self = policyNaive()
        end
        
        function init(self, nbActions, mode, horizon, numArms)
            if strcmp(mode, 'confidence')
                eps = horizon(1);
                delta = horizon(2);
                self.l = ceil(4/eps^2 * log(2*nbActions/delta));
                self.isBudget = 0;
            else
                self.isBudget = 1;
            end
            self.t = 1;
            self.k = nbActions;
            self.m = numArms;
            self.N = zeros(1, nbActions);
            self.S = zeros(1, nbActions);
        end
        
        function action = decision(self)
            action = mod(self.t, self.k) + 1;
            self.lastAction = action;
        end
        
        function getReward(self, reward)
            self.N(self.lastAction) = self.N(self.lastAction) + 1; 
            self.S(self.lastAction) = self.S(self.lastAction) + reward;
            self.t = self.t + 1;
        end
        
        function J = getRecommendation(self)
            [~, sp] = sort(self.S ./ self.N, 2, 'descend');
            J = sp(1:self.m);
        end
        
        function r = isConfident(self)
            r = ~self.isBudget && ~any(self.N < self.l);
        end
        
    end
end