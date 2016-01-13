classdef policyUCBE < ExpPolicy
    % UCB-E for any bandit
    
    properties
        t % Number of the round
        lastAction % Stores the last action played
        N % Number of times each action has been chosen
        S % Cumulated reward with each action
        a = 1 % Parameter
    end
    
    methods
        function self = policyUCBE(a)
            if (nargin >= 1)
                self.a = a;
            end
        end
        
        function init(self, nbActions, mode, horizon, numArms)
            if ~strcmp(mode, 'budget')
                throw(MException('EXPPOLICY:BadParameter', ...
                    'UCB-E can only be used for fixed budget'));
            end
            if numArms > 1
                throw(MException('EXPPOLICY:BadParameter', ...
                    'SR can only find the best arm'));
            end
            self.t = 1;
            self.N = zeros(1, nbActions);
            self.S = zeros(1, nbActions);
        end
        
        function action = decision(self)
            if any(self.N==0)
                action = find(self.N==0, 1);
            else
                ucb =  self.S./self.N + sqrt(self.a./self.N);
                m = max(ucb); I = find(ucb == m);
                action = I(1+floor(length(I)*rand));
            end
            self.lastAction = action;
        end
        
        function getReward(self, reward)
            self.N(self.lastAction) = self.N(self.lastAction) + 1; 
            self.S(self.lastAction) = self.S(self.lastAction)  + reward;
            self.t = self.t + 1;
        end
        
        function J = getRecommendation(self)
            [~, J] = max(self.S./self.N);
        end
        
    end
end