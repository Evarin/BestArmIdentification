classdef policyUCB < ExpPolicy
    % The classic UCB policy, with optimal tuning
    % of the constant (=1/2), for fixed budget and best arm identification
    %
    % Rewards are assumed to be bounded in [0,1]


    properties
        c = 1/2 % Parameter of the UCB
    end
    
    methods
        function self = policyUCB()            
        end
        
        function init(self, nbActions, mode, horizon, numArms)
            if ~strcmp(mode, 'budget')
                throw(MException('EXPPOLICY:BadParameter', ...
                    'UCB can only be used for fixed budget'));
            end
            if numArms > 1
                throw(MException('EXPPOLICY:BadParameter', ...
                    'UCB can only find the best arm'));
            end
            self.t = 1;
            self.N = zeros(1, nbActions);
            self.S = zeros(1, nbActions);
        end
        
        function action = decision(self)
            if any(self.N == 0)
                action = find(self.N==0, 1);
            else
                ucb =  self.S./self.N + sqrt(self.c*log(self.t)./self.N);
                m = max(ucb); I = find(ucb == m);
                action = I(1+floor(length(I)*rand));
            end
            self.lastAction = action;
        end
        
        function getReward(self, reward)
            self.N(self.lastAction) = self.N(self.lastAction) + 1; 
            self.S(self.lastAction) = self.S(self.lastAction) + reward;
            self.t = self.t + 1;
        end
        
        function J = getRecommendation(self)
            [~, J] = max(self.S./self.N);
        end
    end

end
