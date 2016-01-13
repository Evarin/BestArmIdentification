classdef policySE < ExpPolicy
    % Successive Elimination for all bandits
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
        round
        lastA
    end
    
    methods
        function self = policySE()
        end
        
        function init(self, nbActions, mode, horizon, numArms)
            if ~strcmp(mode, 'confidence')
                throw(MException('EXPPOLICY:BadParameter', ...
                    'Successive Elimination can only be used for fixed confidence'));
            end
            if numArms > 1
                throw(MException('EXPPOLICY:BadParameter', ...
                    'Successive Elimination can only find the best arm'));
            end
            self.eps = horizon(1);
            self.delta = horizon(2);
            self.t = 1;
            self.round = 1;
            self.A = 1:nbActions;
            self.N = zeros(1, nbActions);
            self.S = zeros(1, nbActions);
            self.lastA = 1;
        end
        
        function action = decision(self)
            action = mod(self.t, length(self.A)) + 1;
            self.lastAction = action;
        end
        
        function getReward(self, reward)
            self.N(self.lastAction) = self.N(self.lastAction) + 1; 
            self.S(self.lastAction) = self.S(self.lastAction) + reward;
            if self.t - self.lastA == length(self.A)
                p = self.S ./ self.N;
                pm = max(p(self.A));
                alpha = 2*sqrt(log(5*length(self.N)*self.round^2)/self.round);
                % [ pm - p(self.A), alpha]
                while true
                    [m, im] = max(pm-p(self.A));
                    if m < alpha
                        break;
                    end
                    self.A = self.A(1:length(self.A)~=im);
                end
                self.lastA = self.t;
                self.round = self.round + 1;
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