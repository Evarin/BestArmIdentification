classdef policySE < ExpPolicy
    % Successive Elimination for fixed confidence and almost best arm
    % identification
    %
    % From Action Elimination and Stopping Conditions for the
    % Multi-Armed Bandit and Reinforcement Learning Problems
    % E. Even-Dar, S. Mannor, Y. Mansour
    
    properties
        delta % probability of success asked
        eps % precision asked
        A % current subset of arms
        r % current round
        nextStop % next t to call checkpoint
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
            self.r = 1;
            self.A = 1:nbActions;
            self.N = zeros(1, nbActions);
            self.S = zeros(1, nbActions);
            self.nextStop = nbActions;
        end
        
        function action = decision(self)
            action = self.A(mod(self.t, length(self.A)) + 1);
            self.lastAction = action;
        end
        
        function getReward(self, reward)
            self.N(self.lastAction) = self.N(self.lastAction) + 1; 
            self.S(self.lastAction) = self.S(self.lastAction) + reward;
            if self.t == self.nextStop
                self.checkpoint();
            end
            self.t = self.t + 1;
        end
        
        function checkpoint(self)
            p = self.S ./ self.N;
            pm = max(p(self.A));
            alpha = 2*sqrt(log(5*length(self.N)*self.r^2)/self.r);
            while true
                [m, im] = max(pm-p(self.A));
                if m < alpha
                    break;
                end
                self.A = self.A(1:length(self.A)~=im);
            end
            self.r = self.r + 1;
            self.nextStop = self.t + length(self.A);
        end
        
        function J = getRecommendation(self)
            J = self.A;
        end
                
        function r = isConfident(self)
            r = length(self.A) == 1;
        end
        
    end
end