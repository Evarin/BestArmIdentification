classdef policyME < ExpPolicy
    % Median Elimination for fixed confidence, almost best arm
    % identification
    %
    % From Action Elimination and Stopping Conditions for the
    % Multi-Armed Bandit and Reinforcement Learning Problems
    % E. Even-Dar, S. Mannor, Y. Mansour
    
    properties
        delta % inner round parameter
        eps % inner round parameter
        A % current subset of arms
        r % current round
        nextStop % next t to call checkpoint
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
            self.r = 1;
            self.A = 1:nbActions;
            self.N = zeros(1, nbActions);
            self.S = zeros(1, nbActions);
            self.nextStop = 0;
            self.t = 0;
            self.checkpoint();
            self.t = 1;
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
            if self.nextStop > 0
                p = self.S ./ self.N;
                sp = sort(p(self.A));
                med = sp(ceil(length(p)/2));
                self.A = self.A(p(self.A)>=med);
                self.r = self.r + 1;
                self.eps = self.eps * 0.75;
                self.delta = self.delta / 2;
            end
            self.nextStop = self.t + length(self.A) * ...
                round( 4/self.eps^2*log(3/self.delta) );
        end
        
        function J = getRecommendation(self)
            J = self.A;
        end
        
        function r = isConfident(self)
            r = length(self.A) == 1;
        end
        
    end
end