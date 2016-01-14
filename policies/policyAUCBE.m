classdef policyAUCBE < ExpPolicy
    % Adaptative UCB-E for any bandit
    %
    % From 
    % Audibert, Bubeck, Munos
    
    properties
        t % Number of the round
        lastAction % Stores the last action played
        N % Number of times each action has been chosen
        S % Cumulated reward with each action
        a = 1 % Parameter
        nextStop
        r
        T
        H
    end
    
    methods
        function self = policyAUCBE(c)
            if (nargin >= 1)
                self.a = c;
            end
        end
        
        function init(self, nbActions, mode, horizon, numArms)
            if ~strcmp(mode, 'budget')
                throw(MException('EXPPOLICY:BadParameter', ...
                    'Adaptative UCB-E can only be used for fixed budget'));
            end
            if numArms > 1
                throw(MException('EXPPOLICY:BadParameter', ...
                    'Adaptative UCB-E can only find the best arm'));
            end
            self.T = horizon(1);
            self.t = 1;
            self.N = zeros(1, nbActions);
            self.S = zeros(1, nbActions);
            self.r = 0;
            self.checkpoint();
        end
        
        function action = decision(self)
            if self.t <= length(self.N)
                action = self.t;
            else
                B = self.S ./ self.N + sqrt(self.a * self.T ./ self.H ./ self.N);
                [~, action] = max(B); 
            end
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
            if self.r == 0
                self.H = length(self.N);
            else
                m = self.S ./ self.N;
                delta = max(m) - m;
                sd = sort(delta);
                span = (length(self.N)-self.r+1):length(self.N);
                self.H = max(span./(sd(span).^2));
            end
            logK = 0.5 + sum(1./(2:length(self.N))); 
            self.nextStop = self.t + (length(self.N) - self.r) * ceil(1/logK * (self.T - length(self.N)) / (length(self.N)-self.r));
            self.r = self.r + 1;
        end
        
        function J = getRecommendation(self)
            [~, J] = max(self.S./self.N);
        end
        
    end
end