classdef policyAUCBE < ExpPolicy
    % Adaptative UCB-E for fixed budget, best-arm identification
    %
    % From Best Arm Identification in Multi-Armed Bandits
    % by J.-Y. Audibert, S. Bubeck, R. Munos
    
    properties
        a = 1 % Parameter
        T % budget
        nextStop % next t to call checkpoint
        r % round
        H % H_1,k
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
            self.k = nbActions;
            self.T = horizon(1);
            self.N = zeros(1, nbActions);
            self.S = zeros(1, nbActions);
            self.r = 0;
            self.t = 0;
            self.checkpoint();
            self.t = 1;
        end
        
        function action = decision(self)
            if self.t <= self.k
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
                self.H = self.k;
            else
                m = self.S ./ self.N;
                delta = max(m) - m;
                sd = sort(delta); % ascend
                span = (self.k-self.r+1):self.k;
                self.H = max(span ./ (sd(span).^2));
            end
            logK = 0.5 + sum(1 ./ (2:self.k)); 
            self.nextStop = self.t + (self.k - self.r) * ...
                ceil(1/logK * (self.T - self.k) / (self.k - self.r));
            self.r = self.r + 1;
        end
        
        function J = getRecommendation(self)
            [~, J] = max(self.S ./ self.N);
        end
        
    end
end