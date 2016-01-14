classdef policySR < ExpPolicy
    % Successive Rejects for fixed budget and best arm identification
    %
    % From Best Arm Identification in Multi-Armed Bandits
    % by J.-Y. Audibert, S. Bubeck, R. Munos
    
    properties
        A % current subset of arms
        steps % n_0...(K-1)
        r % current round
        nextStop % next t to call checkpoint
    end
    
    methods
        function self = policySR()
        end
        
        function init(self, nbActions, mode, horizon, numArms)
            if ~strcmp(mode, 'budget')
                throw(MException('EXPPOLICY:BadParameter', ...
                    'SR can only be used for fixed budget'));
            end
            if numArms > 1
                throw(MException('EXPPOLICY:BadParameter', ...
                    'SR can only find the best arm'));
            end
            self.A = 1:nbActions;
            T = horizon(1);
            logK = 0.5 + sum(1./(2:nbActions)); 
            nk = ceil(((T-nbActions)/logK) ./...
                (nbActions - (0:(nbActions-2))) );
            dnk = nk - [0 nk(1:end-1)];
            self.steps = [0 cumsum(dnk.*(nbActions:-1:2)) T+1];
            self.r = 2;
            self.nextStop = self.steps(self.r);
            self.t = 1;
            self.N = zeros(1, nbActions);
            self.S = zeros(1, nbActions);
        end
        
        function action = decision(self)
            action = self.A(mod(self.t, length(self.A)) + 1);
            self.lastAction = action;
        end
        
        function getReward(self, reward)
            self.N(self.lastAction) = self.N(self.lastAction) + 1; 
            self.S(self.lastAction) = self.S(self.lastAction)  + reward;
            if self.t == self.nextStop
                self.checkpoint();
            end
            self.t = self.t + 1;
        end
        
        function checkpoint(self)
            p = self.S(self.A) ./ self.N(self.A);
            m = min(p); I = find(p == m);
            tokill = floor(1+rand*length(I));
            self.A = self.A(1:length(self.A) ~= I(tokill));
            self.r = self.r + 1;
            self.nextStop = self.steps(self.r);
        end
        
        function J = getRecommendation(self)
            J = self.A(1);
        end
        
    end
end