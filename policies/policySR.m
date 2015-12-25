classdef policySR < ExpPolicy
    % Successive rejects for any bandit
    %
    % From Best Arm Identification in Multi-Armed Bandits, J-Y Audibert...
    
    properties
        t % Number of the round
        N % Number of times each action has been chosen
        S % Cumulated reward with each action
        A % Actives arms
        steps % n_0...(K-1)
        curstep % current step
        lastAction % Stores the last action played
    end
    
    methods
        function self = policySR()
        end
        
        function init(self, nbActions, mode, horizon)
            if ~strcmp(mode, 'budget')
                throw(MException('EXPPOLICY:BadParameter', ...
                    'SR can only be used for fixed budget'));
            end
            self.A = 1:nbActions;
            logK = 0.5 + sum(1./(2:nbActions)); 
            nk = ceil(((horizon-nbActions)/logK) ...
                ./ (nbActions - (0:(nbActions-2))) );
            dnk = nk - [0 nk(1:end-1)];
            self.steps = [0 cumsum(dnk.*(nbActions:-1:2)) horizon+1];
            self.curstep = 2;
            self.t = 1;
            self.N = zeros(1, nbActions);
            self.S = zeros(1, nbActions);
        end
        
        function action = decision(self)
            self.lastAction = 1+mod(self.t,length(self.A));
            action = self.A(self.lastAction);
        end
        
        function getReward(self, reward)
            self.N(self.lastAction) = self.N(self.lastAction) + 1; 
            self.S(self.lastAction) = self.S(self.lastAction)  + reward;
            if self.t == self.steps(self.curstep)
                rews = self.S./self.N;
                m = min(rews); I = find(rews == m);
                tokill = floor(1+rand*length(I));
                tokeep = (1:length(self.A) ~= tokill);
                self.N = self.N(tokeep);
                self.A = self.A(tokeep);
                self.S = self.S(tokeep);
                self.curstep = self.curstep + 1;
            end
            self.t = self.t + 1;
        end
        
        function J = getRecommendation(self)
            J = self.A(1);
        end
        
    end
end