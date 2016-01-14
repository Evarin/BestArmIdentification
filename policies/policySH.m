classdef policySH < ExpPolicy
    % Sequential Halving for fixed budget and best arm identification
    %
    % From Almost Optimal Exploration in Multi-Armed Bandits
    % by Z. Karnin, T. Koren, O. Somekh
    
    properties
        a = 1 % Parameter
        A % current subset of arms
        T % maximum number of pulls
        nextStop % next t to call checkpoint
    end
    
    methods
        function self = policySH(a)
            if (nargin >= 1)
                self.a = a;
            end
        end
        
        function init(self, nbActions, mode, horizon, numArms)
            if ~strcmp(mode, 'budget')
                throw(MException('EXPPOLICY:BadParameter', ...
                    'SH can only be used for fixed budget'));
            end
            if numArms > 1
                throw(MException('EXPPOLICY:BadParameter', ...
                    'SH can only find the best arm'));
            end
            self.t = 1;
            self.T = horizon(1);
            self.nextStop = nbActions * ...
                floor(self.T / nbActions / ceil(log2(nbActions)));
            self.A = 1:nbActions;
            self.N = zeros(1, nbActions);
            self.S = zeros(1, nbActions);
        end
        
        function action = decision(self)
            action = self.A(mod(self.t, length(self.A)) + 1);
            self.lastAction = action;
        end
        
        function getReward(self, reward)
            self.N(self.lastAction) = self.N(self.lastAction) + 1; 
            self.S(self.lastAction) = self.S(self.lastAction) + reward;
            if self.t == self.nextStop
                p = self.S ./ self.N;
                [~, sp] = sort(p(self.A), 2, 'descend');
                self.A = self.A(sp(1:ceil(length(sp)/2)));
                self.nextStop = self.t + length(self.A) * ...
                    floor(self.T / length(self.A) / ceil(log2(length(self.N))));
            end
            self.t = self.t + 1;
        end
        
        function J = getRecommendation(self)
            [~, J] = max(self.S./self.N);
        end
        
        function r = isConfident(self)
            r = length(self.A) == 1;
        end
    end
end