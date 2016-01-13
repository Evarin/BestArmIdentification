classdef policyExpGap < ExpPolicy
    % Exponential-Gap policy for any bandit
    %
    % From Almost Optimal Exploration in Multi-Armed Bandits
    % Karnin, Koren, Somekh
    
    properties
        lastAction % Stores the last action played
        N % Number of times each action has been chosen
        S % Cumulated reward with each action
        t
        eps
        delta0
        delta
        m
        l % number of attempts
        A
        r
        phase
        tr
        Ar
    end
    
    methods
        function self = policyExpGap()
        end
        
        function init(self, nbActions, mode, horizon, numArms)
            if ~strcmp(mode, 'confidence')
                throw(MException('EXPPOLICY:BadParameter', ...
                    'Exponential-Gap policy can only be used for fixed confidence'));
            end
            if numArms > 1
                throw(MException('EXPPOLICY:BadParameter', ...
                    'Exponential-Gap can only find the best arm'));
            end
            self.delta0 = horizon(2);
            self.A = 1:nbActions;
            self.r = 1;
            self.phase = 0;
            self.N = zeros(1, nbActions);
            self.S = zeros(1, nbActions);
            self.t = 1;
        end
        
        function action = decision(self)
            if self.phase == 0
                er = 2^(-self.r)/4;
                dr = self.delta0/(50*self.r^3);
                self.tr = self.t + length(self.A) * round(2/er^2 * log(2/dr))-1;
                self.phase = 1;
            end
            action = mod(self.t, length(self.A)) + 1;
            self.lastAction = action;
        end
        
        function getReward(self, reward)
            self.N(self.lastAction) = self.N(self.lastAction) + 1; 
            self.S(self.lastAction) = self.S(self.lastAction) + reward;
            if self.t == self.tr
                p = self.S ./ self.N;
                switch self.phase
                    case 1 % Start ME
                        self.Ar = self.A(:);
                        self.eps = 2^(-self.r)/4 /8; % for ME
                        self.delta = self.delta0 /(50*self.r^3)/2;
                        self.phase = 2;
                    case 2 % normal ME
                        sp = sort(p(self.A));
                        med = sp(ceil(length(p)/2));
                        self.A = self.A(p(self.A)>=med);
                        self.lastA = self.t;
                        self.l = self.l + 1;
                        self.eps = self.eps * 0.75;
                        self.delta = self.delta / 2;
                        if length(self.A) == 1
                            self.phase = 0;
                        end
                end
                switch self.phase
                    case 2
                        self.tr = self.t + length(self.A) * round(1/(self.eps/2)^2*log(3/self.delta));
                    case 0
                        er = 2^(-self.r)/4;
                        self.A = self.Ar(self.p(self.Ar) >= self.p(self.A) - er);
                        self.r = self.r + 1;
                end
            end
            self.t = self.t + 1;
        end
        
        function J = getRecommendation(self)
            [~, J] = self.A;
        end
        
        function r = isConfident(self)
            r = length(self.A) == 1;
        end
        
    end
end