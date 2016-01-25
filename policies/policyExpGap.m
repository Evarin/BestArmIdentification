classdef policyExpGap < ExpPolicy
    % Exponential-Gap policy for fixed confidence, best arm identification
    %
    % From Almost Optimal Exploration in Multi-Armed Bandits
    % by Z. Karnin, T. Koren, O. Somekh
    
    properties
        delta % probability of success asked
        nextStop % next t to call checkpoint
        A % current subset of arms
        r % current round
        er % epsilon_r
        dr % delta_r
        phase % phase (itself or ME)
        MEpolicy % ME's instance
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
            self.delta = horizon(2);
            self.A = 1:nbActions;
            self.r = 1;
            self.phase = 0;
            self.N = zeros(1, nbActions);
            self.S = zeros(1, nbActions);
            self.t = 0;
            self.checkpoint();
            self.t = 1;
        end
        
        function action = decision(self)
            switch self.phase
                % TODO Could be rewritten so it uses previous pulls
                case 1 % itself's procedure
                    action = self.A(mod(self.t, length(self.A)) + 1);
                case 2 % ME
                    action = self.A(self.MEpolicy.decision());
            end
            self.lastAction = action;
        end
        
        function getReward(self, reward)
            switch self.phase
                case 1
                    self.N(self.lastAction) = self.N(self.lastAction) + 1; 
                    self.S(self.lastAction) = self.S(self.lastAction) + reward;
                    if self.t == self.nextStop
                        self.checkpoint();
                    end
                case 2
                    self.MEpolicy.getReward(reward);
                    if self.MEpolicy.isConfident()
                        self.checkpoint();
                    end
            end
            self.t = self.t + 1;
        end
        
        function checkpoint(self)
            p = self.S ./ self.N;
            switch self.phase
                case 1 % Start ME
                    self.MEpolicy = policyME;
                    self.MEpolicy.init(length(self.A), 'confidence', ...
                        [self.er/2, self.dr], 1)
                    self.phase = 2;
                case 2 % ME ended
                    ba = self.MEpolicy.getRecommendation();
                    self.A = self.A(p(self.A) >= p(self.A(ba)) - self.er);
                    self.phase = 0;
            end
            if self.phase == 0 % Restarting the loop
                self.er = 2^(-self.r)/4;
                self.dr = self.delta / (50*self.r^3);
                self.nextStop = self.t + length(self.A) * round(2/self.er^2 * log(2/self.dr));
                self.r = self.r + 1;
                self.phase = 1;
            end
        end
        
        function J = getRecommendation(self)
            J = self.A;
        end
        
        function r = isConfident(self)
            r = length(self.A) == 1;
        end
        
    end
end