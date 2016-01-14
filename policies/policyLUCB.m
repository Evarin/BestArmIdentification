classdef policyLUCB < ExpPolicy
    % LUCB1 for fixed confidence, multiple almost-best arms identification
    %
    % From PAC Subset Selection in Stochastic MAB
    % by S. Kalyanakrishman, A. Tewari, P. Auer, P. Stone, 2012
    
    properties
        delta % probability of success asked
        eps % precision asked
        r % Current round
        stopping % stopping criterion reached
        toSample % next arm to sample (l)
        high % best sampled arms
    end
    
    methods
        function self = policyLUCB()
        end
        
        function init(self, nbActions, mode, horizon, numArms)
            if ~strcmp(mode, 'confidence')
                throw(MException('EXPPOLICY:BadParameter', ...
                    'LUCB can only be used for fixed confidence'));
            end
            self.eps = horizon(1);
            self.delta = horizon(2);
            self.m = numArms;
            self.t = 1;
            self.r = 1;
            self.k = nbActions;
            self.stopping = 0;
            self.toSample = 0;
            self.high = zeros(1, self.m);
            self.N = zeros(1, nbActions);
            self.S = zeros(1, nbActions);
        end
        
        function action = decision(self)
            if self.t <= self.k
                action = self.t;
            elseif self.toSample == 0
                p = self.S ./ self.N;
                [~, sp] = sort(p, 2, 'descend');
                self.high = sp(1:self.m);
                low = sp(self.m+1:end);
                betas = self.fbeta(self.N, self.r);
                
                % Confidence bounds
                [phs, hs] = min(p(self.high) - betas(self.high));
                h = self.high(hs);
                [pls, ls] = max(p(low) + betas(low));
                l = low(ls);
                self.stopping = pls - phs < self.eps;
                
                % Actions
                self.toSample = l;
                action = h;
                self.r = self.r + 1;
            else
                action = self.toSample;
                self.toSample = 0;
            end
            self.lastAction = action;
        end
        
        function getReward(self, reward)
            self.N(self.lastAction) = self.N(self.lastAction) + 1; 
            self.S(self.lastAction) = self.S(self.lastAction) + reward;
            self.t = self.t + 1;
        end
        
        function J = getRecommendation(self)
            J = self.high;
        end
        
        function b = fbeta(self, u, t)
            b = sqrt(0.5*log(1.25*self.k*t^4/self.delta)./u);
        end
        
        function r = isConfident(self)
            r = self.stopping;
        end
        
    end
end