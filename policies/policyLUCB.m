classdef policyLUCB < ExpPolicy
    % LUCB1 for Bernouilli Bandits (only?)
    %
    % From PAC Subset Selection in Stochastic MAB, Kalyanakrishman, Tewari,
    % Auer, Stone, 2012
    
    properties
        t % Number of the round
        N % Number of times each action has been chosen
        S % Cumulated reward with each action
        lastAction % Stores the last action played
        delta
        eps
        m
        round % Current round
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
            self.round = 1;
            self.N = zeros(1, nbActions);
            self.S = zeros(1, nbActions);
            self.stopping = 0;
            self.toSample = 0;
            self.high = zeros(1, self.m);
        end
        
        function action = decision(self)
            if self.t <= length(self.N)
                action = self.t;
            elseif self.toSample == 0
                p = self.S ./ self.N;
                [~, ordre] = sort(p, 2, 'descend');
                self.high = ordre(1:self.m);
                low = ordre(self.m+1:end);
                betas = self.fbeta(self.N, self.round);
                
                % Confidence bounds
                [phs, hs] = min(p(self.high) - betas(self.high));
                h = self.high(hs);
                [pls, ls] = max(p(low) + betas(low));
                l = low(ls);
                self.stopping = pls - phs < self.eps;
                
                % Actions
                self.toSample = l;
                action = h;
                self.round = self.round + 1;
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
            b = sqrt(0.5*log(1.25*length(self.N)*t^4/self.delta)./u);
        end
        
        function r = isConfident(self)
            r = self.stopping;
        end
        
    end
end