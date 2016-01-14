classdef policyUGapE < ExpPolicy
    % UGapE for almost best arm identification, for fixed budget or fixed
    % confidence
    %
    % From Best Arm Identification: A Unified Approach to Fixed Budget 
    % and Fixed Confidence
    % by V. Gabillon, M. Ghavamzadeh, A. Lazaric
    
    properties
        eps
        delta % for fixed confidence
        a = 0.5 % Parameter (==a for budget, ==c for confidence)
        isBudget % budget (1) or confidence (0)
        betas % beta_k(t-1)
        b = 1 % bound on the max arm response
        B % B_k(t)
        J % selected arms
    end
    
    methods
        function self = policyUGapE(a, b)
            if (nargin >= 1)
                self.a = a;
            end
            if (nargin >= 2)
                self.b = b;
            end
        end
        
        function init(self, nbActions, mode, horizon, numArms)
            if numArms > 1
                throw(MException('EXPPOLICY:BadParameter', ...
                    'UGapE can only find the best arm'));
            end
            self.t = 1;
            self.m = numArms;
            self.N = zeros(1, nbActions);
            self.S = zeros(1, nbActions);
            self.J = zeros(0, 1);
            if strcmp(mode, 'budget')
                self.isBudget = 1;
                self.eps = horizon(2);
            else
                self.isBudget = 0;
                self.eps = horizon(1);
                self.delta = horizon(2);
            end
        end
        
        function action = decision(self)
            if self.t <= length(self.N)
                action = self.t;
            else
                if self.isBudget
                    self.betas = self.b * sqrt(self.a ./ self.N);
                else
                    self.betas = self.b * sqrt(self.a * ...
                        log(4*length(self.N)*(self.t-1)^3 / self.delta) ...
                        ./ self.N);
                end
                mu = self.S ./ self.N;
                U = mu + self.betas;
                [~, ou] = sort(U, 2, 'descend');
                L = mu - self.betas;
                mth_item = @(arr, m) arr(m);
                self.B = arrayfun(@(k) U(mth_item(ou(ou ~= k), self.m)) - L(k), ...
                                  1:length(L)); % m-max operator
                [~, ob] = sort(self.B, 2, 'ascend');
                self.J = ob(1:self.m);
                % u_t = argmax_{j\notin J} U_j(t)
                [~, u] = max(U(ob(self.m+1:end)));
                u = ob(u + self.m);
                [~, l] = min(L(self.J));
                l = self.J(l);
                if self.betas(l) > self.betas(u)
                    action = l;
                else
                    action = u;
                end
            end
            self.lastAction = action;
        end
        
        function getReward(self, reward)
            self.N(self.lastAction) = self.N(self.lastAction) + 1; 
            self.S(self.lastAction) = self.S(self.lastAction) + reward;
            self.t = self.t + 1;
        end
        
        function J = getRecommendation(self)
            J = self.J;
        end
        
        function r = isConfident(self)
            if self.isBudget || isempty(self.J)
                r = 0;
            else
                r = min(self.B(self.J)) < self.eps;
            end
        end
        
    end
end