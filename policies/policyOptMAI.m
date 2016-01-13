classdef policyOptMAI < ExpPolicy
    % OptMAI for all bandits
    %
    % From Optimal PAC Multiple Arm Identification with Applications to 
    % Crowdsourcing, Zhou, Chen, Li
    
    properties
        t % Number of the round
        N % Number of times each action has been chosen
        S % Cumulated reward with each action
        lastAction % Stores the last action played
        Q % max number of rounds
        A % Active arms
        T % Top arms
        m % number of top arms to pick
        round % current round
        phase % QE, AR ou reset
        phaseStart
        betar % beta^r(1-beta)
        stopping
    end
    
    methods
        function self = policyOptMAI()
        end
        
        function init(self, nbActions, mode, horizon, numArms)
            if ~strcmp(mode, 'budget')
                throw(MException('EXPPOLICY:BadParameter', ...
                    'OptMAI can only be used for fixed budget'));
            end
            self.Q = horizon(1);
            self.m = numArms;
            self.t = 1;
            self.N = zeros(1, nbActions);
            self.S = zeros(1, nbActions);
            self.T = zeros(1, 0);
            self.A = 1:nbActions;
            self.round = 0;
            self.phase = 0;
            self.phaseStart = 0;
            self.stopping = 0;
        end
        
        function action = decision(self)
            if self.phase == 0
                self.betar = (exp(0.2)*0.75) ^self.round * 0.25 * exp(0.2);
                if length(self.T)<self.m && length(self.A)>0
                    if length(self.A)> 4*self.m
                        self.phase = 1; % QE
                        self.T = zeros(1, 0);
                    else
                        self.phase = 2; % AR
                    end
                    self.phaseStart = self.t;
                else
                    self.stopping = 1;
                end
            end
            curt = self.t - self.phaseStart;
            action = self.A(1 + mod(curt, length(self.A)));
            self.lastAction = action;
        end
        
        function getReward(self, reward)
            self.N(self.lastAction) = self.N(self.lastAction) + 1; 
            self.S(self.lastAction) = self.S(self.lastAction) + reward;
            self.t = self.t + 1;
            if self.t - self.phaseStart == floor(self.betar * self.Q/length(self.A))*length(self.A)
                p = self.S(self.A) ./ self.N(self.A);
                sp = sort(p, 2, 'descend');
                switch self.phase
                    case 1 % QE
                        self.A = self.A(p >= sp(ceil(length(sp)/4)));
                        self.phase = 0;
                    case 2 % AR
                        s = length(self.A);
                        K = self.m - length(self.T); % question: K-th largest regarding what set?
                        Delta = max([p-sp(K+1); sp(K)-p]);
                        subS = 1:length(self.A); % correspondance original A -> current A
                        while length(self.T)<K && length(self.A)>3*s/4
                            [~, ia] = max(Delta(subS)); % relative to the current A
                            a = subS(ia); % relative to the original A
                            ta = self.A(ia); % relative to the true set
                            subS = subS(1:length(subS) ~= ia);
                            self.A = self.A(1:length(self.A) ~= ia);
                            if p(a) >= sp(K+1)
                                self.T = [self.T ta];
                            end
                        end
                        self.phase = 0;
                end
            end
        end
        
        function J = getRecommendation(self)
            [~, J] = max(self.N);
        end
        
        function b = fbeta(self, u, t)
            b = sqrt(0.5*log(1.25*length(self.N)*t^4/self.delta)./u);
        end
        
        function r = isConfident(self)
            r = self.stopping;
        end
        
    end
end