classdef policyOptMAI < ExpPolicy
    % OptMAI for fixed budget and multiple best arms identification
    %
    % From Optimal PAC Multiple Arm Identification with Applications to 
    % Crowdsourcing
    % by Y. Zhou, X. Chen, J. Li
    
    properties
        Q % max number of rounds
        A % current subset of arms
        T % Top arms
        m % number of top arms to pick
        r % current r
        phase % reset, QE, or AR
        nextStop % next t to call checkpoint
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
            self.N = zeros(1, nbActions);
            self.S = zeros(1, nbActions);
            self.T = zeros(1, 0);
            self.A = 1:nbActions;
            self.r = 0;
            self.phase = 0;
            self.stopping = 0;
            self.t = 0;
            self.checkpoint();
            self.t = 1;
        end
        
        function action = decision(self)
            action = self.A(mod(self.t, length(self.A)) + 1);
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
            if self.phase == 0 % reset
                self.betar = (exp(0.2)*0.75) ^self.r * 0.25 * exp(0.2);
                if length(self.T) < self.m && ~isempty(self.A)
                    if length(self.A)> 4*self.m
                        self.phase = 1; % QE
                        self.T = zeros(1, 0);
                    else
                        self.phase = 2; % AR
                    end
                else
                    self.stopping = 1;
                end
            end
            self.nextStop = self.t + length(self.A) * ...
                floor(self.betar * self.Q/length(self.A));
        end
        
        function J = getRecommendation(self)
            [~, J] = max(self.N);
        end
        
        function r = isConfident(self)
            r = self.stopping;
        end
        
    end
end