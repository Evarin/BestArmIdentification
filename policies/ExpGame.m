classdef ExpGame<handle
    % Generic bandit game interface for pure exploration
    
    properties
        nbActions % number of actions available
        arms % expectations of arms
        tabR % internal: array of all rewards
        N % internal: counters for rewards
        means % expectations of bandits
    end
    
    methods
        function self = ExpGame(arms)
            self.arms = arms;
            self.nbActions = length(self.arms);
            self.means = zeros(1, self.nbActions);
            for i=1:self.nbActions
                self.means(i) = arms{i}.mean;
            end
        end
        
        function [ J ] = play(self, policy, mode, horizon)
            if ~strcmp(mode, 'budget')
                throw(MException('EXPPOLICY:BadParameter', ...
                    'UCB-E can only be used for fixed budget'));
                policy.init(self.initRewards(horizon), horizon);
                reward = zeros(1, horizon);
                action = zeros(1, horizon);
                for t = 1:horizon
                    action(t) = policy.decision();
                    reward(t) = self.reward(action(t));
                    policy.getReward(reward(t));
                end
                J = policy.getRecommendation();
            else
                throw(MException('EXPGAME:NotImplemented', ...
                    'Only fixed budget setup is implemented'));
            end
        end
        
        function K = initRewards(self,n)
            % initiates the reward process, and returns the number of
            % actions
            K = length(self.arms);
            self.tabR = zeros(K, n);
            for t=1:n
                for a=1:K
                    self.tabR(a, t) = self.arms{a}.play();
                end
            end
            self.N = zeros(1,K);
        end
        
        function r = reward(self, a)
            self.N(a) = self.N(a) + 1;
            r = self.tabR(a, self.N(a));            
        end
    end
end
