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
        
        function [ J, t ] = play(self, policy, mode, horizon, numArms)
            if strcmp(mode, 'budget')
                policy.init(numel(self.means), 'budget', horizon, numArms);
                reward = zeros(1, horizon(1));
                action = zeros(1, horizon(1));
                for t = 1:horizon(1)
                    action(t) = policy.decision();
                    reward(t) = self.reward(action(t));
                    policy.getReward(reward(t));
                    if policy.isConfident()
                        break
                    end
                end
                J = policy.getRecommendation();
            elseif strcmp(mode, 'confidence')
                policy.init(numel(self.means), 'confidence', horizon, numArms);
                t = 0;
                while ~policy.isConfident()
                    t = t + 1;
                    action = policy.decision();
                    reward = self.reward(action);
                    policy.getReward(reward);
                end
                J = policy.getRecommendation();
            else
                throw(MException('EXPGAME:BadParameter', ...
                    ['Uknown mode ' mode]));
            end
        end
        
        function r = reward(self, a)
            r = self.arms{a}.play(); 
        end
    end
end
