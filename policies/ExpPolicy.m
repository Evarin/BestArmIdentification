classdef ExpPolicy<handle
    % Generic policy interface for the pure exploration problem
    
    properties
        t % number of pulls
        lastAction % last arm pulled
        S % sum of rewards
        N % number of pulls for each arm
        k % number of arms
        m % number of arms asked
    end
    
    methods
        function init(self, nbActions, horizon, numArms), end % to be called before a new game
        function a = decision(self), end % chooses the next action
        function getReward(self, reward), end % update after new observation
        function J = getRecommendation(self), end % Outputs the final recommandation
        function r = isConfident(self) % Fixed confidence setting : is it converged
            r = 0;
        end
    end
    
end
