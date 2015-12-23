classdef ExpPolicy<handle
    % Generic policy interface for the pure exploration problem
    
    properties
    end
    
    methods
        function init(self, nbActions, horizon), end % to be called before a new game
        function a = decision(self), end % chooses the next action
        function getReward(self, reward), end % update after new observation
        function J = getRecommendation(self), end % Outputs the final recommandation
        function r = isConfident(self) % Fixed confidence setting : is it converged
            r = 1;
        end
    end
    
end
