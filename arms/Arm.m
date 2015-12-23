classdef Arm
    % Generic class for bandits arms
    
    properties
        mean % expectation of the arm
        var % variance
    end
    
    methods
        function reward = play(self), end
    end
    
end

