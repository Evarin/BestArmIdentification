classdef armExp<Arm
    % arm with trucated exponential distribution
    
    properties
        lambda % parameter of the exponential distribution
    end
    
    methods
        function self = armExp(lambda)
            self.lambda=lambda;
            self.mean = (1/lambda)*(1-exp(-lambda));
            self.var = 1; % compute it yourself !
        end
        
        function [reward] = play(self)
            reward = min(-1/self.lambda*log(rand),1);
        end
                
    end    
end