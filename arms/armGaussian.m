classdef armGaussian<Arm
    %ARMGAUSSIAN Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods
        function self = armGaussian(mu, sigma)
            self.mean = mu;
            self.var = sigma;
        end
        
        function [reward] = play(self)
            reward = self.mean + self.var * randn(1);
        end
    end
    
end

