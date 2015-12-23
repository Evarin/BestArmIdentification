classdef armBeta<Arm
    % arm having a Beta distribution
    
    properties
        a % first parameter
        b % second parameter
    end
    
    methods
        function self = armBeta(a,b)
            self.a=a; 
            self.b = b;
            self.mean = a/(a+b);
            self.var = (a*b)/((a+b)^2*(a+b+1));
        end
        
        function [reward] = play(self)
            reward = betarnd(self.a,self.b);
        end
                
    end    
end