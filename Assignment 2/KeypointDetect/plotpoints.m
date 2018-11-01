%/////////////////////////////////////////////////////////////////////////////////////////////
%
% plotpoints - visualize features generated by detect_features
% Usage:  plotpoints(p,img,num_flag)
%
% Parameters:  
%            
%            img :      original image
%            p:         vector of points
%            numflag :  0 - plot with crosshairs    1-plot with number index
%
% Returns:   nothing, generates figure
%
% Author: 
% Scott Ettinger
% scott.m.ettinger@intel.com
%
% May 2002
%/////////////////////////////////////////////////////////////////////////////////////////////

function [] = plotpoints(p,img,num_flag)

if ~exist('num_flag')
    num_flag = 0;
end

figure(gcf)
imagesc(img)
hold on
colormap gray

for i=1:size(p,1)
    x = p(i,1);
    y = p(i,2);
    
       
    if num_flag ~= 1
        % Ahmet: original g+
        plot(x,y,'r.', 'MarkerSize', 15);         %draw box around real feature
    else
        plot(x,y,'r.10', 'MarkerSize', 15); 
        text(x,y,sprintf('%d',i),'color','m');
    end
        
end

hold off;
