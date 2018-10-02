% circle_red_right.m
%{ 
    Creates an animation of a red circle moving from left to right and 
    exports it as a '.avi' file.
    Instructions: run from project root.
%}

clear; 
close all;

% global data
numberOfFrames = 100;
filename = strcat(mfilename, '.avi');
disp(filename)
t = 0:0.01:1;    % Time data, steps of 0.01 from 0 to 1
x = 3*t; % Position data, linear movement
% x = sin(2*pi*t); % Position data, back and forth movement

% Draw figure
figure(1);
set(gcf,'Renderer', 'OpenGL'); 
h = plot(x(1) , 1.5, 'o', 'MarkerSize', 75, 'MarkerFaceColor', 'r');
set(h,'EraseMode', 'normal');

% Axes
xlim([-1,4]);
ylim([-1,4]);

% create AVI object
vidObj = VideoWriter(strcat(pwd, '\animations\output\', filename));
vidObj.Quality = 100;
vidObj.FrameRate = 24;
open(vidObj)

% Animation Loop / movie creation
for i = 1:numberOfFrames
    set(h, 'XData', x(i));
    
    drawnow;
    
    frame = getframe(1);
    M(i) = frame;
    writeVideo(vidObj, frame);
    
    i=i+1;
end

% write video file and open it in new window
close(vidObj)
winopen(strcat(pwd, '\animations\output\', filename))
