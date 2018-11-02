%% This script helps you to debug your implementation of SIFTDescriptor.m.

%% Clear all
clc; close all; clear all;

%% Add path
addpath('KeypointDetect');

%% Load image
% original yard1
img = imread('./data/yosemite2.jpg');
figure;
imagesc(img);

%% Detect keypoints
[feature, ~, imp] = detect_features(img);

%% Build descriptors
descriptors = SIFTDescriptor(imp, feature(:,8:9), feature(:,3));

%% Ahmet
%disp(size(feature))
%disp(size(descriptors))
%disp(feature)
%plotpoints(feature,img) 

%% Visualize n descriptors
n = 80;
hold on;
PlotSIFTDescriptor(descriptors(1:n,:)', feature(1:n,1:3)');
hold off;