function [out, feat]=main(input)
%% Input reshape


%% RNN ����

    mo1=load('weight.mat');

    [out,feat]=RNN_feedforward(input,4,mo1);
    
end