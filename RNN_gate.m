function [ht]=RNN_gate(h,x,Wg,Wc,Bg,Bc,act)

% h : 전단 출력, 1x30
% x : 입력, 1x1
% Wg : update gate weight, 31x60, (z와 r의 concatenate)
% Wc : hidden gate candinate weight, 31x30
% act : Activation function

num_hidden=size(Wc,2);

% in1=[h x];
in1=[x h];

% Zt=sigmoid(in1*Wg(:,1:num_hidden)+Bg(:,1:num_hidden));
% Rt=sigmoid(in1*Wg(:,num_hidden+1:end)+Bg(:,num_hidden+1:end));
Rt=sigmoid(in1*Wg(:,1:num_hidden)+Bg(:,1:num_hidden));
Zt=sigmoid(in1*Wg(:,num_hidden+1:end)+Bg(:,num_hidden+1:end));

% in2=[Rt.*h x];
in2=[x Rt.*h];
if strcmp(act,'tanh')
    Ht=tanh(in2*Wc+Bc);
elseif strcmp(act,'relu')
    Ht=max(in2*Wc+Bc,0);
else
    error('no activation')
end

% ht=(1-Zt).*h+Zt.*Ht;
ht=(1-Zt).*Ht+Zt.*h;
end

function [y]=sigmoid(x)

y=1./(1+exp(-x));
end

