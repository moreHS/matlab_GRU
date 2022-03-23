function [fc_in,outh1]=RNN_feedforward(input,num_fcuse,coef)
%% model1 Weight Load
% struct2var(coef);

%% 2. RNN Layers
num_hidden = size(coef.w1c,2);
outh1=zeros(1,num_hidden);

fcin=[];
for i=1:length(input)
    outh1=RNN_gate(outh1,input(:,i)',coef.w1g,coef.w1c,coef.b1g,coef.b1c,'tanh');

    if i>length(input)-num_fcuse
        fcin=[fcin; outh1];
    end
end
% fc_in=fcin;

fc_in=[];
for i=1:size(fcin,1)
    for j=1:size(fcin,2)
        for k=1:size(fcin,3)
            fc_in=[fc_in fcin(i,j,k)];
        end
    end
end

%% 3. Fully Layers
num_fc_layer = 2;

for i=1:num_fc_layer
    eval(['temp_weight=coef.w' num2str(i) 'fc;'])
    eval(['temp_bias=coef.b' num2str(i) 'fc;'])
    if i~=num_fc_layer
        eval(['temp_batch_g=coef.batch_g_fc' num2str(i) ';'])
        eval(['temp_batch_b=coef.batch_b_fc' num2str(i) ';'])
        eval(['temp_batch_m=coef.batch_m_fc' num2str(i) ';'])
        eval(['temp_batch_v=coef.batch_v_fc' num2str(i) ';'])

        fc_out=op_fc(fc_in,temp_weight,temp_bias,temp_batch_g,temp_batch_b,temp_batch_m,temp_batch_v);
    else
        fc_out=op_fc(fc_in,temp_weight,temp_bias);
    end
    fc_in=fc_out;
end