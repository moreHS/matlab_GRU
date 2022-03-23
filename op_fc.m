%% Operation of Fully-Connected layers
%% 2018.04.18 ByeongKeun Kang

%input: convolutional layers output (1 by features)
%weight: weights (previous features by current features)
%bias: 1 by feature_map (1 by current features)
%batch_g: batch-normalization gamma value
%batch_b: batch-normalization beta value
%batch_m: batch-normalization population mean
%batch_v: batch-normalization population variance

function [layer_output]=op_fc(input,weight,bias,batch_g,batch_b,batch_m,batch_v)
    if nargin>3
        %Fully_connected Matrix calculation
        layer_output=input*weight+bias;
        %Batch-normalization
        layer_output=batch_norm(layer_output,batch_g,batch_b,batch_m,batch_v);
        %Activation function: ReLU
        layer_output=max(layer_output,0);
    else
        %Fully_connected Matrix calculation
        layer_output=input*weight+bias;
    end
end