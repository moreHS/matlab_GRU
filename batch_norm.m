%% Batch-normalization
%% 2018.01.04 ByeongKeun Kang

%input: data_point by filter number
%gamma: batch-normalization gamma value
%beta: batch-normalization beta value
%batch_m: batch-normalization population mean
%batch_v: batch-normalization population variance
function batch_conv=batch_norm(conv_out,gamma,beta,batch_m,batch_v)
    esp=0.001; %epsilon

    %Subtract mean vector
    xmu=conv_out-repmat(batch_m,size(conv_out,1),1);
    
    %Variance(add epsilon for numerical stability)
    conv_sqrtvar=sqrt(batch_v+esp);
    
    %Invert sqrtvar
    ivar=1./conv_sqrtvar;
    
    %Normalize
    xhat=xmu.*repmat(ivar,size(xmu,1),1);
    
    %Transformation
    batch_conv=repmat(gamma,size(xhat,1),1).*xhat+repmat(beta,size(xhat,1),1);
end