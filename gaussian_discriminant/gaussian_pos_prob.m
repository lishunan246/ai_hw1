function p = gaussian_pos_prob(X, Mu, Sigma, Phi)
%GAUSSIAN_POS_PROB Posterior probability of GDA.
%   p = GAUSSIAN_POS_PROB(X, Mu, Sigma) compute the posterior probability
%   of given N data points X using Gaussian Discriminant Analysis where the
%   K gaussian distributions are specified by Mu, Sigma and Phi.
%
%   Inputs:
%       'X'     - M-by-N matrix, N data points of dimension M.
%       'Mu'    - M-by-K matrix, mean of K Gaussian distributions.
%       'Sigma' - M-by-M-by-K matrix (yes, a 3D matrix), variance matrix of
%                   K Gaussian distributions.
%       'Phi'   - 1-by-K matrix, prior of K Gaussian distributions.
%
%   Outputs:
%       'p'     - N-by-K matrix, posterior probability of N data points
%                   with in K Gaussian distributions.

N = size(X, 2);
K = length(Phi);
P = zeros(N, K);

% Your code HERE
M=size(X,1);
l=zeros(N,K);
px=l;
for k=1:K
    SigmaK=Sigma(:,:,k);
    t=Mu(:,k);
    MuK=repmat( t,1,N);
    
   % l(:,k)=1/(2*pi*sqrt(det(SigmaK)))*exp(-0.5*(X-MuK)'*inv(SigmaK)*(X-MuK));
   l(:,k)=mvnpdf(X(:,k),t,SigmaK);
end


px=l*repmat(Phi,N,1);
p=l.*repmat(Phi,N,1)./px;


