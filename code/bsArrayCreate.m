function bsInd = bsArrayCreate(burnIn,M,skip)
% function bsInd = bsArrayCreate(burnIn,M,skip)
%
% This generates an array of bsInd values, ones for each of the total draws in the
% whole MCMC run, so that we can save the values of the backward-sampled states only in
% situations when we need to keep 

% Build the total draws
tD = burnIn + M*skip;

% Build an array of all zeros to start
bsInd = zeros(tD,1);
allOnes = ones(tD,1);

% Plug in zeros wherever we don't need to have a state path draw, one wherever we do.
bsInd(burnIn+1:skip:end,1) = allOnes(burnIn+1:skip:end,1);

