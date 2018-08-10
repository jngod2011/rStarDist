function estimateModel(job_id_in, Worker_id_in)
% 
% The code can be run as a script to generate the estimates used in the latest public version of
% the paper, but the majority of the work was conducted using a scheduler on a cluster and that
% functionality has been preserved in this code.  Running the code in MATLAB with no inputs
% should work fine and generate the estimates used in the latest version of the paper.
%
% Kurt Lewis
% August, 2018

% Determine if this is a SLURM-style run (with worker and job IDs) or a stand-alone where we
% generate fake versions associated with the 
switch nargin
    
  case 2
    % This should only be used if ou are using a SLURM-style generator to combine multiple chains
    job_id    = str2double(job_id_in);
    Worker_id = str2double(Worker_id_in);
 
  case 0
    
    job_id = str2double(datestr(now,'yyyymmdd'));
    Worker_id = str2double(datestr(now,'HHMMSS'));
    
  otherwise
    error('The number of inputs should be 2 (Job and Worker ID) or 0 (not a SLURM run).')
    
end

if (isdeployed == false)
    fprintf('\nRunning interactively with addpath and recursive paths as normal.\n')
end

%% The percentiles for the figures
hiP = 95;
lowP = 5;

%% This is where outputs should be delivered.  Data outputs, things like the rStar and bounds estimates
outDataDir = '../data/output';

%% Switch for baseline vs alternative model
baseAlt = 1 % 0 = base, 1 = alt

%% SETTING A SEED FOR THE RNG
% Generally, if run on a single machine, it will use the dates to build the rng seed, otherwise it will use SLURM worker
% IDs if run using a job array (this organization is performed above by setting the Worker_id to
% the time when using the single machine run above).  Replication of results in WP version
% current with the first release of the updated paper (August 2018) is accomplished with seed 1234567.

cede = 1234567
rng(cede)

%% If using on a SLURM_style scheduler, 
%rng(Worker_id,'twister');

%% MAT file output.
%% This cannot be a relative reference, it must be direct because the file generated
%% will be multiple GBs and we can't risk dropping it into a home or production folder
%% where it could cause the system to seize.
if ispc
    matOutDir = 'C:/scratch/';
else
    matOutDir = '/scratch/';
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% This block controls whether it is the baseline or the alternative specification being
%% estimated 

%% File names and reporting information, Alternative Specification
if baseAlt == 1
    nameFile = ['AltModel_1234567_' num2str(job_id) '_' num2str(Worker_id) '.mat'];
    shortDes = 'AltModelForDistribution'
    estLoc = [1 2 3 4 5 6 7 8 9 10 13]
elseif baseAlt == 0
    %%% File names and reporting information, Baseline Specification
    nameFile = ['BaseModel_1234567_' num2str(job_id) '_' num2str(Worker_id) '.mat'];
    shortDes = 'BaseModelForDistribution'
    estLoc = [1 2 3 4 5 6 7 8 9 10]
else 
    error('Set which model is being estimated via baseAlt switch');
end


% The mat file that saves all the output
sf = [matOutDir,nameFile]

%% Run Structures
%% Run on a single machine
runInfo = ['250K burn in, 25 skip with 10000 total draws, meant for SINGLE MACHINE RUN']
burnIn = 250000
M = 10000
skip = 25

%%% Run via SLURM with 20 chains, 10K final output via aggregation
%runInfo = ['250K burn in, 10 skip with 500 total draws, meant for SLURM WITH 20 CHAINS']
%burnIn = 250000
%M = 500
%skip = 25

%%% for directory checking, diagnostics, etc.
%runInfo = ['250 burn in, 1 skip with 500 total draws, meant for TESTING']
%burnIn = 250
%M = 250
%skip = 1

totDraw = burnIn + (M*skip)

% The step size parameter for the MH algorithm when everything takes the same size step
% on average.  We will have random step sizes that average this size. 
stepSize = 0.02^2;

% How often (meaning after how many steps) to report information from the algorithm in the output
modShow = 500;

%% Data
if (isdeployed == false)
    [allData,hdr] = xlsread('../data/input/LVGdata20170807_distribute.xls');
else 
    [allData,hdr] = xlsread('LVGdata20170807_distribute.xls');
end

% We have loaded in the full data file, and we need to get the information that will
% allow us to initialize the state variables.  This information is essentially the
% trend component of the HP filter of the lgdp data starting from the beginning of the
% estimation sample.

startInd = datefind(datenum(1961,03,31),x2mdate(allData(:,1)),1)

% The state initialization will begin using the calendar year before the estimation
% starts, so I need to get the trend component from the HP filter for 1960Q2-Q4, along
% with the growth rate of that trend (as measured by the first difference since this is
% log data).  
hpStart = startInd - 4;
gdpHP = allData(hpStart:end,2)*100;

% Get the trend and the cycle using quarterly settings for the HP filter, which is a
% smoothing parameter of 36000, replicating LW.
[Tr,Cy] = hpfilter(gdpHP,36000);
gHP = Tr(2:end) - Tr(1:end-1);
yPinit = Tr(4:-1:2);
gInit = gHP(3:-1:2);
zInit = zeros(2,1);

% So that we are careful to nest their structure, use the same Initial conditions for the states that HLW use 
m0hp = [yPinit; gInit; zInit]

V0hp = [0.7  0.2  0.0 0.2  0.2 0.0  0.0;...
        0.2  0.2  0.0 0.0  0.0 0.0  0.0;...
        0.0  0.0  0.2 0.0  0.0 0.0  0.0;...
        0.2  0.0  0.0 0.2  0.2 0.0  0.0;...
        0.2  0.0  0.0 0.2  0.2 0.0  0.0;...
        0.0  0.0  0.0 0.0  0.0 0.2  0.2;...
        0.0  0.0  0.0 0.0  0.0 0.2  0.2]


%%% This is all the data
dates = x2mdate(allData(startInd:end,1));
gdp = allData(startInd:end,2)*100;
infl = allData(startInd:end,3); 
gdpl1 = allData(startInd:end,4)*100;
gdpl2 = allData(startInd:end,5)*100;
infll1 = allData(startInd:end,6);
infll2 = allData(startInd:end,7);
infll3 = allData(startInd:end,8);
infll4 = allData(startInd:end,9);
rl1 = allData(startInd:end,10);
rl2 = allData(startInd:end,11);

% Extra series enabled to test additional exogenous variables, set to zero when not in use.
xSeries = zeros(size(gdp));

% Organize the data into matrices
y = [gdp';infl'];
u = [ones(size(gdpl1'));gdpl1';gdpl2';rl1';rl2';infll1'; (1/3)*(infll2+infll3+infll4)'; ...
         xSeries'];

%% Get some features of the data sorted out:
[m,T] = size(y)
[n,T1] = size(u)

%% Throw errors if the data isn't matching int erms of number of time periods
if (T~=T1)
    error('The number of observations for y and u are not the same')
end

%% Theta structure
%% Place the HLW estimates here for comparison and, in some applications initialization
%% These are the values in the August 2016 FRBSF WP code base, not the version of Table 1 of HLW

a1pa2 = 0.9421956; % Alpha_1 + alpha_2
a2 = 0;  % alpha_2 unknown, set to zero for now 
a1 = a1pa2 - a2;
ar = -0.07056424;  % a_r
bY = 0.07842287;   % b_y
b1 = 0.6715626;    % b_pi
s1 = 0.3545614;    % sig_ytilde in HLW
s2 = 0.79564918;   % sig_pi in HLW
s3 = 0.154233;     % sig_z in HLW 
s4 = 0.5802112;    % sig_ystar in HLW
s5 = 0.120119/4;   % sig_g in HLW
aX = 0;            % Extra exogenous variable potential coefficient can be set as needed

rhoG = 1;          % Picking 1 for both rhos, so that if we choose not to estimate them (removing estLoc 12 and 13)

% Random walk version of rhoZ
rhoZ = 1;          % they are just the usual random walks.  The initial draw is determined by prior draw each time.

% IID version of rhoZ
%rhoZ = 0;          % they are just the usual random walks.  The initial draw is determined by prior draw each time.

muZ = 0;           % Again, using 0 as the default value so that just not estimating it
                   % drops it from the model

muG = 0;           % Again, using 0 as the default value so that just not estimating it
                   % drops it from the model

% Lambda values from HLW
lambG = 0.05175658; 
lambZ = 0.03069520;

thetaHLW = [a1 a2 ar b1 bY s1 s2 s3 s4 s5 aX rhoG rhoZ muG muZ];

%% Keep the names of the variables
varNames = {'a1', 'a2', 'ar', 'b1', 'bY', 's1', 's2', 's3', 's4', 's5', 'aX', 'rhoG', ...
            'rhoZ', 'muG', 'muZ'};
varNamesLatex = {'$a_1$', '$a_2$', '$a_r$', '$b_1$', '$b_Y$', '$s_1$', '$s_2$',...
                 '$s_3$', '$s_4$', '$s_5$', '$a_X$', '$\rho_g$', ...
                 '$\rho_z$', '$\mu_g$', '$\mu_z$'};

% Construct the parameters based on the real info
[Ahlw,Bhlw,Chlw,Dhlw,Fhlw,Ghlw,Shlw] = paramMat_distribute(thetaHLW,y,u)

% A flag for using Carter-Kohn Forward Filter, Backward Sample for the states
bsIndOverall = 1;
bsInd = bsArrayCreate(burnIn,M,skip);

% Setting the parameters which will be estimated (estLoc) and those that will take assumed values (realLoc)    
realLoc = setdiff(1:15,estLoc);

% Draw from the prior
pDraw = priorDraw_distribute;

%% Initiate the collection of theta draws, and the step shock proerties
tDraws = NaN(totDraw,length(pDraw));
shkMu = zeros(size(pDraw));
shkSig = diag([1 1 1 1 1 .5 .5 500 .5 .25 1 1 100 1 1]);

% Use the HLW values
theta0 = thetaHLW;

% Except for the places we are NOT trying to estimate, here we put in the prior
theta0(estLoc) = pDraw(estLoc);

% If we aren't directly estimating a1, then it is implied by the estimate of a2 and
% the pre-defined value of a1pa2 from HLW Table 1
if (ismember(1,estLoc)==0)
    theta0(1) = a1pa2 - theta0(2);
end

% If we aren't directly estimating sigma_g (s5), then it is implied by the estimate of
% sigma_ystar (s4) and lambda_G
if (ismember(10,estLoc)==0)
    theta0(10) = lambG*theta0(9);
end

% If we aren't directly estimating sigma_z (s3), then it is implied by the estimate of
% sigma_ytilde (s1), ar and lambda_z
if (ismember(8,estLoc)==0)
    theta0(8) = -lambZ*theta0(6)/theta0(3);
end

% Now, evaluate the prior for this theta
Lpi0 = priorEval_distribute(theta0,estLoc)

% Construct the state space for the given parameters
[A,B,C,D,F,G,S] = paramMat_distribute(theta0,y,u)

% Based on the model, we need to get the size of the state variable
q = size(A,1);

%% Initiate the collection of the state variables and the likelihood value that corresponds
sDraws = NaN(q,T,totDraw);
mDraws = NaN(q,T,totDraw);
LLdraws = NaN(totDraw,1);
LpiDraws = NaN(totDraw,1);

% Evaluate the model with ffbs
[LL0,LLt,StDraw0,m,V] = ffbs_kflfvg(y,u,A,B,C,D,F,G,S,m0hp,V0hp,1);

% Get the baseline R value
R0 = Lpi0 + LL0;

%% The first iteration of the MH loop
Rold = R0;
tOld = theta0;
sOld = StDraw0;
mOld = m;
VOld = V;
LLold = LL0;
LpiOld = Lpi0;
AR = NaN(totDraw,1);

LpiProp = -Inf;
while (LpiProp == -Inf)
    
    % Create the proposal, we aren't going far from the prior here because we know it
    % will be within the range where the priorEval won't return -Inf.
    tProp = tOld + mvnrnd(shkMu,1e-5*stepSize*shkSig);
    
    % Only estimate the parameters we ask to estimate (for debugging)
    tProp(realLoc) = thetaHLW(realLoc);
    
    % If we aren't directly estimating a1, then it is implied by the estimate of a2 and
    % the pre-defined value of a1pa2 from HLW Table 1
    if (ismember(1,estLoc)==0)
        tProp(1) = a1pa2 - tProp(2);
    end
    
    % If we aren't directly estimating sigma_g (s5), then it is implied by the estimate of
    % sigma_ystar (s4) and lambda_G
    if (ismember(10,estLoc)==0)
        tProp(10) = lambG*tProp(9);
    end
     
    % If we aren't directly estimating sigma_z (s3), then it is implied by the estimate of
    % sigma_ytilde (s1), ar and lambda_z
    if (ismember(8,estLoc)==0)
        tProp(8) = -lambZ*tProp(6)/tProp(3);
    end
        
    % Get the log-prior evaluation, it may be neg. infinity
    LpiProp = priorEval_distribute(tProp,estLoc);
    
end

% Construct the state space structures for the given parameters
[A,B,C,D,F,G,S] = paramMat_distribute(tProp,y,u);

% Get the loglikelihood of the proposed parameter vector
[LLProp,LLtProp,StDrawProp,mProp,VProp] = ffbs_kflfvg(y,u,A,B,C,D,F,G,S,m0hp,V0hp,1);

% Assemble the proposed statistic
Rprop = LpiProp + LLProp;
R = Rprop - Rold;

% Get the comparison
w = log(rand);

% Checking the comparison, put either the proposed value or the old value into the
% vector for the first entry
if (R >= w)
    % Assign the proposed values to the array
    LLdraws(1,1) = LLProp;
    LpiDraws(1,1) = LpiProp;
    tDraws(1,:) = tProp;
    sDraws(:,:,1) = StDrawProp;
    mDraws(:,:,1) = mProp;
    Rlog(:,1) = R;
    wlog(:,1) = w;
    
    % Set the old to the proprosed
    Rold = Rprop;
    tOld = tProp;
    sOld = StDrawProp;
    mOld = mProp;
    VOld = VProp;
    LLold = LLProp;
    LpiOld = LpiProp;
    
    AR(1,1) = 1;
    
else

    % Assign the old values to the array
    LLdraws(1,1) = LLold; 
    LpiDraws(1,1) = LpiOld;
    tDraws(1,:) = tOld;                     
    sDraws(:,:,1) = sOld;
    mDraws(:,:,1) = mOld;
    VDraws(:,:,:,1) = VOld;
    Rlog(:,1) = R;
    wlog(:,1) = w;
    
    AR(1,1) = 0;

end

% Clear the temp variables
clear A B C D F G S 

% Now that we have the initial setup for the MH loop, we can run through the rest of
% the draws.

disp('Entering main Loop')

% Initialize skipping counter
j = 1;

for i = 2:totDraw
    
    LpiProp = -Inf;
    while (LpiProp == -Inf)
        
        % Create the proposal
        tProp = tOld + mvnrnd(shkMu,stepSize*shkSig);
        
        % Correct for the real stuff vs the estimated stuff 
        tProp(realLoc) = thetaHLW(realLoc);                
        
        % If we aren't directly estimating a1, then it is implied by the estimate of a2
        % and the pre-defined a1pa2 from the HLW Table 1
        if (ismember(1,estLoc)==0)
            tProp(1) = a1pa2 - tProp(2);
        end
        
        % If we aren't directly estimating sigma_g (s5), then it is implied by the estimate of
        % sigma_ystar (s4) and lambda_G
        if (ismember(10,estLoc)==0)
            tProp(10) = lambG*tProp(9);
        end
         
        % If we aren't directly estimating sigma_z (s3), then it is implied by the estimate of
        % sigma_ytilde (s1), ar and lambda_z
        if (ismember(8,estLoc)==0)
            tProp(8) = -lambZ*tProp(6)/tProp(3);
        end
                
        % Get the log-prior evaluation, it may be neg. infinity
        LpiProp = priorEval_distribute(tProp,estLoc);

    end

    % Construct the state space structures for the given parameters
    [A,B,C,D,F,G,S] = paramMat_distribute(tProp,y,u);
    
    % Get the loglikelihood of the proposed parameter vector
    [LLProp,LLtProp,StDrawProp,mProp,VProp] = ffbs_kflfvg(y,u,A,B,C,D,F,G,S,m0hp,V0hp,1);

    % Assemble the proposed statistic
    Rprop = LpiProp + LLProp;
    R = Rprop - Rold;

    % Get the comparison
    w = log(rand);
    
    % Checking the comparison, put either the proposed value or the old value into the
    % vector for the first entry
    if (R >= w)
                
        % Assign the proposed values to the array
        LLdraws(i,1) = LLProp;
        LpiDraws(i,1) = LpiProp;
        tDraws(i,:) = tProp;
        sDraws(:,:,i) = StDrawProp;
        mDraws(:,:,i) = mProp;
        VDraws(:,:,:,i) = VProp;
        Rlog(:,i) = R;
        wlog(:,i) = w;

    
        % Set the old to the proprosed
        Rold = Rprop;
        tOld = tProp;
        sOld = StDrawProp;
        mOld = mProp;
        VOld = VProp;
        LLold = LLProp;
        LpiOld = LpiProp;
        
        % Change the Accept-Reject
        AR(i,1) = 1;
    
    else

        % Assign the old values to the array
        LLdraws(i,1) = LLold;  
        LpiDraws(i,1) = LpiOld;
        tDraws(i,:) = tOld;                     
        sDraws(:,:,i) = sOld;
        mDraws(:,:,i) = mOld;
        VDraws(:,:,:,i) = VOld;
        Rlog(:,i) = R;
        wlog(:,i) = w;
        
        % Change the Accept-Reject
        AR(i,1) = 0;
    end
    
    if (mod(i,modShow) == 0)

        rptStr = sprintf('Draw %i of %i, %3.1f percent.  Overall A/R: %3.1f, Last %i A/R: %3.1f',...
                          i,totDraw,(i/totDraw*100),(nanmean(AR)*100),modShow,...
                          (nanmean(AR((i-(modShow-1)):i,1))*100));
        
        disp(rptStr);
    end
    
    
    % Clear the temp variables
    clear A B C D F G S  
    clear LLProp LLtProp StDrawProp mProp VProp w R
    
end

%% At this point we want to extract the every skipth draw
tDrawsSkip = tDraws(burnIn+1:skip:end,:);
LLdrawsSkip = LLdraws(burnIn+1:skip:end,1);
LpiDrawsSkip = LpiDraws(burnIn+1:skip:end,1);
sDrawsSkip = sDraws(:,:,burnIn+1:skip:end);
mDrawsSkip = mDraws(:,:,burnIn+1:skip:end);
VDrawsSkip = VDraws(:,:,:,burnIn+1:skip:end);

% Now that we have the final versions of the m, V and s draws that we will keep, we can
% dump the very memory intensive full collection of draws of each of those.
clear VDraws mDraws sDraws

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Go through the draws and build the rStar and sigR
% Need this vector to extract the elements of m that are used to build rStar given that
% r* = 4g + z

% To extract potential output
pp = [1 0 0 0 0 0 0]';

% To extract the path of z
qqZ = [0 0 0 0 0 1 0]';

% To extract the path of g
qqG = [0 0 0 1 0 0 0]';

for (ii = 1:M)
    
    % The draws
    mTemp = mDrawsSkip(:,:,ii);
    VTemp = VDrawsSkip(:,:,:,ii);
    sTemp = sDrawsSkip(:,:,ii);
    muGTemp = tDrawsSkip(ii,14);
    rhoGTemp = tDrawsSkip(ii,12);
    qq = [0 0 0 4*rhoGTemp 0 1 0]';
    rCTemp = 4*muGTemp*(1-rhoGTemp)*ones(size(mTemp,2),1);
    
    % Generate the gap from the mean state estimate
    yPDraws(:,ii) = (pp'*mTemp)';
    zDraws(:,ii) = (qqZ'*mTemp)';    
    gDraws(:,ii) = (qqG'*mTemp)';    
    
    % Generate one-sided mean estimates of rStar from the mean state estimates
    rStarDraws(:,ii) = rCTemp + (qq'*mTemp)';
    
    % Smoothed built from Carter-Kohn style forward-filter, backward sampled states
    rStarSmoothDraws(:,ii) = rCTemp + (qq'*sTemp)';
    
    % Generate the standard error for rStar from the variance covariance matrix of the
    % state estimates
    for (jj = 1:size(VTemp,3))
        sigyPDraws(jj,ii) = sqrt(pp'*VTemp(:,:,jj)*pp);
        sigzDraws(jj,ii) = sqrt(qqZ'*VTemp(:,:,jj)*qqZ);
        siggDraws(jj,ii) = sqrt(qqG'*VTemp(:,:,jj)*qqG);
        sigRDraws(jj,ii) = sqrt(qq'*VTemp(:,:,jj)*qq);
    end
    
    clear mTemp VTemp sTemp muGTemp rhoGTemp qq rcTemp
    
end

%%% Much space can be saved by dropping the draws of the V matrix
%clear VDrawsSkip

% Save all of the results in a mat file
sf
save(sf,'-v7.3')
disp('Full .mat file saved.');

disp('Script complete.')
