# rStarDist
Code from Lewis and Vazquez-Grande (2018) WP -- Measuring the Natural Rate of Interest: a Note on Transitory Shocks

The code can be run as a script to generate the estimates used in the latest [public version of the paper](https://www.federalreserve.gov/econres/feds/files/2017059r1pap.pdf "August, 2018 Working Paper").  The code was run using MATLAB 2017, but earlier versions should work as well.  Cloning the project and running the rStarDist/code/estimateModel.m script should yield the estimates found in the latest public version of the paper. 

Users can ignore the two inputs to the main function (estimateModel.m).  These exist to enable usage of the specific scheduler used on our local cluster.  The code will run as normal if the executed as a script with no inputs, and it will generate the estimates used in the paper linked above.
