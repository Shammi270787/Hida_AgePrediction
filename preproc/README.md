Han Peng et al. "Accurate brain age prediction with lightweight deep neural networks"
'...The image preprocessing pipeline is described in ( Alfaro-Almagro et al., 2018)''

Fidel Alfaro-Almagro et al "Image processing and Quality Control for the first 10,000 brain imaging datasets from UK Biobank"

 ''... Tools used to achieve this include BET (Brain Extraction Tool, Smith (2002))
and FLIRT (FMRIB's Linear Image Registration Tool, Jenkinson and Smith, 2001,
Jenkinson et al., 2002),14 in conjunction with the MNI152 “nonlinear 6th generation”
standard-space T1 template'
###  --------------------------------------------------------------------------------

input arguments:

-i: subject's full path

-p: (optional) prefix of output file. Default: aff_

-o: (optional) output directory. Default: current directory

-t: (optional) -f value of bet. Default: 0.3 (it is recommended to be set after visual inspection of the results)

###  --------------------------------------------------------------------------------
### Make sure that $FSLDIR points to the location where FSL is installed.
### For more information regarding your shell configuration visit 
### https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation/ShellSetup 
