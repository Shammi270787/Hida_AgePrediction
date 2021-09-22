#!/bin/bash
Usage() {
    echo ""
    echo "Usage: linear_reg  -input_nifti_name.nii.gz <-output_prefix> <-output_dir> <-bet_f_value> "
    echo ""
    echo "e.g.,   linear_reg -i subj-001_T1w.nii.gz [-p aff_] [-o /results] [-t .3]"
    echo ""
    exit 1
}

cur_dir="$(pwd)"
while getopts ":i:p:o:t:" opt; do
  case $opt in
    i) subj="$OPTARG"
    ;;
    p) pre_name="$OPTARG"
    ;;
    o) output_dir="$OPTARG"
    ;;
    t) threshold="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

[ "$subj" = "" ] && Usage

if [ -z "$pre_name"  ] ; then
    pre_name=aff_
fi

if [ -z "$threshold"  ] ; then
    threshold=0.3
fi

if [ -z "$output_dir"  ] ; then
    output_dir=$cur_dir
fi

tmp_dir=$(mktemp -d -t ci-XXXXXXXXXX)
echo $tmp_dir
echo 'input file:' $subj
echo 'output file prefix:' $pre_name
echo 'output dir: ' $output_dir
echo 'bet f value: ' $threshold

#
cd $tmp_dir
echo "Current working directory: $(pwd)"
#
cp $subj .
g="$(basename -- $subj)"
echo "Copying subject $g to working dir"

echo ""
echo "Starting the processing..."
$FSLDIR/bin/fslreorient2std $g std_${g} ;
$FSLDIR/bin/robustfov -i std_${g} -r fov_${g} ;
${FSLDIR}/bin/bet fov_${g} brain_${g} ;
${FSLDIR}/bin/standard_space_roi brain_${g} cut_${g} -roiNONE -ssref ${FSLDIR}/data/standard/MNI152_T1_1mm_brain -altinput fov_${g} ;
${FSLDIR}/bin/bet cut_${g} brain_${g} -f $threshold ;
${FSLDIR}/bin/flirt -in brain_${g} -ref ${FSLDIR}/data/standard/MNI152_T1_1mm_brain -out $pre_name${g}

cp $pre_name${g} $output_dir/$pre_name${g}
cd $cur_dir
echo "removing working dir..."
rm -rf $tmp_dir
