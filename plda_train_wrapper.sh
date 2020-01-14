. ./cmd.sh
. ./path.sh
set -e


main_dir=/home/data2/SRE2019/prashantk/NeuralPlda
# data=$1
# eval=$2
echo "Stage 10"



# Compute the mean vector for centering the evaluation xvectors.
$train_cmd $main_dir/logs/compute_mean.log \
ivector-mean scp:$main_dir/xvector_train.scp \
$main_dir/Kaldi_Models/Train/mean.vec || exit 1;

echo "Mean done"

# This script uses LDA to decrease the dimensionality prior to PLDA.
lda_dim=170
$train_cmd $main_dir/logs/lda.log \
ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
"ark:ivector-subtract-global-mean scp:$main_dir/xvector_train.scp ark:- | ivector-normalize-length ark:-  ark:- |" \
ark:$main_dir/utt2spk_train_new $main_dir/Kaldi_Models/Train/transform.mat || exit 1;

# Train the PLDA model.
$train_cmd $main_dir/logs/plda.log \
ivector-compute-plda ark:$main_dir/spk2utt_train_new \
"ark:ivector-subtract-global-mean scp:$main_dir/xvector_train.scp ark:- | transform-vec $main_dir/Kaldi_Models/Train/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
$main_dir/Kaldi_Models/Train/plda || exit 1;

# $train_cmd exp/xvectors_sre18_dev_unlabeled/log/plda_adapt.log \
# ivector-adapt-plda --within-covar-scale=0.75 --between-covar-scale=0.25 \
# $main_dir/xvectors_${data}/plda \
# "ark:ivector-subtract-global-mean scp:$main_dir/xvectors_sre18_dev_unlabeled/xvector.scp ark:- |  transform-vec $main_dir/xvectors_${data}/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
# $main_dir/xvectors_sre18_dev_unlabeled/plda_adapt || exit 1;
echo "Stage 10 done"
