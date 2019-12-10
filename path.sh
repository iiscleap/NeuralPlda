export KALDI_ROOT="/state/partition1/softwares/Kaldi_Jan2018/kaldi"
# export KALDI_ROOT="/state/partition1/softwares/kaldi_sre"
#export KALDI_ROOT="/home/student1/softwares/kaldi_diarization"
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=/home/data2/SRE2019/prashantk/voxceleb/v2/utils/:$KALDI_ROOT/tools/openfst/bin:/home/data2/SRE2019/prashantk/voxceleb/v2:$PATH:$KALDI_ROOT/src/nnet3bin
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C
export PATH="/state/partition1/softwares/kaldi/src/featbin:$PATH"
export PATH="/state/partition1/softwares/kaldi/tools/sph2pipe_v2.5:$PATH"
LMBIN=$KALDI_ROOT/tools/irstlm/bin
SRILM=$KALDI_ROOT/tools/srilm/bin/i686-m64
BEAMFORMIT=$KALDI_ROOT/tools/BeamformIt

export PATH=$PATH:$LMBIN:$BEAMFORMIT:$SRILM
