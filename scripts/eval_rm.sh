set -ex

GPU_ID=$1
data_file=$2

CUDA_VISIBLE_DEVICES=${GPU_ID} python eval_rm.py \
    --data_dir ${data_file} \
    --seed 42 \
    --reuse \
    --pattern_name "*/*/*tst_generated.pkl" # "*/*tst_generated.pkl"