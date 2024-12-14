seed=${1:-6}
dataset=${2:-"Jigsaw"}
data_dir=${3:-"../jigsaw-dataset"}
arch=${4:-"transformer"}
method=${5:-"crest"}
subset_dir=${6:-"../subset"}
model_path=${7:-"/home/ruichenbruin24/cs260d-final-project/CREST/outputs/Jigsaw_transformer_lr0.1_warm-20_train0.10_random0.01-start0_batchsize16_crest-batchnummul3.0-interalmul400.0_thresh-factor0.1_coreset_momentum_seed_156/model_final.pt"}

python crest_train.py \
--dataset ${dataset} \
--data_dir ${data_dir} \
--arch ${arch} \
--selection_method ${method} \
--seed ${seed} \
--batch_size 16 \
--save_subset_dir ${subset_dir} \
--epochs 2 \
--interval_mul 200 \
--model_name_or_path ${model_path} \
--check_interval 200 \
--batch_num_mul 3