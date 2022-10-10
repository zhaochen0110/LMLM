export CUDA_VISIBLE_DEVICES=5
source activate pivot
#mlm_prob=0.15
#pivot_num=100
gpu_num=1
#mlm_prob=0.6
#pivot_num=800
mlm_prob=0.3
pivot_num=800
seed_list=(5 10 15 20 42)
for year in "16" ;do
  for loop in $(seq 0 4); do
    python src/finetuning.py \
      --model_name_or_path saved_models/mlm_model/poliaff${year}-${mlm_prob}-${pivot_num}-mlm-${gpu_num}gpu \
      --train_file /data1/pivot-extraction/data/poliaff/train/20${year}.csv \
      --validation_file /data1/pivot-extraction/data/poliaff/test/2017.csv \
      --do_train \
      --do_eval \
      --seed ${seed_list[${loop}]} \
      --per_device_eval_batch_size 128 \
      --per_device_train_batch_size 128 \
      --output_dir saved_models/finetuning_model/poliaff-${mlm_prob}-${pivot_num}-${year}-17-${gpu_num}gpu/${seed_list[$loop]} \
      --output saved_models/finetuning_model/poliaff-${mlm_prob}-${pivot_num}-${year}-17-${gpu_num}gpu/${seed_list[$loop]} \
      --overwrite_output_dir \
      --save_steps 100000 \
      --num_train_epochs 5 \
      --max_seq_length 128 \
      --use_special_tokens
  done
done