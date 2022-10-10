export CUDA_VISIBLE_DEVICES=3
source activate pivot
#mlm_prob=0.15
#pivot_num=300
gpu_num=1
#mlm_prob=0.6
#pivot_num=800
mlm_prob=0.3
pivot_num=800

python src/run_mlm.py \
    --model_name_or_path bert-base-uncased \
    --train_file /data1/pivot-extraction/data/ner/ner_2017._unlabel.txt \
    --validation_file /data1/pivot-extraction/data/ner/ner_2017._unlabel.txt \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --line_by_line \
    --do_train \
    --do_eval \
    --num_train_epochs 5 \
    --pivot_file /data1/pivot-extraction/saved_usage/poliaff_16_17.txt \
    --mlm_prob ${mlm_prob} \
    --pivot_num ${pivot_num} \
    --max_seq_length 128 \
    --overwrite_output_dir \
    --save_steps 100000 \
    --output_dir saved_models/mlm_model/poliaff16-${mlm_prob}-${pivot_num}-mlm-${gpu_num}gpu

#    --logging_strategy steps \
#    --logging_dir logs/reverse-test \
#    --logging_steps 10 \
#
#python src/finetuning.py \
#    --model_name_or_path saved_models/mlm_model/test${mlm_prob}-${pivot_num}-mlm-${gpu_num}gpu-reverse \
#    --train_file /data1/szc/pivot-extraction/data/arxiv_2009_part_train.csv \
#    --validation_file /data1/szc/pivot-extraction/data/arxiv_2021_part_test.csv \
#    --do_train \
#    --do_eval \
#    --seed 15 \
#    --per_device_eval_batch_size 128 \
#    --per_device_train_batch_size 128 \
#    --output_dir saved_models/finetuning_model/test${mlm_prob}-${pivot_num}-09-21-${gpu_num}gpu-reverse \
#    --output saved_models/finetuning_model/test${mlm_prob}-${pivot_num}-09-21-${gpu_num}gpu-reverse \
#    --overwrite_output_dir \
#    --save_steps 100000 \
#    --num_train_epochs 3 \
#    --max_seq_length 128 \
#    --use_special_tokens