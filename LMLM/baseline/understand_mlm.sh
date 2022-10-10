export CUDA_VISIBLE_DEVICES=2
source activate pivot
mlm_prob=0.15
pivot_num=100


gpu_num=4
for year in "2017" "2019" "2021";do
  for method in 'jsd';do
    python /data1/temporal_master/src/run_mlm.py \
        --model_name_or_path bert-base-uncased \
        --train_file /data1/pivot-extraction/data/arxiv_word/2015/random_sentence.csv \
        --validation_file /data1/pivot-extraction/data/arxiv_word/$year/100pivot_2000sentence/${method}_sentence.csv \
        --per_device_train_batch_size 64 \
        --per_device_eval_batch_size 64 \
        --line_by_line \
        --do_eval \
        --seed 42 \
        --num_train_epochs 5 \
        --pivot_file /data1/pivot-extraction/data/arxiv_word/$year/frequency_word.csv \
        --mlm_prob ${mlm_prob} \
        --pivot_num ${pivot_num} \
        --max_seq_length 128 \
        --overwrite_output_dir \
        --output_dir /data1/temporal_master/save_models/05_19/frequency/bert-base/${method}/${year}/
  done
done

for year in "2017" "2019" "2021";do
  for method in 'jsd';do
    python /data1/szc/temporal_master/src/run_mlm.py \
        --model_name_or_path bert-base-uncased \
        --train_file /data1/szc/pivot-extraction/data/arxiv_word/2015/random_sentence.csv \
        --validation_file /data1/szc/pivot-extraction/data/arxiv_word/$year/100pivot_2000sentence/${method}_sentence.csv \
        --per_device_train_batch_size 64 \
        --per_device_eval_batch_size 64 \
        --line_by_line \
        --do_eval \
        --seed 42 \
        --num_train_epochs 5 \
        --pivot_file /data1/szc/pivot-extraction/data/arxiv_yake/yake_${year}.txt \
        --mlm_prob ${mlm_prob} \
        --pivot_num ${pivot_num} \
        --max_seq_length 128 \
        --overwrite_output_dir \
        --output_dir /data1/szc/temporal_master/save_models/05_19/important/bert-base/${method}/${year}/

# test
#for year in "2017" "2019" "2021";do
#  for method in 'jsd';do
#    python /data1/szc/temporal_master/src/run_mlm.py \
#        --model_name_or_path /data1/szc/temporal_master/save_models/our_model/2015_2021 \
#        --train_file /data1/szc/pivot-extraction/data/arxiv_word/2015/random_sentence.csv \
#        --validation_file /data1/szc/pivot-extraction/data/arxiv_word/$year/100pivot_2000sentence/${method}_sentence.csv \
#        --per_device_train_batch_size 64 \
#        --per_device_eval_batch_size 64 \
#        --line_by_line \
#        --do_eval \
#        --seed 42 \
#        --num_train_epochs 5 \
#        --pivot_file /data1/szc/pivot-extraction/saved_usage/${method}_15_${year}.txt \
#        --mlm_prob ${mlm_prob} \
#        --pivot_num ${pivot_num} \
#        --max_seq_length 128 \
#        --output_dir /data1/szc/temporal_master/save_models/caogao

#        --overwrite_output_dir \
#        --save_steps 100000 \
  done
done

# /data1/szc/temporal_master/save_models/05_14/new_word/semantic/all_bert/$year/$method

#python understand.py \
#    --model_name_or_path /data1/szc/temporal_master/save_models/random_pretraning/arxiv_2015 \
#    --train_file /data1/szc/pivot-extraction/data/arxiv_word/2015/frequency_sentence.csv \
#    --validation_file /data1/szc/pivot-extraction/data/arxiv_word/2015/2015-2021-100pivot_2000sentence/semantic_sentence.csv \
#    --per_device_train_batch_size 64 \
#    --per_device_eval_batch_size 64 \
#    --line_by_line \
#    --do_eval \
#    --num_train_epochs 5 \
#    --pivot_file /data1/szc/pivot-extraction/data/arxiv_yake/yake_2015.txt \
#    --mlm_prob ${mlm_prob} \
#    --pivot_num ${pivot_num} \
#    --seed 42 \
#    --max_seq_length 128 \
#    --overwrite_output_dir \
#    --save_steps 100000 \
#    --output_dir /data1/szc/temporal_master/caogao
