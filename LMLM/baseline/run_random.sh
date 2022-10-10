#!/bin/sh

export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7

for year in "2021" "2019" "2017";do
  python random_pretrain.py \
      --model_name_or_path bert-base-uncased \
      --train_file /data1/pivot-extraction/data/arxiv/arxiv_$year.csv \
      --validation_file /data1/pivot-extraction/data/arxiv_word/2021/100pivot_2000sentence/jsd_sentence.csv \
      --line_by_line \
      --do_train \
      --do_eval \
      --per_device_train_batch_size 64 \
      --save_steps 3000 \
      --per_device_eval_batch_size 128 \
      --output_dir /data1/temporal_master/save_models/random_pretraning/arxiv_$year/ \
      --overwrite_output_dir \
      --num_train_epochs 20 \
      --seed 42 \
      --logging_dir /data1/temporal_master/logs/caogao \
      --logging_first_step \
      --logging_steps 100 \
      --max_seq_length 128
done


#for year in "2017" "2019" "2021";do
#  for method in 'jsd' 'ed' 'apd';do
#    python random_pretrain.py \
#        --model_name_or_path bert-base-uncased \
#        --train_file /data1/szc/pivot-extraction/data/arxiv_word/2015/random_sentence.csv \
#        --validation_file /data1/szc/pivot-extraction/data/arxiv_word/$year/100pivot_2000sentence/${method}_sentence.csv \
#        --line_by_line \
#        --do_eval \
#        --per_device_train_batch_size 80 \
#        --save_steps 3000 \
#        --per_device_eval_batch_size 16 \
#        --output_dir /data1/szc/temporal_master/save_models/05_14/sentence/semantic/bert-base/$year/$method \
#        --overwrite_output_dir \
#        --num_train_epochs 20 \
#        --seed 42 \
#        --logging_dir /data1/szc/temporal_master/logs/caogao \
#        --logging_first_step \
#        --logging_steps 100 \
#        --max_seq_length 128
#  done
#done

# "frequency_sentence.csv" "important_sentence.csv" "semantic_sentence.csv"

#for year in "2015" ;do
#  for setentence in "frequency_sentence.csv"  ; do
#    python random_pretrain.py \
#        --model_name_or_path /data1/szc/temporal_master/save_models/random_pretraning/arxiv_2015 \
#        --train_file  /data1/szc/pivot-extraction/data/arxiv/arxiv_$year.csv \
#        --validation_file /data1/szc/pivot-extraction/data/arxiv_word/2015/2015-2021-100pivot_2000sentence/$setentence \
#        --line_by_line \
#        --do_eval \
#        --per_device_train_batch_size 80 \
#        --save_steps 100000000 \
#        --per_device_eval_batch_size 128 \
#        --output_dir /data1/szc/temporal_master/save_models/random_pretraning/2021_100pivot_2000sentence/2015-2015/$setentence \
#        --overwrite_output_dir \
#        --num_train_epochs 5 \
#        --seed 42 \
#        --logging_dir /data1/szc/temporal_master/logs \
#        --logging_first_step \
#        --logging_steps 100 \
#        --max_seq_length 128
#  done
#done
