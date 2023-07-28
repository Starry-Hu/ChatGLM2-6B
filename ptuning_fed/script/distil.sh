PRE_SEQ_LEN=128
LR=1e-4

NUM_GPUS=2
#bs=5
#gradient_accumulation_steps=8
num_student_layers=16
pad=2
MODEL="gpt2-xl"


run_dir=$1
cuda_visible=$2  # 0,1,2,3
bs=$3

case $run_dir in
  hit) run_dir='/data/xiangjing';;
  ali-dsw) run_dir='/mnt/workspace';;
  ali-dlc) run_dir='/root/data';;
esac


process_count=$(echo "$cuda_visible" | awk -F"," '{print NF}')

gradient_accumulation_steps=$(echo "80 / ($process_count * $bs)" | bc)


# total bs=80
CUDA_VISIBLE_DEVICES=$cuda_visible torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS main_distil.py \
    --do_distil \
    --train_tokenized_dataset ${run_dir}/datasets/pile/gpt_tokenized/00 \
    --val_tokenized_dataset ${run_dir}/datasets/wikitext-2-raw-v1/gpt_tokenized/ \
    --preprocessing_num_workers 88 \
    --model_name_or_path ${run_dir}/pretrain/nlp/${MODEL} \
    --output_dir ${run_dir}/output/fedllm/distilled/emulators/${MODEL}_${num_student_layers}_${pad}_${pad} \
    --overwrite_output_dir \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size $bs \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --logging_steps 500 \
    --save_steps 2000 \
    --eval_steps 200 \
    --lr_scheduler_type cosine \
    --learning_rate $LR \
    --warmup_steps 2000 \
    --weight_decay 0.1 \
    --num_student_layers $num_student_layers \
    --student_l_pad $pad \
    --student_r_pad 2 \
    --num_train_epochs 1 \
    --lm_weight 1.0 \
    --kd_weight 30.0 \
    --block_size 512 \
    --bf16 \
    --save_total_limit 3 \
    --seed 42

#    --model_name_or_path THUDM/chatglm2-6b \
#    --quantization_bit 4 \
#    --pre_seq_len $PRE_SEQ_LEN \
