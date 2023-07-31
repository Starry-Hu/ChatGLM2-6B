PRE_SEQ_LEN=128
LR=2e-2

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

echo "cuda_visible: $cuda_visible"
NUM_GPUS=$(echo "$cuda_visible" | awk -F"," '{print NF}')

gradient_accumulation_steps=$(( 80 / ($NUM_GPUS * $bs) ))
echo "gradient_accumulation_steps: ${gradient_accumulation_steps}"

MASTER_PORT=$(shuf -n 1 -i 10000-65535)

CUDA_VISIBLE_DEVICES=$cuda_visible torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT test_load_client.py \
    --do_train \
    --do_eval \
    --train_file AdvertiseGen/train.json \
    --validation_file AdvertiseGen/dev.json \
    --preprocessing_num_workers 10 \
    --prompt_column content \
    --response_column summary \
    --overwrite_cache \
    --model_name_or_path ${run_dir}/pretrain/nlp/${MODEL} \
    --output_dir ${run_dir}/output/adgen-chatglm2-6b-pt-$PRE_SEQ_LEN-$LR \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 128 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps 100 \
    --logging_steps 10 \
    --save_steps 50 \
    --learning_rate $LR \
    --num_student_layers 16 \
    --student_l_pad 2 \
    --student_r_pad 2 \
    --lm_weight 0.8 \
    --kd_weight 0.2 \
    --use_lora \
    --load_student ${run_dir}/output/fedllm/distilled/emulators/${MODEL}_${num_student_layers}_${pad}_${pad}/checkpoint-45000/student.pt


#    --do_distil \

#    --model_name_or_path THUDM/chatglm2-6b \
#    --quantization_bit 4 \
#    --pre_seq_len $PRE_SEQ_LEN \
