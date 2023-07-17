PRE_SEQ_LEN=128
LR=2e-2
NUM_GPUS=4

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS main.py \
    --do_train \
    --do_eval \
    --train_file AdvertiseGen/train.json \
    --validation_file AdvertiseGen/dev.json \
    --preprocessing_num_workers 10 \
    --prompt_column content \
    --response_column summary \
    --overwrite_cache \
    --model_name_or_path /data/xiangjing/pretrain/nlp/gpt2-xl \
    --output_dir output/adgen-chatglm2-6b-pt-$PRE_SEQ_LEN-$LR \
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
    --use_lora

#    --do_distil \

#    --model_name_or_path THUDM/chatglm2-6b \
#    --quantization_bit 4 \
#    --pre_seq_len $PRE_SEQ_LEN \
