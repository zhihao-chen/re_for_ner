CURRENT_DIR=/data/chenzhihao/RE-For-NER
export DATA_DIR=$CURRENT_DIR/datas
export OUTPUT_DIR=$CURRENT_DIR/datas/output
export LOG_DIR=$CURRENT_DIR/datas/log_out
MODEL_NAME_OR_PATH="hfl/chinese-roberta-wwm-ext"
#MODEL_NAME_OR_PATH=/data/chenzhihao/pretrain_bert/output_dapt
#MODEL_NAME_OR_PATH=$CURRENT_DIR/output
MODEL_TYPE="bert"

python run_with_accelerate.py \
  --model_name_or_path=$MODEL_NAME_OR_PATH \
  --data_dir=$DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --model_type=$MODEL_TYPE \
  --log_dir=$LOG_DIR \
  --do_train \
  --do_eval \
  --do_predict \
  --do_lower_case \
  --loss_type=lsr \
  --soft_label \
  --train_max_seq_length=512 \
  --eval_max_seq_length=512 \
  --per_gpu_train_batch_size=8 \
  --per_gpu_eval_batch_size=8 \
  --learning_rate=1e-5 \
  --num_train_epochs=50.0 \
  --logging_steps=50 \
  --save_steps=50 \
  --overwrite_output_dir \
  --seed=42 \
  --cuda=1 \
  --patience=10 \
  --local_rank=-1 \
  --rdrop \
  --rdrop_alpha=5 \
  --dropout_rate=0.5