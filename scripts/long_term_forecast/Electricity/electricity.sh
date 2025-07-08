export CUDA_VISIBLE_DEVICES=0
model_name=TimesNet

root_path="./dataset/"
data_path="electricity.csv"  # 引用变量
enc_in=321  # electricity 数据集有 321 个特征

for pred_len in 96 192 336 720
do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path "$root_path" \
    --data_path "$data_path" \
    --model_id "TimesNet_electricity_all${pred_len}" \
    --model "$model_name" \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len "$pred_len" \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in "$enc_in" \
    --dec_in "$enc_in" \
    --c_out "$enc_in" \
    --d_model 16 \
    --d_ff 32 \
    --des "Exp_electricity_all" \
    --itr 1 \
    --top_k 5
done