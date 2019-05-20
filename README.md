# bert_for_text_matching
### what I did:

1. Optimize the BERT fine tune mode to a simplified form.
2. Use the  tf.estimator api to replace the original tpu-estimator api.
3. Now, it supports for training text matching/similarity model, aka, classic sentence pair task.

## fit your own data

If you want to use the fine tune code to train on you own data for specific task, you can just change the Precessor class.

## usage

```shell
export BERT_BASE_DIR=/path/to/bert/uncased_L-12_H-768_A-12

python train.py \
  --do_train=true \
  --do_eval=true \
  --do_predict=true \
  --data_dir=/path/to/your/data \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=/path/for/output/
```

## reporting issues

Please let me know, if you encounter any problems.