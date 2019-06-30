# bert_for_text_matching
该仓库以文本匹配任务为例展示了如何使用**Bert**预训练模型在特定领域语料上进行微调，并使用**tensorflow serving**部署在生产环境中用做推理。

## What I did:

- 支持文本匹配任务
- Bert官方代码库使用了大量TPU相关逻辑，这里做了相应简化
- 使用标准的tf.data和tf.estimator api构建模型
- 使用tensorflow serving将模型部署到生产环境

## Data

采用LCQMC中文文本匹配数据集作为模型微调对象。支持的训练方式为pointwise，句子对0/1二分类。

## Requirements

python 3

tensorflow 1.12.0

docker `(for tensorflow serving)`

## Usage

####  step 1 领域数据微调

```shell
export BERT_BASE_DIR=/path/to/bert/chinese_L-12_H-768_A-12

python train.py \
  --do_train=true \
  --do_eval=true \
  --do_predict=true \
  --data_dir=/path/to/your/data \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=64 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=/path/for/output/
```

#### step 2 导出模型

tensorflow serving部署之前需要将模型从checkpoint转化为saved_model格式。

```shell
python export.py \
  -c path/for/bert_config\
  -m path/to/checkpoints \
  -o path/for/saved_model 
```

#### step 3 部署模型

- 安装docker，并拉取tensorflow serving镜像(若使用GPU加速，还需安装nvidia-docker)

  ```
  docker pull tensorflow/serving/:1.12.0-gpu
  ```

- 启动容器服务，对外提供rest接口

  ```shell
  docker run -p 8501:8501 \
    --mount type=bind,source=path/to/your/local/saved_models,target=/models \
    -e MODEL_NAME=serving_model -t tensorflow/serving:1.12.0-gpu
  ```

## Reporting issues

Please let me know, if you encounter any problems.