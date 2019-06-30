"""Export model as a saved_model using tf.estimator api"""

__author__ = "Bin Zhong"

import argparse
import tensorflow as tf

from train import model_fn_builder

def serving_input_receiver_fn():
    """Serving input_fn that builds features from placeholders
    Returns
    -------
    tf.estimator.export.ServingInputReceiver
    """
    input_ids = tf.placeholder(dtype=tf.string, shape=[None, 64], name='input_ids')
    input_mask = tf.placeholder(dtype=tf.string, shape=[None, 64], name='input_mask')
    segment_ids = tf.placeholder(dtype=tf.string, shape=[None, 64], name='segment_ids')
    receiver_tensors = {'input_ids': input_ids,
                        'input_mask': input_mask,
                        'segment_ids': segment_ids}
    features = {'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids}
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Export model from checkpoint to saved_model.")
    parser.add_argument("-o", "--output_dir", required=True, type=str, help="path for output saved model")
    parser.add_argument("-m", "--model_dir", required=True, type=str, help="model dir")
    parser.add_argument("-c", "--bert_config", required=True, type=str, help="path for bert config")
    args = parser.parse_args()

    model_dir = args.model_dir
    output_dir = args.output_dir
    bert_config = args.bert_config

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=2,
        init_checkpoint=None,
        learning_rate=2e-5,
        num_train_steps=1000,
        num_warmup_steps=1000,
        use_one_hot_embeddings=False)

    estimator = tf.estimator.Estimator(model_fn, model_dir, params=None)
    estimator.export_saved_model(output_dir, serving_input_receiver_fn)