import os
import time
import datetime
import operator
import glob
import tensorflow as tf
import numpy as np
import data_helpers as model
from tensorflow.contrib import learn

query = raw_input("\n\t Enter your question? like \n\t  what is your name? \n\n")
query = query.strip()
query = model.clean_str(query)
query = query.split(" ")

sentences, y_test = model.load_data_and_labels()
sequence_length = max(len(x) for x in sentences)
sentences_padded = model.pad_sentences(sentences)
vocabulary, vocabulary_inv = model.build_vocab(sentences_padded)
num_padding = sequence_length - len(query)
new_sentence = query + ["<PAD/>"] * num_padding
x = np.array([vocabulary[word] for word in new_sentence])
x_test = np.array([x])

"""
Reference from https://github.com/dennybritz/cnn-text-classification-tf/blob/master/eval.py
"""
# Parameters
# ==================================================
# Eval Parameters
# tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 64)")
# tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
# tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# # Misc Parameters
# tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
# tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags()

dirr = max(glob.glob(os.path.join("runs/", '*/')), key=os.path.getmtime)
checkpoint_file = tf.train.latest_checkpoint(dirr + 'checkpoints/')
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=False)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = model.batch_iter(x_test, 30, 1, shuffle=False)

        all_predictions = []
        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            print batch_predictions            
            all_predictions = np.concatenate([all_predictions, batch_predictions])

if(all_predictions[0] == 0):
    print 'Who'
elif(all_predictions[0] == 1):
    print 'When'
elif(all_predictions[0] == 2):
    print 'What'
elif(all_predictions[0] == 3):
    print 'Affirmation'
else:
    print 'Unknown'
