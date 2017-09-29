import tensorflow as tf

def my_metrics(_y, pred):
    original_y, predicted_y = tf.argmax(_y, 1), tf.argmax(pred, 1)
    correct_pred = tf.equal(original_y, predicted_y)
    incorrect_pred = tf.logical_not(correct_pred)

    # confusion matrix
    #               true 1   true 0
    # predicted 1     TP       FP
    # predicted 0     FN       TN

    tp = tf.reduce_sum(tf.boolean_mask(original_y, correct_pred))
    tn = tf.reduce_sum(tf.boolean_mask(1 - original_y, correct_pred))
    fp = tf.reduce_sum(tf.boolean_mask(original_y, incorrect_pred))
    fn = tf.reduce_sum(tf.boolean_mask(1 - original_y, incorrect_pred))
    
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.int64))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fscore = 2 / (1/precision + 1/recall)
    
    return accuracy, precision, recall, fscore