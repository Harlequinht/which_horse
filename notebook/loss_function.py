from keras.utils import get_custom_objects
import tensorflow as tf
from keras import backend as K


def custom_loss_function(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)  # Convert y_true to float32
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    
    win_horse = y_true[:, 0]
    win_odds = y_true[:, 1]
    gain_loss_vector = K.concatenate([(win_horse * (win_odds - 1)) + (1 - win_horse) * -1, 
    K.zeros_like(win_odds)], axis=1)
    return -0.1 * K.mean(K.sum(gain_loss_vector * y_pred, axis=1))
    #return -1 * tf.reduce_mean(tf.reduce_sum(gain_loss_vector * y_pred, axis=1))
    # gain_loss_vector = K.concatenate([win_horse * (win_odds - 1) + (1 - win_horse) * -1, -> profit or loss of winning/losing on a bet of £1
    # K.zeros_like(win_odds)], axis=1) -> this is padding as y_true is longer then y_pred

get_custom_objects().update({"custom_loss_function": custom_loss_function})