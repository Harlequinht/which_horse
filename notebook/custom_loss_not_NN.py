def custom_loss_function_not_nn(y_true, y_pred):
    return -0.1 * K.mean(K.sum(gain_loss_vector * y_pred, axis=1))