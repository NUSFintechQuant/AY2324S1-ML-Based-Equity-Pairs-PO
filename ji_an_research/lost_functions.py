# Custom loss functions
import keras.backend as K
import tensorflow as tf

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def mae(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true), axis=-1)


def mape(y_true, y_pred):
    return 100. * K.mean(K.abs((y_pred - y_true) / K.clip(K.abs(y_true), K.epsilon(), None)), axis=-1)

def huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    is_small_error = K.abs(error) <= delta
    squared_loss = 0.5 * K.square(error)
    linear_loss = delta * (K.abs(error) - 0.5 * delta)
    return K.mean(tf.where(is_small_error, squared_loss, linear_loss))



loss_functions = [rmse, mae, mape, huber_loss]
loss_names = ["RMSE", "MAE", "MAPE", "Huber Loss"]

