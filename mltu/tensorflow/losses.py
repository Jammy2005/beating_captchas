# # import tensorflow as tf


# # class CTCloss(tf.keras.losses.Loss):
# #     """ CTCLoss objec for training the model"""
# #     def __init__(self, name: str = "CTCloss") -> None:
# #         super(CTCloss, self).__init__()
# #         self.name = name
# #         self.loss_fn = tf.keras.backend.ctc_batch_cost

# #     def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None) -> tf.Tensor:
# #         """ Compute the training batch CTC loss value"""
# #         batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
# #         input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
# #         label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

# #         input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
# #         label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

# #         loss = self.loss_fn(y_true, y_pred, input_length, label_length)

# #         return loss

# import tensorflow as tf


# class CTCloss(tf.keras.losses.Loss):
#     """CTCLoss object for training the model."""
#     def __init__(self, name: str = "CTCloss", reduction: str = tf.keras.losses.Reduction.NONE, **kwargs) -> None:
#         # Pass the `reduction` argument to the base class
#         super(CTCloss, self).__init__(name=name, reduction=reduction, **kwargs)
#         self.loss_fn = tf.keras.backend.ctc_batch_cost

#     def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
#         """Compute the training batch CTC loss value."""
#         batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
#         input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
#         label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

#         input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
#         label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

#         loss = self.loss_fn(y_true, y_pred, input_length, label_length)
#         return loss

#     def get_config(self):
#         """Serialize the configuration of the loss."""
#         base_config = super(CTCloss, self).get_config()
#         return base_config
import tensorflow as tf

class CTCloss(tf.keras.losses.Loss):
    """Custom CTCLoss class."""
    def __init__(self, name="CTCloss", reduction=tf.keras.losses.Reduction.NONE, **kwargs):
        super(CTCloss, self).__init__(name=name, reduction=reduction, **kwargs)
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        """Compute the CTC loss."""
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        return loss

    def get_config(self):
        """Return the config for serialization."""
        config = super(CTCloss, self).get_config()
        return config
