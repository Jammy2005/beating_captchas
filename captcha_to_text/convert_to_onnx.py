import tensorflow as tf
from mltu.tensorflow.losses import CTCloss
import keras
keras.config.enable_unsafe_deserialization()
import tf2onnx
import tensorflow as tf
model = tf.keras.models.load_model(
    "/Users/ahmadjamshaid/Desktop/internship/captcha/beating_captcha/Beating-Captchas/Models/02_captcha_to_text/202501220907/model.keras",
    # "/Users/ahmadjamshaid/Desktop/internship/captcha/beating_captcha/Beating-Captchas/Models/02_captcha_to_text/202501191031/model.keras",
    custom_objects={"CTCloss": CTCloss},
    compile=False
)

input_signature = [tf.TensorSpec(model.input_shape, tf.float32, name="input")]


# Convert to ONNX
onnx_output_path = "/Users/ahmadjamshaid/Desktop/internship/captcha/beating_captcha/Beating-Captchas/Models/model.onnx"
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature, output_path=onnx_output_path)

print(f"Model successfully converted to ONNX and saved at {onnx_output_path}")
