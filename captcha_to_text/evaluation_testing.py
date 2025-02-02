import cv2
import typing
import numpy as np

from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer

class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, image: np.ndarray):
        image = cv2.resize(image, self.input_shapes[0][1:3][::-1])

        image_pred = np.expand_dims(image, axis=0).astype(np.float32)

        preds = self.model.run(self.output_names, {self.input_names[0]: image_pred})[0]

        text = ctc_decoder(preds, self.char_list)[0]

        return text

if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm
    from mltu.configs import BaseModelConfigs

    configs = BaseModelConfigs.load("/Users/ahmadjamshaid/Desktop/internship/captcha/beating_captcha/Beating-Captchas/Models/02_captcha_to_text/202501220907/configs.yaml") #202212211205

    model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)

    # df = pd.read_csv("/Users/ahmadjamshaid/Desktop/internship/captcha/beating_captcha/Beating-Captchas/Models/02_captcha_to_text/202501220907/val.csv").values.tolist()   # val set
    df = pd.read_csv("/Users/ahmadjamshaid/Desktop/internship/captcha/beating_captcha/Beating-Captchas/Datasets/test2.csv").values.tolist()     # test set

    accum_cer = []
    for image_path, label in tqdm(df):
        image = cv2.imread(image_path.replace("\\", "/"))

        prediction_text = model.predict(image)

        cer = get_cer(prediction_text, label)
        print(f"Image: {image_path}, Label: {label}, Prediction: {prediction_text}, CER: {cer}")

        accum_cer.append(cer)

    print(f"Average CER: {np.average(accum_cer)}")
    
    # Test a specific image
    test_image_path = "/Users/ahmadjamshaid/Desktop/captcha-images-v3 2/Xmo5hQ.png"  # Replace with your specific image path
    image = cv2.imread(test_image_path)

    if image is not None:
        prediction_text = model.predict(image)
        print(f"Specific Image Prediction: {prediction_text}")
    else:
        print(f"Error: Could not read the image at {test_image_path}")