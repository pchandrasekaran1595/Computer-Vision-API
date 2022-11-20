import io
import re
import cv2
import json
import onnx
import base64
import numpy as np
import onnxruntime as ort

from PIL import Image

ort.set_default_logger_severity(3)

#####################################################################################################

class Model(object):
    def __init__(self, infer_type: str):
        self.infer_type: str = infer_type
        self.ort_session = None
        
        if re.match(r"^classify$", self.infer_type, re.IGNORECASE):
            self.path: str = "static/models/classifier.onnx"
            self.size: int = 768
            self.labels: dict = json.load(open("static/labels/labels_cls.json", "r"))
            self.MEAN: list = [0.485, 0.456, 0.406]
            self.STD: list  = [0.229, 0.224, 0.225]
        
        elif re.match(r"^detect$", self.infer_type, re.IGNORECASE):
            self.path: str = "static/models/detector.onnx"
            self.size: int = 640
            self.labels: dict = json.load(open("static/labels/labels_det.json", "r"))
        
        elif re.match(r"^segment$", self.infer_type, re.IGNORECASE):
            self.path: str = "static/models/segmenter.onnx"
            self.size: int = 520
            self.labels: dict = json.load(open("static/labels/labels_seg.json", "r"))
            self.MEAN: list = [0.485, 0.456, 0.406]
            self.STD: list  = [0.229, 0.224, 0.225]
        
        elif re.match(r"^face$", self.infer_type, re.IGNORECASE):
            self.model = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    def setup(self):
        model = onnx.load(self.path)
        onnx.checker.check_model(model)
        self.ort_session = ort.InferenceSession(self.path)
    
    def preprocess(self, image) -> np.ndarray:
        image = image / 255
        image = cv2.resize(src=image, dsize=(self.size, self.size), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)
        if not re.match(r"^detect$", self.infer_type, re.IGNORECASE):
            for i in range(image.shape[0]): image[i, :, :] = (image[i, :, :] - self.MEAN[i]) / self.STD[i]
        image = np.expand_dims(image, axis=0)
        return image.astype("float32")
    
    def infer(self, image: np.ndarray):

        if re.match(r"^classify$", self.infer_type, re.IGNORECASE):
            return self.labels[str(np.argmax(self.ort_session.run(None, {self.ort_session.get_inputs()[0].name : self.preprocess(image)})))].split(",")[0].title()
        
        elif re.match(r"^detect$", self.infer_type, re.IGNORECASE):
            image_h, image_w, _ = image.shape

            result = self.ort_session.run(None, {self.ort_session.get_inputs()[0].name : self.preprocess(image)})
            result = result[0][0]
    
            box = result[1:5]
            label_index = int(result[-2])
            score = result[-1]

            x1 = int(box[0] * image_w / self.size)
            y1 = int(box[1] * image_h / self.size)
            x2 = int(box[2] * image_w / self.size)
            y2 = int(box[3] * image_h / self.size)

            return self.labels[str(label_index)], score, (x1, y1, x2, y2)
    
        elif re.match(r"^segment$", self.infer_type, re.IGNORECASE):
            detected_labels = []
            image_h, image_w, _ = image.shape

            result = self.ort_session.run(None, {self.ort_session.get_inputs()[0].name : self.preprocess(image)})
            class_index_image = np.argmax(result[0].squeeze(), axis=0)
            disp_image = cv2.resize(src=segmenter_decode(class_index_image), dsize=(image_w, image_h), interpolation=cv2.INTER_AREA)
            
            class_indexes = np.unique(class_index_image)
            for index in class_indexes:
                if index != 0:
                    detected_labels.append(self.labels[str(index)].title())
            return disp_image, detected_labels
        
        elif re.match(r"^face$", self.infer_type, re.IGNORECASE):
            temp_image = cv2.cvtColor(src=image.copy(), code=cv2.COLOR_RGB2GRAY)
            detections = self.model.detectMultiScale(image=temp_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
            return detections

#####################################################################################################

def segmenter_decode(class_index_image: np.ndarray) -> np.ndarray:
    colors = np.array([(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                       (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                       (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                       (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

    r, g, b = np.zeros(class_index_image.shape, dtype=np.uint8), \
              np.zeros(class_index_image.shape, dtype=np.uint8), \
              np.zeros(class_index_image.shape, dtype=np.uint8)

    for i in range(21):
        indexes = (class_index_image == i)
        r[indexes] = colors[i][0]
        g[indexes] = colors[i][1]
        b[indexes] = colors[i][2]
    return np.stack([r, g, b], axis=2)

#####################################################################################################

def softmax(x) -> float:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def decode_image(imageData) -> np.ndarray:
    header, imageData = imageData.split(",")[0], imageData.split(",")[1]
    image = np.array(Image.open(io.BytesIO(base64.b64decode(imageData))))
    image = cv2.cvtColor(src=image, code=cv2.COLOR_BGRA2RGB)
    return header, image


def encode_image_to_base64(header: str="data:image/png;base64", image: np.ndarray=None) -> str:
    assert image is not None, "Image is None"
    _, imageData = cv2.imencode(".jpeg", image)
    imageData = base64.b64encode(imageData)
    imageData = str(imageData).replace("b'", "").replace("'", "")
    imageData = header + "," + imageData
    return imageData

#####################################################################################################