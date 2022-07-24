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
            self.path: str = "static/classifier.onnx"
            self.size: int = 768
            self.labels: dict = json.load(open("static/labels_cls.json", "r"))
            self.MEAN: list = [0.485, 0.456, 0.406]
            self.STD: list  = [0.229, 0.224, 0.225]
        
        elif re.match(r"^detect$", self.infer_type, re.IGNORECASE):
            self.path: str = "static/detector.onnx"
            self.size: int = 416
            self.labels: dict = json.load(open("static/labels_det.json", "r"))
        
        elif re.match(r"^segment$", self.infer_type, re.IGNORECASE):
            self.path: str = "static/segmenter.onnx"
            self.size: int = 520
            self.labels: dict = json.load(open("static/labels_seg.json", "r"))
            self.MEAN: list = [0.485, 0.456, 0.406]
            self.STD: list  = [0.229, 0.224, 0.225]
        
        elif re.match(r"^face$", self.infer_type, re.IGNORECASE):
            self.model = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    def setup(self):
        model = onnx.load(self.path)
        onnx.checker.check_model(model)
        self.ort_session = ort.InferenceSession(self.path)
    
    def preprocess_1(self, image) -> np.ndarray:
        image = image / 255
        image = cv2.resize(src=image, dsize=(self.size, self.size), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)
        for i in range(image.shape[0]):
            image[i, :, :] = (image[i, :, :] - self.MEAN[i]) / self.STD[i]
        image = np.expand_dims(image, axis=0)
        return image.astype("float32")
    
    def preprocess_2(self, image: np.ndarray) -> np.ndarray:
        h, w, _ = image.shape
        scale = min(self.size / w, self.size / h)

        nh, nw = int(h * scale), int(w * scale)
        hh: int = (self.size - nh) // 2
        ww: int = (self.size - nw) // 2

        image = cv2.resize(src=image, dsize=(nw, nh), interpolation=cv2.INTER_AREA)
        new_image = np.ones((self.size, self.size, 3), dtype=np.uint8) * 128
        
        if abs(nh-(self.size - 2*hh)) == 1: new_image[hh:self.size-hh-1, ww:self.size-ww, :] = image
        elif abs(nw-(self.size - 2*ww)) == 1: new_image[hh:self.size-hh, ww:self.size-ww-1, :] = image
        else: new_image[hh:self.size-hh, ww:self.size-ww, :] = image

        new_image = new_image.transpose(2, 0, 1).astype("float32")
        new_image /= 255
        new_image = np.expand_dims(new_image, axis=0)
        return new_image

    def infer(self, image: np.ndarray):

        if re.match(r"^classify$", self.infer_type, re.IGNORECASE):
            image = self.preprocess_1(image)
            input = {self.ort_session.get_inputs()[0].name : image}

            return self.labels[str(np.argmax(self.ort_session.run(None, input)))].split(",")[0].title()
        
        elif re.match(r"^detect$", self.infer_type, re.IGNORECASE):
            image_h, image_w, _ = image.shape
            image = self.preprocess_2(image=image)

            input = {
                self.ort_session.get_inputs()[0].name : image,
                self.ort_session.get_inputs()[1].name : np.array([image_h, image_w], dtype=np.float32).reshape(1, 2),
            }
            
            boxes, scores, indices = self.ort_session.run(None, input)

            out_boxes, out_scores, out_classes = [], [], []

            if len(indices[0]) != 0:
                for idx_ in indices[0]:
                    out_classes.append(idx_[1])
                    out_scores.append(scores[tuple(idx_)])
                    idx_1 = (idx_[0], idx_[2])
                    out_boxes.append(boxes[idx_1])
                
                x1, y1, x2, y2 = int(out_boxes[0][1]), int(out_boxes[0][0]), int(out_boxes[0][3]), int(out_boxes[0][2])
                
                return self.labels[str(out_classes[0])], out_scores[0], (x1, y1, x2, y2)
            return None, None, None
    
        elif re.match(r"^segment$", self.infer_type, re.IGNORECASE):
            detected_labels = []
            h, w, _ = image.shape
            
            image = self.preprocess_1(image)

            input = {self.ort_session.get_inputs()[0].name : image}
            result = self.ort_session.run(None, input)
            class_index_image = np.argmax(result[0].squeeze(), axis=0)
            disp_image = cv2.resize(src=segmenter_decode(class_index_image), dsize=(w, h), interpolation=cv2.INTER_AREA)
            
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