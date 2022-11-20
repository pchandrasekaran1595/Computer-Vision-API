from fastapi import FastAPI, status
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from starlette.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from static.utils import Model, decode_image, encode_image_to_base64


classifier = Model(infer_type="classify")
classifier.setup()

detector = Model(infer_type="detect")
detector.setup()

segmenter = Model(infer_type="segment")
segmenter.setup()

face = Model(infer_type="face")


VERSION: str = "1.0.0"
STATIC_PATH: str = "static"


class Image(BaseModel):
    imageData: str


origins = [
    "http://localhost:4041",
    "https://pcs-cv-web-client.netlify.app"
]


app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_PATH), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return JSONResponse({
        "statusText" : "Root Endpoint of Computer Vision API",
        "statusCode" : status.HTTP_200_OK,
        "version" : VERSION,
    })


@app.get("/version")
async def get_version():
    return JSONResponse({
        "statusText" : "Computer Vision API Version Fetch Successful",
        "statusCode" : status.HTTP_200_OK,
        "version" : VERSION,
    })


@app.get("/classify")
async def get_classifiy_infer():
    return JSONResponse({
        "statusText" : "Classification Inference Endpoint",
        "statusCode" : status.HTTP_200_OK,
        "version" : VERSION,
    })


@app.get("/detect")
async def get_detect_infer():
    return JSONResponse({
        "statusText" : "Detection Inference Endpoint",
        "statusCode" : status.HTTP_200_OK,
        "version" : VERSION,
    })


@app.get("/face")
async def get_face_detect_infer():
    return JSONResponse({
        "statusText" : "Face Detection Inference Endpoint",
        "statusCode" : status.HTTP_200_OK,
        "version" : VERSION,
    })


@app.get("/segment")
async def get_segment_infer():
    return JSONResponse({
        "statusText" : "Segmentation Inference Endpoint",
        "statusCode" : status.HTTP_200_OK,
        "version" : VERSION,
    })


@app.post("/classify")
async def post_classify_infer(image: Image):
    _, image = decode_image(image.imageData)

    label = classifier.infer(image=image)

    return JSONResponse({
        "statusText" : "Classification Inference Complete",
        "statusCode" : status.HTTP_200_OK,
        "label" : label,
    })


@app.post("/detect")
async def post_detect_infer(image: Image):
    _, image = decode_image(image.imageData)

    label, score, box = detector.infer(image)

    if label is not None:
        return JSONResponse({
            "statusText" : "Detection Inference Complete",
            "statusCode" : status.HTTP_200_OK,
            "label" : label,
            "score" : str(score),
            "box" : box,
        })
    else:
        return JSONResponse({
            "statusText" : "No Detections",
            "statusCode" : 500,
        })


@app.post("/face")
async def post_face_detect_infer(image: Image):
    _, image = decode_image(image.imageData)

    face_detections_np = face.infer(image)

    if len(face_detections_np) > 0:
        face_detections: list = []
        for (x, y, w, h) in face_detections_np:
            face_detections.append([int(x), int(y), int(w), int(h)])
        
        return JSONResponse({
            "statusText" : "Face Detection Complete",
            "statusCode" : status.HTTP_200_OK,
            "face_detections" : face_detections,
        })
    else:
        return JSONResponse({
            "statusText" : "No Detections",
            "statusCode" : status.HTTP_500_INTERNAL_SERVER_ERROR,
        })


@app.post("/segment")
async def post_segment_infer(image: Image):
    _, image = decode_image(image.imageData)

    segmented_image, labels = segmenter.infer(image)
    segmented_image_data = encode_image_to_base64(header="data:image/png;base64", image=segmented_image)
    
    return JSONResponse({
        "statusText" : "Segmentation Inference Complete",
        "statusCode" : status.HTTP_200_OK,
        "labels" : labels,
        "imageData" : segmented_image_data,
    })
