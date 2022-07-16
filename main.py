from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from starlette.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from static.utils import Model, decode_image, encode_image_to_base64

VERSION = "1.0.0"

class Image(BaseModel):
    imageData: str


STATIC_PATH = "static"

origins = [
    "http://localhost:10001",
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
        "statusCode" : 200,
        "version" : VERSION,
    })


@app.get("/version")
async def get_version():
    return JSONResponse({
        "statusText" : "Version Fetch Successful",
        "statusCode" : 200,
        "version" : VERSION,
    })


@app.get("/classify")
async def get_classifiy_infer():
    return JSONResponse({
        "statusText" : "Classification Inference Endpoint",
        "statusCode" : 200,
        "version" : VERSION,
    })


@app.get("/detect")
async def get_detect_infer():
    return JSONResponse({
        "statusText" : "Detection Inference Endpoint",
        "statusCode" : 200,
        "version" : VERSION,
    })


@app.get("/segment")
async def get_segment_infer():
    return JSONResponse({
        "statusText" : "Segmentation Inference Endpoint",
        "statusCode" : 200,
        "version" : VERSION,
    })


@app.post("/classify")
async def post_classify_infer(image: Image):
    _, image = decode_image(image.imageData)

    model = Model(infer_type="classify")
    model.setup()
    label = model.infer(image=image)

    return JSONResponse({
        "statusText" : "Classification Inference Complete",
        "statusCode" : 200,
        "label" : label,
    })


@app.post("/detect")
async def post_detect_infer(image: Image):
    _, image = decode_image(image.imageData)

    model = Model(infer_type="detect")
    model.setup()
    label, score, box = model.infer(image)

    if label is not None:
        return JSONResponse({
            "statusText" : "Tiny Yolo V3 Inference Inference Complete",
            "statusCode" : 200,
            "label" : label,
            "score" : str(score),
            "box" : box,
        })
    else:
        return JSONResponse({
            "statusText" : "No Detections",
            "statusCode" : 500,
        })


@app.post("/segment")
async def post_segment_infer(image: Image):
    _, image = decode_image(image.imageData)

    model = Model(infer_type="segment")
    model.setup()

    segmented_image, labels = model.infer(image)
    segmented_image_data = encode_image_to_base64(header="data:image/png;base64", image=segmented_image)
    
    return JSONResponse({
        "statusText" : "Segmentation Inference Complete",
        "statusCode" : 200,
        "labels" : labels,
        "imageData" : segmented_image_data,
    })
