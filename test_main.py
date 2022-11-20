from fastapi.testclient import TestClient
from main import app, VERSION

client = TestClient(app=app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "statusText" : "Root Endpoint of Computer Vision API",
        "statusCode" : 200,
        "version" : VERSION,
    }
 

def test_get_version():
    response = client.get("/version")
    assert response.status_code == 200
    assert response.json() == {
        "statusText" : "Computer Vision API Version Fetch Successful",
        "statusCode" : 200,
        "version" : VERSION,
    }
    
      
def test_get_classify_infer():
    response = client.get("/classify")
    assert response.status_code == 200
    assert response.json() == {
        "statusText" : "Classification Inference Endpoint",
        "statusCode" : 200,
        "version" : VERSION,
    }


def test_get_detect_infer():
    response = client.get("/detect")
    assert response.status_code == 200
    assert response.json() == {
        "statusText" : "Detection Inference Endpoint",
        "statusCode" : 200,
        "version" : VERSION,
    }


def test_get_face_detect_infer():
    response = client.get("/face")
    assert response.status_code == 200
    assert response.json() == {
        "statusText" : "Face Detection Inference Endpoint",
        "statusCode" : 200,
        "version" : VERSION,
    }


def test_get_segment_infer():
    response = client.get("/segment")
    assert response.status_code == 200
    assert response.json() == {
        "statusText" : "Segmentation Inference Endpoint",
        "statusCode" : 200,
        "version" : VERSION,
    }
