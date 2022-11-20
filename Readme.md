### **Computer Vision API using FastAPI**

<br>

1. Install Python
2. Run `pip install virtualenv`
3. Run `make-env.bat` or `make-env-3.9.bat`
4.  Setup `.vscode` `launch.json` and run

**OR**


1. Pull the docker image using `docker pull pchandrasekaran1595/cv-api` or `docker pull prashanthacsq/cv-api` (Uses Python-3.8)
2. Run `docker-run.bat`. 
3. Thee API will now be served at `http://localhost:4040` or `http://127.0.0.1:4040` or `http://<IP>:4040`

<br>

**Endpoints**

1. `/classify` - returns highest confidence prediction label
2. `/detect` &nbsp;&nbsp;&nbsp; - returns highest confidence bounding box and associated label
2. `/face` &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - returns detection bounding boxes
3. `/segment` &nbsp; - returns list of labels and base64 encoded image data

<br>
