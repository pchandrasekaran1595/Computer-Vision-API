### **Computer Vision API using FastAPI**

<br>

1. Install Python
2. Run `pip install virtualenv`
3. Run `make-env.bat` or `make-env-3.9.bat`
4. Run `start-api-server.bat` (or setup `.vscode`).
5. The API will now be served at `http://127.0.0.1:10000`

**OR**


1. Pull the docker image using `docker pull pchandrasekaran1595/cv-api-fastapi` (Uses Python-3.8)
2. Run `docker-run.bat`. 
3. The API will now be served at `http://127.0.0.1:10000`

<br>

**Endpoints**

1. `/classify` - returns highest confidence prediction label
2. `/detect` &nbsp;&nbsp;&nbsp; - returns highest confidence bounding box and associated label
3. `/segment` &nbsp; - returns list of labels and base64 encoded image data

<br>

**Deployed on heroku at https://pcs-cv-api.herokuapp.com (Uses Python-3.9.13)**
