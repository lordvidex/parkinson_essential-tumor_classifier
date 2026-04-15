### 1. Yandex Application and Token
To fetch items from Yandex with the `yadisk` python lib, and also update files to disk, it's necessary to create a yandex application.

#### 1.1. Generate YA_TOKEN
- Create a Yandex App to get client_id and client_secret: https://oauth.yandex.com/client/new
- Use the following URL to generate the token (replace client_id).
```
https://oauth.yandex.com/owl/authorize/error?response_type=token&client_id=954552f44e2a4e69ade4abf5ecc8bf2a&redirect_uri=https://oauth.yandex.ru/verification_code
```

### 2.  Environment preparation

#### 2.1. Create a virtual environment and install needed dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### 2.2 Secrets 
Create .env file and add the following line:
```env
YA_TOKEN=your_generated_token
SOURCE_PUBLIC_URL=<url of yandex disk public folder>
```

### 3. Running the program
### 3.1 Running partial jobs.
To handle the large size of patient files, we use a Python script that supports parallel execution via limit and offset parameters, allowing multiple script instances to process data concurrently.

#### 3.1.1 python script
```bash

source .venv/bin/activate # no need for imports because venv contains all packages needed
export $(grep -v '^#' .env | xargs) # load .env file
# change the values of offset, limit and metadata-path.
python3 ./yadisk_dicom_to_png.py --offset=0 --limit=10 --metadata-path=./artifacts/Загрузки/MRT_PNGs/metadata-1.csv
```
#### 3.1.2 [ALTERNATIVE] Jupyter notebook
- set the parameters of LIMIT and OFFSET at the top of the notebook

```bash
source .venv/bin/activate
# run each cell one by one
```
### 3.2 combining metadata files
To combine all intermediate metadata files for different jobs, `merge_metadata.py` can be run to combine them into a single metadata.csv file.
```bash
source .venv/bin/activate
chmod +x ./merge_metadata.py
python3 ./merge_metadata.py --metadata-dir=./artifacts/Загрузки/MRT_PNGs --output=metadata.csv
``` 