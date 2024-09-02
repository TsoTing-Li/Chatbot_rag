
# Chatbot Service

This Repository provides a Chatbot include RAG function and a simple frontend.

## Pre-requirement
### Download model file
* Build environment
    ```bash
    python3 -m venv download_model
    source download_model/bin/activate
    pip install --upgrade huggingface_hub python-dotenv
    ```

* Set your Hugging Face token in the appropriate variable. [Get serverless api token](https://huggingface.co/docs/api-inference/index) 
    1. Create the `.env` file.
    2. Enter `HF_API_TOKEN={your_huggingface_token}`

* Execute `download_file.py`, all files will be stored in `hf_models` folder
* Exit environment
    ```bash
    deactivate
    ```

## User Usage
### Start the Service

To start the Chatbot service, use Docker Compose with the provided configuration file:

```bash
docker compose -f ./compose.yaml up -d
```

### Shutdown service
```bash
docker compose down
```

### WebUI
> **WebUI url: http://192.168.55.14:8501**

1. Open WebUI with url
2. Register user name and department
3. Then start asking questions

## Advanced
###  Update vector database
> **Only support call api now**
* Prepare your PDF file (any structure)
    ```text
    folder/
        /data1.pdf
        /data2.pdf
        ...
        subfolder/
            /sub_data1.pdf
            /sub_data2.pdf
            ...
    ```

* Upload file
    ```bash
    # Request
        curl -X 'POST' \
            'http://192.168.55.13:8001/update/' \
            -H 'accept: application/json' \
            -H 'Content-Type: multipart/form-data' \
            -F 'files=@{filename1};type=application/pdf' \
            -F 'files=@{filename2};type=application/pdf' \
            -F ...
            -F 'username={username}' \
            -F 'department={department}'

    # Response
        {"task_id": {task_id}}
    ```
* Embedding PDF
    ```bash
    # Request
        curl -X 'POST' \
            'http://192.168.55.15:8777/embed/doc/' \
            -H 'accept: application/json' \
            -H 'Content-Type: application/json' \
            -d '{
            "data_folder": {task_id},
            "recreate": {is_recreate_DB}
        }'

    # Response
        {"task_id": {task_id}}
    ```
* Listen task status
    ```bash
    # Upload file
        ws://192.168.55.13:8001/ws/{task_id}
    # Embed PDF
        ws://192.168.55.15:8777/ws/{task_id}
    ```