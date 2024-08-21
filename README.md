
# Chatbot Service

This README provides instructions on how to start, use, and update the Chatbot service.

## User Usage

### Pre-Start

Before starting the Chatbot service, you need to set the corresponding Hugging Face token in the `tools/setting.py` file.

1. Open the `tools/setting.py` file.
2. Set your Hugging Face token in the appropriate variable. [Get serverless api token](https://huggingface.co/docs/api-inference/index)

### Start the Service

To start the Chatbot service, use Docker Compose with the provided configuration file:

```bash
docker compose -f ./compose.yaml up -d

# Operate With WebUI : http://127.0.0.1:8000/static/index.html
# Open API docs : http://127.0.0.1:8000/docs
```

### Shutdown service
```bash
docker compose down
```

###  How to update vector database?
* Follow [Development readme](/docs/README.DEV.md) to lunch environment.
* Follow [Vectorization](/docs/Vectorization.md) to update new data to vector database.
    

## Other
* [Development readme](/docs/README.DEV.md) 
* [Update map](/docs/UPDATE.md)
* [Todo](/docs/TODO.md)

