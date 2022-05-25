# Introduction

# How to use our tool

# Model
## Fine-tuning BART
Run the script in `src/run_summarization.py`

## Baseline: iTAPE
Please refer to the original repository.

# Evaluation
## Human Evaluation
1. Sample 30 issues

2. Analyze the result



# iTiger Installation
##Prerequisite:
    1. Clone the project at <path_to_directory>
    2. Download model from <TODO: model_link>
    3. Put the model under <path_to_directory>/iTiger (same directory as `src`)

## Usage

### iTiger Backend:
1. Build the image: 

    ```docker build -t issue-title-generator .```

2. Start a container: 

    ```docker run --name itiger-svc -p 8000:8000 -v <path_to_directory>/Issue-Title-Generator:/app -itd issue-title-generator```

3. -- Optional --
Publish local endpoint using ngrok to serve multiple client:

    ```docker run --rm -it --link itiger-svc wernight/ngrok ngrok http -region=eu itiger-svc:8000```


### iTiger UI
3. Install Tampermonkey on Chrome / Opera / Safari

3. Add `Github Issue TItle Recommender.user.js` to tampermonkey extension

   (WARNING: you need to adjust the `base_url` to your deployed server's endpoint)
   if iTiger's backend is deployed in your local machine, change the `base_url` to localhost:8000. Otherwise, use the ngrok host as `base_url`.



### NOTES:
* We have an available server for testing purpose at `http://a359-202-161-44-1.eu.ngrok.io`
    endpoint: `<URL>/predict?text=<issue src>`
    example: `http://a359-202-161-44-1.eu.ngrok.io/predict?text="windows not closed properly"`
