# Introduction
The walkthrough video is available [online](https://smu-my.sharepoint.com/:v:/g/personal/tingzhang_2019_phdcs_smu_edu_sg/Eb2E0ykaqsFInt3p835L_bEBUlmzEwSXExIOL25OXJbYDA?e=cJ3W7P).

# Model
## Fine-tuning BART
Run the script in `src/run_summarization.py`.

## Baseline: iTAPE
Please refer to the original [repository](https://github.com/imcsq/iTAPE).

# Evaluation
## Human Evaluation
1. Sample 30 issues
- [The sampled file for evaluation](./human-evaluation/sampled-for-evaluation.csv).
- [The sampled ground truth file](./human-evaluation/sampled-ground.csv).

2. Analyze the result
Run the script in `src/analyze_human_eval.py`.


# iTiger Installation
## Pre-requisites:

1. Clone the project at <path_to_directory>
2. Download model from [here](https://smu-my.sharepoint.com/:u:/g/personal/ivanairsan_smu_edu_sg/EaxUXHAlwGtLsKTRSmse48IBq63hI4l-IjrXGLVMj-6Y-A?e=qH6DsU)
3. Extract and put the `model` folder under <path_to_directory>/iTiger (same level as `src`)

## Usage

### iTiger Backend:
1. Build the image: 

    ```docker build -t itiger .```

2. Start a container: 

    ```docker run --name itiger-svc -p 8000:8000 -v <path_to_directory>/iTiger:/app -itd itiger```

3. (Optional) Publish local endpoint using ngrok to serve multiple client:

    ```docker run --rm -it --link itiger-svc wernight/ngrok ngrok http -region=eu itiger-svc:8000```


### iTiger UI
3. Install Tampermonkey on [Chrome](https://chrome.google.com/webstore/detail/tampermonkey/dhdgffkkebhmkfjojejmpbldmpobfkfo?hl=en) / [Opera](https://addons.opera.com/en/extensions/details/tampermonkey-beta/) / [Safari](https://www.tampermonkey.net/?browser=safari)

3. Add [GithubIssueTItleRecommender.user.js](GithubIssueTItleRecommender.user.js) to Tampermonkey extension

   (WARNING: you need to adjust the `base_url` variable in the script, setting it to your deployed server's endpoint)

   If iTiger's backend is deployed in your local machine, change the `base_url` to localhost:8000. 
   
   Otherwise, use the ngrok host as `base_url`.



### NOTES:
* We have an available server for testing purpose at `http://a359-202-161-44-1.eu.ngrok.io`

    endpoint: `<URL>/predict?text=<issue src>`
    
    example: `http://a359-202-161-44-1.eu.ngrok.io/predict?text="windows not closed properly"`
