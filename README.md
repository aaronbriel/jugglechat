![JuggleChat](https://github.com/aaronbriel/jugglechat/blob/master/logo.png?raw=true)

An Eclectic and Malleable Multi-Chatbot Framework.

![JuggleChat](https://github.com/aaronbriel/jugglechat/blob/master/architecture.png?raw=true)

## Installation
Ubuntu:

	sudo apt install python3
	sudo apt install python3-pip
	sudo apt install virtualenv (or sudo apt install python3-virtualenv)

Clone repo and install dependencies:

    git clone https://github.com/aaronbriel/jugglechat.git
    cd jugglechat
    make setup
    
Install redis.

    sudo apt install redis

Install docker.

    sudo apt install docker

Update .env BASE_PATH ie "/home/aaron/jugglechat/"

## Configuration

### Configure Celery for Redis instance:
See config.py: BROKER_URL, BACKEND_URL

### Start Celery:
    make start_celery
    
View redis (celery task) data in browser:

    npm install -g redis-commander
    redis-commander

### Starting rasa server: 
    make start_rasa
    
### Starting haystack (QA and FAQ chatbots):
Start docker:

    sudo systemctl start docker
    
Start docker container:

    make elastic_start

Store FAQ and QA retrieval content into ElasticSearch:
    
    make store_faq 
    make store_faq_experiment       
    make store_qa
    make store_qa_experiment
    
You may need to run the following to resolve a Permission Denied issue on Ubuntu:

    sudo chmod 777 /tmp/tika*
    
If you run into the following error you will need to either increase disc size or free up the index:

    elasticsearch.exceptions.AuthorizationException: AuthorizationException(403, 'cluster_block_exception', 'index [qa_
    default] blocked by: [FORBIDDEN/12/index read-only / allow delete (api)];')
    
On Ubuntu, you also may need to do the following, then log out and back in:

	sudo groupadd docker
	sudo usermod -aG docker ${USER} 
	
A viable solution can be found here: 
https://stackoverflow.com/questions/50609417/elasticsearch-error-cluster-block-exception-forbidden-12-index-read-only-all

For example: 

    curl -XPUT -H "Content-Type: application/json" https://localhost:9200/_all/_settings -d '{"index.blocks.read_only_allow_delete": null}'


View content of ElasticSearch (default port: 9200, default index_names: faq_default, qa_default):

    curl https://localhost:PORT/INDEX_NAME/_search?pretty
Ex:

    curl https://localhost:9200/faq/_search?pretty
    
Start Haystack REST API:

    make start_haystack

Do sanity check of haystack:
    
    curl --request POST --url 'http://127.0.0.1:8000/models/1/doc-faq' --data '{"questions": ["can covid19 pass to pets?"]}' 

View API documentation: http://127.0.0.1:8000/docs

### Calling celery task that passes data to rasa core:
    from celery_app.tasks import call_rasa
    result = call_rasa_core.delay("hello")
    result.get()
    
### Calling celery task that passes data to rasa nlp:
    from celery_app.tasks import call_rasa
    result = call_rasa_nlp.delay("hello")
    result.get()

### Calling celery task that passes data to haystack faq:
    from celery_app.tasks import call_faq
    result = call_faq.delay("How is the virus spreading?")
    result.get()
    
### Calling celery task that passes data to haystack qa:
    from celery_app.tasks import call_qa
    result = call_qa.delay("Where was COVID19 first discovered?")
    result.get()
    
### Calling celery task that does single chat with dialogpt:
    from celery_app.tasks import call_convai
    result = call_convai.delay("Who is mort?")
    result.get()

### Calling celery task that does Abstractive Summarization:
    from celery_app.tasks import call_summarizer
    result = call_summarizer.delay("Lets do a summary of this overly long sentence please.")
    result.get()
    
### Calling celery tasks via flower API calls (expects rasa, haystack, elasticsearch, celery, and flower to be running):
    curl -X POST -u user1:password1 -d '{"args":["can covid19 pass to pets?"]}' http://localhost:5555/api/task/apply/celery_app.tasks.call_rasa_nlp
    curl -X POST -u user1:password1 -d '{"args":["can covid19 pass to pets?"]}' http://localhost:5555/api/task/apply/celery_app.tasks.call_qa
    curl -X POST -u user1:password1 -d '{"args":["how long does the coronavirus survive on surfaces"]}' http://localhost:5555/api/task/apply/celery_app.tasks.call_rasa_nlp
    curl -X POST -u user1:password1 -d '{"args":["how long does the coronavirus survive on surfaces"]}' http://localhost:5555/api/task/apply/celery_app.tasks.call_faq
    curl -X POST -u user1:password1 -d '{"args":["can I get a summary of the lesson?"]}' http://localhost:5555/api/task/apply/celery_app.tasks.call_rasa_nlp
    curl -X POST -u user1:password1 -d '{"args":["can I get a summary of the lesson?"]}' http://localhost:5555/api/task/apply/celery_app.tasks.call_summarizer

    
## Configuring production instance

### Setting up supervisor

Follow: https://www.digitalocean.com/community/tutorials/how-to-install-and-manage-supervisor-on-ubuntu-and-debian-vps

First confirm redis is installed and running for the celery command.

Install supervisor:

    sudo apt install supervisor

Copy supervisord conf file to local installation:

    sudo cp supervisor/supervisord.conf /etc/supervisor

Start supervisor:

    sudo supervisord

Restart ctl to reflect changes:

    sudo supervisorctl update
    sudo supervisorctl reload

Check that running:

    sudo supervisorctl

view log for celery
vi /home/aaron/jugglechat/celery_app/celery_app.log

kill all celery workers:

    sudo pkill -f "celery_app worker"

### Exposing Flower API on GCP

1. Reserve a static IP and assign to VM. 

2. Allow HTTP traffic

3. Whitelist jugglechat-experiment GCP cloud app URL

### Running a Control Group for the Experiment

## Experiment Configuration and Deployment

For the non-chat control group, simply complete step 4 with experimentalGroup set to "control". For the rest, 
follow all listed steps.

1. Modify celery_app > tasks.py call_rasa_nlp to pass in control_group (ie "qa" or "faq") to Allocator instantiation. This 
forces allocator to return that control group every time. There is no need for control_group with JuggleChat, as it is 
the default.

2. For QA or FAQ, modify supervisord.conf haystack variables to RUN_TYPE=experiment, READER_MODEL_PATH="models/qa_experiment" . 
The former  results in the elastic_search index used for retrieval to either be the faq_experiment (full FAQ dataset) or the 
qa_experiment and the latter points the qa chatbot's model path to the full QA set. For JuggleChat use RUN_TYPE=default,
READER_MODEL_PATH="models/qa_default"

NOTE: For the added experimental run to get results with deepset's extractive-QA model, RUN_TYPE=deepset, 
READER_MODEL_PATH="deepset/roberta-base-squad2-covid", with control_group="qa" for (1)

3. Restart all supervisor processes:


    ```
    sudo supervisorctl
    restart all
    ```

4. In jugglechat-experiment > client > src > index.tsx change experimentalGroup to the experimental group of:
jugglechat (JuggleChat framework), qa (QA bot only), faq (FAQ bot only), or control (no chatbot). Update the 
completionCode if needed.

5. Rebuild the jugglechat-experiment app by deleting all files under build then rebuilding:


    ```
    cd client
    npm run build
    cd ..
    npm run build
    ```
    
6. Redeploy the jugglechat-experiment app:


    ```
    gcloud app deploy
    ```

NOTE: The Worker IDs extracted by jugglechat-experiment > client/src/pages/GetId.tsx are stored in the worker_ids 
database table and can be exported to use for exclusion lists in subsequent experiment runs.

## Experiment Data Processing and Visualization

    source .venv/bin/activate 
    cd experiment

Generates results.json:

    python data_processing.py

Generates accuracy_and_usefulness.png and quiz_scores.png and (optionally) wordcloud images:

    python data_visualization.py