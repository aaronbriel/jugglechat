.PHONY: help

help: ## This help.
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.DEFAULT_GOAL := help

# -----------------------------------------------
setup: py-scrub ## Create a python virtualenv and install requirements.
    # Commenting virtualenv install due to issues it can cause on linux environments
	# pip3 install virtualenv
	# NOTE: 1/25/2021 need to use python3.8 for Rasa requirement
	virtualenv -p python3 .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

install: clean ## Install python requirements.
	. .venv/bin/activate && pip install -r requirements.txt

py-scrub: clean ## Delete virtualenv and egg.
	find . -name '*.egg-info' -type d -prune -exec rm -rf '{}' '+'
	find . -name '.venv' -type d -prune -exec rm -rf '{}' '+'

clean: ## Delete python cache.
	find . -name '*.pyc' -exec rm '{}' ';'
	find . -name '__pycache__' -type d -prune -exec rm -rf '{}' '+'
	find . -name '.pytest_cache' -type d -prune -exec rm -rf '{}' '+'

# -----------------------------------------------
full_install: setup ## Install python requirements, starts redis, install/start elasticsearch and store retrieval documents. Expects redis, docker, and docker-compose to be installed
	redis-server /usr/local/etc/redis.conf
	cd chatbots/haystack/es_container && docker-compose up -d
	. .venv/bin/activate && python chatbots/haystack/faqbot.py --command store
	. .venv/bin/activate && python chatbots/haystack/faqbot.py --command store --run_type experiment
	. .venv/bin/activate && python chatbots/haystack/qabot.py --command store
	. .venv/bin/activate && python chatbots/haystack/qabot.py --command store --run_type experiment

# -----------------------------------------------
start_redis: ## Start redis backend
	# MAC (brew install redis). Starts on 127.0.0.1:6379
	# Start: redis-server /usr/local/etc/redis.conf
	# UBUNTU install: sudo apt install redis-server
	# More info: https://www.digitalocean.com/community/tutorials/how-to-install-and-secure-redis-on-ubuntu-18-04
	service redis-server start

stop_redis: ## Stop redis backend
	service redis-server stop

# -----------------------------------------------
start_rasa: ## Start rasa API
	# Ex: make start_rasa model=nlu-experiment.tar.gz
	. .venv/bin/activate && cd chatbots/rasa && rasa run --enable-api -m models/$(model)

train_rasa: ## Train RASA NLU intent model
	. .venv/bin/activate && cd chatbots/rasa && rasa train nlu

# -----------------------------------------------
start_celery: ## Start celery
	. .venv/bin/activate && python logo.py && celery -A celery_app worker --loglevel info --logfile=celery_app/celery_app.log

start_flower: ## Start flower (celery API), view at http://localhost:5555
	# NOTE: make start_celery first...
	. .venv/bin/activate && celery flower -A celery_app --address=127.0.0.1 --port=5555

# -----------------------------------------------
start_haystack: ## Start haystack API
	# Ex: make start_haystack
	make start_elastic
	. .venv/bin/activate && export RUN_TYPE=default && export READER_MODEL_PATH=models/qa_default && cd chatbots/haystack && gunicorn rest_api.application:app -b 0.0.0.0:8000 -k uvicorn.workers.UvicornWorker

start_haystack_experiment: ## Start haystack API but with experiment model and index
	# Ex: make start_haystack
	. .venv/bin/activate && export RUN_TYPE=experiment && export READER_MODEL_PATH=models/qa_experiment && cd chatbots/haystack && gunicorn rest_api.application:app -b 0.0.0.0:8000 -k uvicorn.workers.UvicornWorker

start_elastic: ## Starts ElasticSearch server (may need to use sudo on GCP, etc)
	# RUN WITHOUT PERSISTENT STORAGE: docker run -d -p 9200:9200 -e "discovery.type=single-node" elasticsearch:7.6.2
	. .venv/bin/activate && cd chatbots/haystack/es_container && docker-compose up -d

stop_elastic: ## Stops ElasticSearch server
	. .venv/bin/activate && docker stop haystack-image

delete_elastic: ## Delete data volumes when bringing down cluster
	. .venv/bin/activate && cd chatbots/haystack/es_container && docker-compose down -v

# -----------------------------------------------
prepare_faq: ## Prepare FAQ data
	. .venv/bin/activate && python chatbots/haystack/faqbot.py --command prepare_data

store_faq: ## Store data in elasticsearch for FAQ
	. .venv/bin/activate && python chatbots/haystack/faqbot.py --command store

store_faq_experiment: ## Store data in elasticsearch for FAQ experimental control group
	. .venv/bin/activate && python chatbots/haystack/faqbot.py --command store --run_type experiment

store_faq_deepset: ## Store deepset's COVID-QA question answer pairs for deepset followup test
	. .venv/bin/activate && python chatbots/haystack/faqbot.py --command store --run_type deepset

prepare_qa: ## Prepare QA data
	. .venv/bin/activate && python chatbots/haystack/qabot.py --command prepare_data

prepare_deepset_data: ## Prepare deepset QA data
	. .venv/bin/activate && python chatbots/haystack/qabot.py --command prepare_deepset_data --run_type deepset

store_qa: ## Store data in elasticsearch for QA
	. .venv/bin/activate && python chatbots/haystack/qabot.py --command store

store_qa_experiment: ## Store data in elasticsearch for QA
	. .venv/bin/activate && python chatbots/haystack/qabot.py --command store --run_type experiment

store_qa_deepset: ## Store deepset's COVID-QA contexts (extracted into txt files) in elasticsearch for deepset QA test
	. .venv/bin/activate && python chatbots/haystack/qabot.py --command store --run_type deepset

eval_faq: ## Evaluate model
	. .venv/bin/activate && python chatbots/haystack/faqbot.py --command eval

train_qa: ## Store data in elasticsearch for QA
	. .venv/bin/activate && python chatbots/haystack/qabot.py --command train

train_qa_experiment: ## Store data in elasticsearch for QA
	. .venv/bin/activate && python chatbots/haystack/qabot.py --command train --run_type experiment

ask_qa: ## Run model in interactive mode
    # Ex: make ask_qa question="'What is COVID 19'"
	. .venv/bin/activate && python chatbots/haystack/qabot.py --command ask --question $(question)

ask_faq: ## Run FAQ type question
    # Ex: make ask_faq question="'How is the virus spreading?'"
	. .venv/bin/activate && python chatbots/haystack/faqbot.py --command ask --question $(question)

create_intent_file_faq: ## Creates intent file for training Rasa NLU
	. .venv/bin/activate && python chatbots/haystack/faqbot.py --command create_intent_file

create_intent_file_qa: ## Creates intent file for training Rasa NLU
	. .venv/bin/activate && python chatbots/haystack/qabot.py --command create_intent_file

# -----------------------------------------------
preprocess_dialogpt: ## Converts JHU dataset to dialog and faq ammenable datasets
	. .venv/bin/activate && cd chatbots/dialogpt && python dialogpt.py --command preprocess

train_dialogpt: ## Run full training with default hyperparameters
	. .venv/bin/activate && cd chatbots/dialogpt && python dialogpt.py --command train

train_dialogpt_experiment: ## Run full training with experiment run_type
	. .venv/bin/activate && cd chatbots/dialogpt && python dialogpt.py --command train --run_type experiment

chat_dialogpt: ## Run dialogtp in interactive chat mode
	. .venv/bin/activate && cd chatbots/dialogpt && python dialogpt.py --command chat

# -----------------------------------------------
test: ## Run test file
	. .venv/bin/activate && python test.py