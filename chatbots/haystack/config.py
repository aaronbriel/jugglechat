import os

from box import Box
from dotenv import load_dotenv
import torch

load_dotenv()

BASE_PATH = os.getenv("BASE_PATH")

FAQ_DATA_PATH = BASE_PATH + 'chatbots/haystack/data/faq/'
FAQ_DATASET_FULL = 'faq-jhu_covid_qa-refined-cleaned.csv'
FAQ_DATASET_JUGGLECHAT = 'faq-jhu_covid_qa-refined-cleaned-FAQ.csv'
FAQ_DATASET_DEEPSET = 'deepset_COVID-QA.csv'

QA_DATA_PATH = BASE_PATH + 'chatbots/haystack/data/qa/'
QA_DATASET_FULL = 'squad-jhu_covid_qa-refined-cleaned.json'
QA_DATASET_JUGGLECHAT = 'squad-jhu_covid_qa-refined-cleaned-QA.json'
QA_DATASET_DEEPSET = 'deepset_COVID-QA.json'

SQUAD_TEMPLATE = BASE_PATH + 'chatbots/haystack/data/qa/squad_template.json'

def get_args(
        chatbot=None,
        run_type='default',
        command='train',
        # Default model to train for extractive QA
        model_name_or_path='deepset/roberta-base-squad2',
        train_filename=None,
        path_to_train_data=None,
        faq_dataset=FAQ_DATASET_JUGGLECHAT,
        qa_dataset=QA_DATASET_JUGGLECHAT,
        embedding_model='deepset/sentence_bert',
        no_ans_boost=None
):
    doc_dir = '{}chatbots/haystack/data/{}/{}/'.format(BASE_PATH, chatbot, run_type)
    output_dir = '{}chatbots/haystack/output/{}/{}'.format(BASE_PATH, chatbot, run_type)
    save_dir = '{}chatbots/haystack/models/{}_{}'.format(BASE_PATH, chatbot, run_type)

    # Set path to train data if not specified
    if not path_to_train_data:
        path_to_train_data = '{}chatbots/haystack/data/{}/'.format(BASE_PATH, chatbot)

    # Using trained model path in case where we are interacting or evaluating
    if 'train' not in command and 'store' not in command:
        model_name_or_path = save_dir
        # NOTE For saving embedding model locally see: https://github.com/deepset-ai/haystack/issues/149
        # embedding_model = 'models/{}'.format(run_type)
        no_ans_boost = -100

    # Using covid trained base model for experiment
    if 'experiment' in run_type:
        faq_dataset = FAQ_DATASET_FULL
        qa_dataset = QA_DATASET_FULL

    if run_type == 'deepset':
        model_name_or_path = 'deepset/roberta-base-squad2-covid'
        faq_dataset = FAQ_DATASET_DEEPSET

    if chatbot == 'qa':
        train_filename = qa_dataset
    elif chatbot == 'faq':
        train_filename = faq_dataset

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # NOTE: To use fp16 install apex as per https://github.com/NVIDIA/apex
        # fp16 = True

    return Box({
        'chatbot': chatbot,
        'run_type': run_type,
        'command': command,
        'output_dir': output_dir,
        'model_name_or_path': model_name_or_path,
        'save_dir': save_dir,
        'train_filename': train_filename,
        'path_to_train_data': path_to_train_data,
        'faq_dataset': faq_dataset,
        'qa_dataset': qa_dataset,
        'use_gpu': torch.cuda.is_available(),
        'embedding_model': embedding_model,
        'doc_dir': doc_dir,
        'no_ans_boost': no_ans_boost
    })
