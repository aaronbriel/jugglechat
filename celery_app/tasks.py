import datetime
import json
import requests
from types import SimpleNamespace

from absum import Augmentor
import pandas as pd

from .celery import app
from .config import (
    ABSUM,
    FAQ,
    FAQ_URL,
    MAIN_TEXT,
    NO_ANSWER_RESPONSE,
    QA,
    QA_URL,
    ABSUM_URL,
    RASA,
    RASA_CORE_URL,
    RASA_NLP_URL,
    URL
)

# TODO temp for experiment
text = MAIN_TEXT

try:
    augmentor = Augmentor(min_length=100, max_length=200, debug=False)
    output = augmentor.get_abstractive_summarization(text)
    # Get all text up to last instance of period
    output = output[:output.rindex('.') + 1]
except Exception as e:
    output = "So sorry, but there was an error getting the summary: {}".format(e)


class Allocator(object):
    """
    This class is instantiated by the intent extraction task, in this case call_rasa_nlp. It routes user
    input text to the chatbot specified in instantiation and returns said module's response to the
    extraction task.
    """

    def __init__(self, text, intent_response, control_group=None, store_data=False):
        self.text = text
        self.intent = intent_response.intent
        self.chatbot = intent_response.chatbot
        self.confidence = intent_response.confidence
        self.full_response = intent_response.full_response
        self.control_group = control_group
        self.store_data = store_data

    def get_response(self):
        """
        Get chatbot response based on chatbot, intent, and confidence
        :return: response, probability and url
        """
        url = ''
        probability = 1
        # If control_group is set, that becomes sole chatbot
        if self.control_group:
            self.chatbot = self.control_group

        if self.chatbot == FAQ:
            response, url, probability = call_faq(self.text)
            if self.intent == URL:
                response = url
            else:
                response = response

        elif self.chatbot == QA:
            response, probability = call_qa(self.text)

        elif self.chatbot == ABSUM:
            response = call_summarizer(self.text)

        else:
            response = NO_ANSWER_RESPONSE

        if self.store_data:
            self.store_allocation(response, probability)

        return response, probability, url, self.chatbot

    def store_allocation(self, response, probability, logfile='output/log.json'):
        """
        Stores allocations as json data structure into a json log file.
        :param response: Response from allocator (ie, underlying chatbot/module).
        :param probability: Probability of correctness of said response
        :param logfile: Logfile to write data to
        """
        timestamp = datetime.datetime.today().strftime('%m-%d-%Y_%H-%M-%S')

        with open(logfile) as json_file:
            data = json.load(json_file)

            temp = data['entries']

            entry = {
                "timestamp": timestamp,
                "input_text": self.text,
                "chatbot": self.chatbot,
                "intent": self.intent,
                "intent_confidence": self.confidence,
                "response": response,
                "response_probability": probability
            }

            temp.append(entry)

        with open(logfile, 'w') as f:
            json.dump(data, f, indent=4)


def get_haystack_response(res, debug=False, chatbot='QA'):
    """
    Function that filters null answers from the haystack response. NOTE: The necessity of this suggests that
    Deepset's no_ans_boost default of 0 may not be functioning for FARMReader.
    :param res:
    :param chatbot: Type of chatbot to get response for
    :return:
    """
    answer = None

    try:
        answers = res['results'][0]['answers']
        df = pd.json_normalize(answers)
        df.dropna(inplace=True)
        df.reset_index(inplace=True)
        answer = df.iloc[df.score.idxmax()]
        response = answer['answer']
        probability = answer['probability']
        # TODO remove once haystack defect logged
        # response = res['results'][0]['answers'][0]['answer']
        # probability = res['results'][0]['answers'][0]['probability']
    except Exception as e:
        if debug:
            response = "So sorry, but there was an error extracting the response from the {} chatbot: '{}'. "\
                .format(chatbot, e)
        else:
            response = NO_ANSWER_RESPONSE
        probability = 1

    try:
        url = answer['meta']['link']
    except Exception:
        url = ''

    return response, probability, url


@app.task
def call_rasa_nlp(text):
    """
    Intent extraction task.
    :param text: Input text of user
    :return: Chatbot/module response
    """
    payload = {'text': text.lower()}
    headers = {'Content-Type': 'application/json'}
    res = requests.post(RASA_NLP_URL, json=payload, headers=headers)
    res_json = json.loads(res.text)
    intent = res_json['intent']['name']
    chatbot = FAQ if intent == URL else intent

    intent_response = {
        'intent': intent,
        'chatbot': chatbot,
        'confidence': res_json['intent']['confidence'],
        'full_response': res_json
    }
    intent_response = SimpleNamespace(**intent_response)

    allocator = Allocator(text, intent_response=intent_response)
    response = allocator.get_response()
    return response


@app.task
def call_rasa_core(text):
    """
    Would be used to call the rasa core chatbot. Currently not implemented.
    :param text: Input text of user
    :return: Chatbot response
    """
    payload = {'text': text.lower()}
    headers = {'Content-Type': 'application/json'}
    res = requests.post(RASA_CORE_URL, json=payload, headers=headers)
    res_json = json.loads(res.text)
    response = res_json['intent']['name']

    return response


@app.task
def call_faq(text):
    """
    Calls Deepset's FAQ-style QA chatbot.
    :param text: Input text of user
    :return: Chatbot response
    """
    payload = {
        'questions': [text],
        'top_k_retriever': 1
    }
    headers = {'Content-Type': 'application/json'}
    res = requests.post(FAQ_URL, json=payload, headers=headers)
    response, probability, url = get_haystack_response(res.json(), chatbot='FAQ')

    return response, url, probability


@app.task
def call_qa(text):
    """
    Calls Deepset's extractive-QA chatbot.
    :param text: Input text of user
    :return: Chatbot response
    """
    payload = {
        'questions': [text],
        'top_k_retriever': 1
    }
    headers = {'Content-Type': 'application/json'}
    res = requests.post(QA_URL, json=payload, headers=headers)
    response, probability, _ = get_haystack_response(res.json(), chatbot='QA')

    return response, probability


@app.task
def call_summarizer(text, min_length=100, max_length=200):
    """
    Instantiates absum Augmentor class to get abstractive summarization of text.
    NOTE: For the purposes of the experiment the calculation is completed during celery initialization. This
    is also consistent with the preferred approach with production instances, namely, that computationally
    expensive operations be completed during initialization whenever possible to improve real-time performance.
    :param text: Text to summarize
    :param min_length: Minimum length of summarization
    :param max_length: Maximum length of summarization
    :return: Summarized text
    """
    # text = MAIN_TEXT
    #
    # try:
    #     augmentor = Augmentor(min_length=min_length, max_length=max_length)
    #     output = augmentor.get_abstractive_summarization(text)
    #     # Get all text up to last instance of period
    #     output = output[:output.rindex('.') + 1]
    # except Exception as e:
    #     output = "So sorry, but there was an error getting the summary: {}".format(e)

    return output

