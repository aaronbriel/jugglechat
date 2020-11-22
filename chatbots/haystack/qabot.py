import argparse
import json

import pandas as pd
from haystack import Finder
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.preprocessor.utils import convert_files_to_dicts, fetch_archive_from_http
from haystack.reader.farm import FARMReader
from haystack.retriever.sparse import ElasticsearchRetriever
from haystack.utils import print_answers


class QABot(object):
    """
    Extractive-QA question answering chatbot. Contains methods for storing embedding documents for retrieval,
    training a QA model, and interacting with the bot. These methods contain elements inspired by:
    https://github.com/deepset-ai/haystack/blob/master/tutorials/Tutorial1_Basic_QA_Pipeline.py

    Training hyperparameters and configuration were taken from the paper,  "COVID-QA: A Question Answering
    Dataset for COVID-19", detailed here: https://huggingface.co/deepset/roberta-base-squad2-covid#hyperparameters

    This also contains several methods for pre-processing the experimental data for SQuAD style format.
    """
    def __init__(self, args):
        self.args = args
        self.document_store = ElasticsearchDocumentStore(
            host="localhost",
            username="",
            password="",
            index="qa_" + self.args.run_type
        )
        self.retriever = ElasticsearchRetriever(document_store=self.document_store)

        self.reader = FARMReader(
            model_name_or_path=args.model_name_or_path,
            use_gpu=args.use_gpu,
            no_ans_boost=args.no_ans_boost
        )

        self.finder = Finder(self.reader, self.retriever)

    def train(self):
        self.reader.train(
            data_dir=self.args.path_to_train_data,
            train_filename=self.args.train_filename,
            use_gpu=self.args.use_gpu,
            save_dir=self.args.save_dir,
            batch_size=24,
            n_epochs=3,
            learning_rate=3e-5,
            warmup_proportion=0.1,
            dev_split=0,
        )

    def eval(self):
        # Download evaluation data, which is a subset of Natural Questions development set containing 50 documents
        doc_dir = "../data/nq"
        s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/nq_dev_subset_v2.json.zip"
        fetch_archive_from_http(url=s3_url, output_dir=doc_dir)
        document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", create_index=False)
        # Add evaluation data to Elasticsearch database
        document_store.add_eval_data("./data/nq/nq_dev_subset_v2.json")

    def interact(self, question):
        """
        Method used for isolated debugging of QA bot through make command or directly
        :param question: Question to ask
        """
        prediction = self.finder.get_answers(question=question, top_k_retriever=1, top_k_reader=10)
        print_answers(prediction, details="minimal")

    def store_documents(self):
        """
        Stores retrieval documents in Elastic Search
        """
        dicts = convert_files_to_dicts(dir_path=self.args.doc_dir)
        self.document_store.write_documents(dicts)


def convert_squad_to_csv(filename):
    """
    Converts SQuAD-formatted dataset to a csv with question and answer columns
    :param filename: SQuAD formatted dataset
    """
    questions = []
    answers = []
    df = pd.DataFrame()

    with open(filename, "r") as file:
        data = json.load(file)
        for document in data["data"]:
            for paragraph in document["paragraphs"]:
                for qa in paragraph["qas"]:
                    for answer in qa["answers"]:
                        ans = answer["text"]
                        ans = ans.replace('\n', '')
                        ans = ans.replace('\t', '')
                        answers.append(ans)
                    questions.append(qa["question"])

    df['question'] = questions
    df['answer'] = answers
    df.to_csv(filename.replace('json', 'csv'), encoding='utf-8', index=False)


def create_intent_file(filename, intent_filename):
    """
    Creating intent file for Rasa NLU.
    Prepending with '- ' for direct insertion of text into rasa/data/nlu.md intent entries.
    :param filename: Name of squad-style data file to convert to intent text file
    :param intent_filename: Name of intent text file to bounce
    """
    questions = []

    with open(filename, "r") as file:
        data = json.load(file)
        for document in data["data"]:
            for paragraph in document["paragraphs"]:
                for qa in paragraph["qas"]:
                    question = qa["question"].replace('\n', '')
                    questions.append('- ' + question + '\n')

    with open(intent_filename, "w") as output:
        output.write(str(''.join(questions)))

    print("Generated {} for intent training.".format(intent_filename))


def clean_squad_file_and_update_answer_starts(filename):
    """
    Removes ":QUESTIONS:" text chunk from contexts (see faqbot.convert_faq_to_annotation_files),
    and updates answer_starts to reflect the removals by subtracting the character count of the
    removed chunk
    of the ":QUESTIONS:" chunk.
    :param filename: SQuAD file to clean
    """
    delimiter = ":QUESTIONS:\n"

    with open(filename, "r") as file:
        data = json.load(file, strict=False)

        for document in data["data"]:
            for paragraph in document["paragraphs"]:

                # extract cleaned context without questions
                context = paragraph["context"]
                context_parts = context.split(delimiter)
                clean_context = context_parts[1]
                answer_start_decrement = len(context_parts[0] + delimiter)
                paragraph["context"] = clean_context

                for qa in paragraph["qas"]:
                    for answer in qa["answers"]:
                        new_answer_start = answer["answer_start"] - answer_start_decrement
                        answer["answer_start"] = new_answer_start

    # Writing modified data to new json file
    new_filename = filename.replace('.json', '-cleaned.json')
    with open(new_filename, "w") as jsonFile:
        json.dump(data, jsonFile)

    print("New SQuAD file generated: {}".format(new_filename))


def create_qa_dataset(squad_file, faq_file):
    """
    Loops over all questions in full SQuAD dataset, and if it is contained in the FAQ dataset
    it is skipped over, thus creating a unique QA dataset consisting of 1/2 of the full set
    :param squad_file: Full SQuAD file
    :param faq_file: FAQ dataset to use for question filter
    """
    faq_df = pd.read_csv(faq_file)
    faq_questions = faq_df["question"].tolist()

    with open(squad_file, "r") as file:
        data = json.load(file, strict=False)

        for document in data["data"]:
            # Using a filter to remove paragraph elements that contain questions in FAQ dataset
            for paragraph in document["paragraphs"]:
                no_faq = lambda qa: qa["question"] not in faq_questions
                paragraph["qas"] = list(filter(no_faq, paragraph["qas"]))

            # Removing all paragraphs that contain empty qa's
            no_qas = lambda paragraph: len(paragraph["qas"]) != 0
            document["paragraphs"] = list(filter(no_qas, document["paragraphs"]))

        # Removing all empty paragraphs
        no_paragraph = lambda document: len(document["paragraphs"]) != 0
        data["data"] = list(filter(no_paragraph, data["data"]))

    # Writing modified data to new json file
    new_filename = squad_file.replace('.json', '-QA.json')
    with open(new_filename, "w") as jsonFile:
        json.dump(data, jsonFile)

    print("New SQuAD file generated: {}".format(new_filename))


def create_qa_storage_docs(squad_file, storage_filename):
    """
    Dumps contexts from squad file into separate text documents used for retriever component
    :param squad_file: SQuAD file
    :param storage_filename: Base filename for storage docs
    """
    doc_count = 1

    with open(squad_file, "r") as file:
        data = json.load(file, strict=False)
        for document in data["data"]:
            for paragraph in document["paragraphs"]:
                new_filename = storage_filename.replace('.txt', str(doc_count) + '.txt')
                with open(new_filename, "w") as output:
                    output.write(paragraph["context"])
                doc_count += 1

    print("Generated QA documents for retrieval.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--command",
        default='all',
        type=str,
        required=True,
        help="The following commands are allowed: 'store', 'train', 'eval', 'ask'"
    )

    parser.add_argument(
        '-r',
        '--run_type',
        default='default',
        type=str,
        required=False,
        help="The following commands are allowed: 'qa', 'experiment'"
    )

    parser.add_argument(
        "--question",
        default="Who is the father of Arya Stark?",
        type=str,
        required=False,
        help="Question to ask the bot"
    )

    cl_args = parser.parse_args()
    command = cl_args.command
    run_type = cl_args.run_type
    question = cl_args.question

    args = get_args(command=command, run_type=run_type, chatbot='qa')
    qabot = QABot(args)

    if command == 'train':
        qabot.train()

    elif command == 'eval':
        qabot.eval()

    elif command == 'ask':
        qabot.interact(question)

    elif command == 'store':
        qabot.store_documents()

    elif 'create_intent_file' in command:
        qa_filename = args.path_to_train_data + args.qa_dataset
        create_intent_file(qa_filename, args.path_to_train_data + "qa_intent.txt")

    elif 'prepare_data' in command:
        # Just in case command accidentally triggered
        print("Are you sure you wish to re-prepare the data? If so, press 'c'.")
        import pdb
        pdb.set_trace()
        
        # Used to fix SQuAD formatted dataset exported from haystack to remove 'QUESTIONS' chunk
        # and update answer starts appropriately
        # clean_squad_file_and_update_answer_starts(
        #     BASE_PATH + "chatbots/haystack/data/qa/squad-jhu_covid_qa-refined.json"
        # )
        #

        create_qa_dataset(
            squad_file=QA_DATA_PATH + QA_DATASET_FULL,
            faq_file=FAQ_DATA_PATH + FAQ_DATASET_JUGGLECHAT,
        )

        create_qa_storage_docs(
            squad_file=QA_DATA_PATH + QA_DATASET_JUGGLECHAT,
            storage_filename=args.doc_dir + QA_DATASET_JUGGLECHAT.replace('.json', '.txt')
        )

    elif 'prepare_deepset_data' in command:
        # Creating deepset storage docs for elastic search
        create_qa_storage_docs(
            squad_file=QA_DATA_PATH + QA_DATASET_DEEPSET,
            storage_filename=args.doc_dir + QA_DATASET_DEEPSET.replace('.json', '.txt')
        )
        # Converting deepset dataset to CSV for elasticsearch indexing
        convert_squad_to_csv(QA_DATA_PATH + FAQ_DATASET_DEEPSET)


if __name__ == "__main__":
    from config import (
        get_args, QA_DATA_PATH, QA_DATASET_JUGGLECHAT, QA_DATASET_DEEPSET,
        QA_DATASET_FULL, FAQ_DATASET_JUGGLECHAT, FAQ_DATA_PATH, FAQ_DATASET_DEEPSET
    )
    main()
else:
    from .config import get_args



