import argparse
from haystack import Finder
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.retriever.dense import EmbeddingRetriever
import pandas as pd
from transformers import AutoTokenizer, AutoModel


class FaqBot(object):
    """
    FAQ-Style question answering chatbot. Contains methods for storing embedding documents for retrieval and
    interacting with the bot. These methods contain elements inspired by:
    https://github.com/deepset-ai/haystack/blob/master/tutorials/Tutorial4_FAQ_style_QA.py

    This also contains several methods for pre-processing the experimental data for FAQ style format.
    """
    def __init__(self, args):
        self.args = args

        # FAQ type finder
        self.document_store = ElasticsearchDocumentStore(
            host="localhost",
            password="",
            index="faq_" + self.args.run_type,
            embedding_field="question_emb",
            embedding_dim=768,
            excluded_meta_data=["question_emb"]
        )

        self.retriever = EmbeddingRetriever(document_store=self.document_store,
                                            embedding_model=self.args.embedding_model,
                                            use_gpu=False)

    def interact(self, question):
        """
        Method to be used for isolated debugging of FAQ bot through make command or directly
        :param question: Question to ask
        """
        finder = Finder(reader=None, retriever=self.retriever)
        prediction = finder.get_answers_via_similar_questions(question=question, top_k_retriever=1)

        print("Answer:\n", prediction['answers'][0]['answer'])
        print("Probability: ", prediction['answers'][0]['probability'])

    def store_documents(self):
        """
        Used to store "documents" or FAQ data (curated question and answer pairs) for later comparison
        to user queries. Here, we are indexing in elasticsearch.
        NOTE: Expects docker instance of elasticsearch to be running (see make elastic command)
        """
        df = pd.read_csv(self.args.path_to_train_data + self.args.faq_dataset)
        # Minor data cleaning
        df.fillna(value="", inplace=True)
        df["question"] = df["question"].apply(lambda x: x.strip())

        # Get embeddings for our questions from the FAQs
        questions = list(df["question"].values)
        df["question_emb"] = self.retriever.embed_queries(texts=questions)
        # convert from numpy to list for ES indexing
        df["question_emb"] = df["question_emb"].apply(list)
        df = df.rename(columns={"answer": "text"})

        # Convert Dataframe to list of dicts and index them in our DocumentStore
        docs_to_index = df.to_dict(orient="records")
        self.document_store.write_documents(docs_to_index)

    def download_embedding_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.args.embedding_model)
        model = AutoModel.from_pretrained(self.args.embedding_model)
        tokenizer.save_pretrained(self.args.embedding_model)
        model.save_pretrained(self.args.embedding_model)


def combine_and_refine_faq_datasets(bouncename):
    """
    Refining JHU covid dataset
    :param bouncename: Name of exported file
    """
    df_jhu_unique = pd.read_csv(BASE_PATH + "chatbots/haystack/data/faq/jhu_covid_qa.csv")
    df_jhu_unique.dropna(inplace=True)

    df_jhu_rating_90 = df_jhu_unique[df_jhu_unique['rating'] >= 90]
    df_jhu_rating_90.drop_duplicates(subset='question1', keep="first", inplace=True)
    df_jhu_rating_90.rename({'question1': 'question'}, axis=1, inplace=True)
    df_jhu_rating_90 = df_jhu_rating_90[['question', 'answer']]

    df_jhu_unique.drop_duplicates(subset='question2', keep="first", inplace=True)
    df_jhu_unique.rename({'question2': 'question'}, axis=1, inplace=True)
    df_jhu_unique = df_jhu_unique[['question', 'answer']]

    df_faqcovidbert = pd.read_csv(BASE_PATH + "chatbots/haystack/data/faq/faq-faq_covidbert.csv")
    df_faqcovidbert = df_faqcovidbert.replace('\n', '. ', regex=True)

    df = df_faqcovidbert.append(df_jhu_unique, ignore_index=True)
    df = df.append(df_jhu_rating_90, ignore_index=True)

    df.drop_duplicates(subset='question', keep="first", inplace=True)

    # Shuffling rows
    df = df.sample(frac=1).reset_index(drop=True)

    df.to_csv(bouncename, encoding='utf-8', index=False)


def convert_faq_answers_to_files(filename, bouncename, include_question_nums=False):
    """
    Converts FAQ style dataset to format amenable to haystack annotation tool, with option to
    include question indices corresponding to row values in excel. If questions are not included,
    docs can be used for retrieval storage in QA-style chatbot.
    :param filename: FAQ style file to bounce
    :param bouncename: Base name of exported annotation files
    :param include_question_nums: Whether to include question numbers in exported files
    """
    df = pd.read_csv(filename)
    count = 1
    unique_answers = df['answer'].unique()

    for answer in unique_answers:
        questions = df.index[df['answer'] == answer].tolist()
        # incrementing by 2 to accommodate excel
        questions = [x + 2 for x in questions]
        questions_str = ', '.join(str(e) for e in questions)

        if include_question_nums:
            text = ":QUESTIONS: " + questions_str + " :QUESTIONS:\n" + answer + "\n\n"
        else:
            text = answer

        new_name = bouncename.replace('.txt', str(count) + '.txt')
        with open(new_name, "w") as text_file:
            text_file.write(text)

        count += 1

    print("Annotation files created.")


def convert_faq_to_dialog_format(filename, bouncename):
    """
    Converts FAQ style dataset to single question + answer format, ammenable
    to DialoGPT and other conversational models
    :param filename: Name of FAQ style data file to convert
    :param bouncename: Name of new dataset to bounce
    :return:
    """
    df = pd.read_csv(filename)
    dialog_df = pd.DataFrame()
    lines = []

    for line in range(0, df.shape[0]):
        lines.append(df.iloc[line]['question'])
        lines.append(df.iloc[line]['answer'])

    dialog_df['lines'] = lines
    dialog_df.to_csv(bouncename, encoding='utf-8', index=False)


def create_intent_file(filename, intent_filename):
    """
    Creating intent file for Rasa NLU, Prepending with '- ' for direct insertion of
     text into rasa/data/nlu.md intent entries, as similarity across questions between faq and qa
     will likely cause overlap issues.
    :param filename: Name of FAQ style data file to convert to intent text file
    :param intent_filename: Name of intent text file to bounce
    """
    df = pd.read_csv(filename)
    questions = []

    for line in range(0, df.shape[0]):
        questions.append('- ' + df.iloc[line]['question'].replace('\n', '') + '\n')

    with open(intent_filename, "w") as output:
        output.write(str(''.join(questions)))

    print("Generated {} for intent training.".format(intent_filename))


def create_faq_control_dataset(filename):
    """
    Creating FAQ control dataset of 2nd half of full dataset (QA is first half). Data was already shuffled
    in combine_and_refine_faq_datasets so a simple split is fine
    :param bouncename: Name of file to export
    """
    df = pd.read_csv(filename)
    count = int(df.shape[0]/2)
    df_control = df[count:]
    bouncename = filename.replace('.csv', '-FAQ.csv')
    df_control.to_csv(bouncename, encoding='utf-8', index=False)
    print("Generated {} for intent training.".format(bouncename))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--command",
        default='all',
        type=str,
        required=True,
        help="The following commands are allowed: 'store', 'ask'"
    )

    parser.add_argument(
        '-r',
        '--run_type',
        default='default',
        type=str,
        required=False,
        help="The following commands are allowed: 'default', 'experiment'"
    )

    parser.add_argument(
        "--question",
        default="How is the virus spreading?",
        type=str,
        required=False,
        help="Question to ask the bot"
    )

    cl_args = parser.parse_args()
    command = cl_args.command
    run_type = cl_args.run_type
    question = cl_args.question

    args = get_args(command=command, run_type=run_type, chatbot='faq')
    faqbot = FaqBot(args)

    if command == 'ask':
        faqbot.interact(question)

    elif 'store' in command:
        faqbot.store_documents()

    elif 'download_model' in command:
        faqbot.download_embedding_model()

    elif 'create_intent_file' in command:
        faq_filename = args.path_to_train_data + args.faq_dataset
        create_intent_file(faq_filename, args.path_to_train_data + 'faq_intent.txt')

    elif 'prepare_data' in command:
        # Just in case command accidentally triggered
        print("Are you sure you wish to re-prepare the data?")
        import pdb
        pdb.set_trace()

        combine_and_refine_faq_datasets(BASE_PATH + "chatbots/haystack/data/faq/faq-jhu_covid_qa-refined.csv")

        # NOTE skipped_questions were removed to then create faq-jhu_covid_qa-refined-cleaned.csv

        # Converting FAQ answers to annotation files
        convert_faq_answers_to_files(
            BASE_PATH + "chatbots/haystack/data/faq/faq-jhu_covid_qa-refined.csv",
            BASE_PATH + "chatbots/haystack/data/qa/annotation/faq-jhu_covid_qa-refined.txt",
            include_question_nums=True
        )

        # Converting FAQ answers to storage retrieval files for QA control group
        convert_faq_answers_to_files(
            BASE_PATH + "chatbots/haystack/data/faq/faq-jhu_covid_qa-refined.csv",
            BASE_PATH + "chatbots/haystack/data/qa/experiment/faq-jhu_covid_qa-refined.txt"
        )

        create_faq_control_dataset(BASE_PATH + "chatbots/haystack/data/faq/faq-jhu_covid_qa-refined-cleaned.csv")

        # Converting FAQ answers to storage retrieval files for QA control group
        convert_faq_answers_to_files(
            BASE_PATH + "chatbots/haystack/data/faq/faq-jhu_covid_qa-refined-cleaned-FAQ.csv",
            BASE_PATH + "chatbots/haystack/data/qa/default/faq-jhu_covid_qa-refined.txt"
        )


if __name__ == "__main__":
    from config import get_args, BASE_PATH

    # # bouncing covid_full to text file for annotation tool
    # convert_faq_to_annotation_format("data/faq/faq_covidbert.csv")

    # # bouncing faq_covidbert.csv to dialog format
    # convert_faq_to_dialog_format("data/faq/faq-faq_covidbert.csv", "data/faq/dialog-faq_covidbert.csv")
    # bouncing faq_covidbert.csv to dialog format
    # convert_faq_to_dialog_format("data/qa/faq-COVID-QA.csv", "data/qa/dialog-COVID-QA.csv")

    # # debugging absum module
    # from absum import Augmentor
    # augmentor = Augmentor(min_length=100, max_length=200)
    # output = augmentor.get_abstractive_summarization(MAIN_TEXT)

    # convert_faq_to_squad_format(BASE_PATH + "chatbots/haystack/data/faq/faq-jhu_covid_qa-full.csv")

    main()
else:
    from .config import get_args, BASE_PATH

