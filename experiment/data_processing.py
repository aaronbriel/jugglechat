import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline


def get_stop_words():
    with open('./stopwords.txt') as f:
        lines = f.read().splitlines()
    return lines


def get_top_words(df, column='evaluation_sentiment', n=5):
    """
    Gets list of top words from a column of sentences using TF-IDF algorithm
    :param df: Dataframe with column to process
    :param column: Column name containing text to concatenate
    :param n: Number of top words to keep
    :return: List of top n words in body
    """
    body = ' '.join(df[:][column])[0]
    list_body = df[column].tolist()

    stop_words = get_stop_words()
    vectorizer = TfidfVectorizer(stop_words=stop_words)

    # Fitting vocabulary
    vectors = vectorizer.fit_transform(list_body)
    feature_names = vectorizer.get_feature_names()
    feature_array = np.array(feature_names)
    response = vectorizer.transform([body])
    tfidf_sorting = np.argsort(response.toarray()).flatten()[::-1]
    top_n = feature_array[tfidf_sorting][:n]

    return top_n.tolist()


def get_and_store_sentiment(csv, column='evaluation_sentiment'):
    """
    Applies transformers pipeline sentiment calculation to each of the evaluation
    sentiment entries of participants
    :param csv: Path to csv containing text sentiment entries
    :param column: sentiment column
    :return: Dataframe with numeric sentiment column
    """
    df = pd.read_csv(csv)
    pd.options.mode.chained_assignment = None
    nlp = pipeline("sentiment-analysis")
    sentiment_dicts = df[column].apply(nlp)
    # Use list comprehension to extract actual values
    sentiments = [d[0]['label'] for d in sentiment_dicts]
    df['sentiment'] = sentiments
    # Use vectorization to convert sentiment to binary
    df['sentiment'][df['sentiment'] == 'NEGATIVE'] = 0
    df['sentiment'][df['sentiment'] == 'POSITIVE'] = 1

    df.to_csv(csv, encoding='utf-8', index=False)
    return df


def get_data_results():
    results = {}

    df_allocations = pd.read_csv('data/allocations_jugglechat.csv')
    qa_allocation_count = df_allocations[df_allocations['chat_bot_type'] == 'qa'].shape[0]
    results['qa_allocation_count'] = qa_allocation_count
    print("JuggleChat QA Allocation Count: ", qa_allocation_count)

    faq_allocation_count = df_allocations[df_allocations['chat_bot_type'] == 'faq'].shape[0]
    results['faq_allocation_count'] = faq_allocation_count
    print("JuggleChat FAQ Allocation Count: ", faq_allocation_count)

    absum_allocation_count = df_allocations[df_allocations['chat_bot_type'] == 'absum'].shape[0]
    results['absum_allocation_count'] = absum_allocation_count
    print("Jugglechat Absum Allocation Count: ", absum_allocation_count)
    nonbot_allocation_count = df_allocations.shape[0] - qa_allocation_count - \
                                 faq_allocation_count - absum_allocation_count
    results['nonbot_allocation_count'] = nonbot_allocation_count
    print("Remainder counts (ie, Rasa could not match question): ", nonbot_allocation_count)

    df_evaluation_results_jugglechat = get_and_store_sentiment('data/evaluation_results_jugglechat.csv')
    mean_sentiment_jugglechat = df_evaluation_results_jugglechat['sentiment'].mean()
    results['mean_sentiment_jugglechat'] = mean_sentiment_jugglechat
    print("Mean Sentiment JuggleChat: ", mean_sentiment_jugglechat)

    # Getting list of participants who requested summaries
    df_absum_ids = df_allocations[df_allocations['chat_bot_type'] == 'absum']['worker_id'].tolist()
    mean_sentiment_absum = df_evaluation_results_jugglechat[
        df_evaluation_results_jugglechat['worker_id'].isin(df_absum_ids)]['sentiment'].mean()
    results['mean_sentiment_absum'] = mean_sentiment_absum
    print("Mean Sentiment Absum: ", mean_sentiment_absum)

    df_absum = df_allocations[(df_allocations['worker_id'].isin(df_absum_ids))]
    absum_qa = df_absum[df_absum['chat_bot_type'] == 'qa'].shape[0]
    results['absum_qa'] = absum_qa
    print("Percentage of Absum participants who also used QA: ", absum_qa)
    absum_faq = df_absum[df_absum['chat_bot_type'] == 'faq'].shape[0]
    results['absum_faq'] = absum_faq
    print("Percentage of Absum participants who also used FAQ: ", absum_faq)

    df_evaluation_results_faq = get_and_store_sentiment('data/evaluation_results_faq.csv')
    mean_sentiment_faq = df_evaluation_results_faq['sentiment'].mean()
    results['mean_sentiment_faq'] = mean_sentiment_faq
    print("Mean Sentiment FAQ: ", mean_sentiment_faq)

    df_evaluation_results_qa = get_and_store_sentiment('data/evaluation_results_qa.csv')
    mean_sentiment_qa = df_evaluation_results_qa['sentiment'].mean()
    results['mean_sentiment_qa'] = mean_sentiment_qa
    print("Mean Sentiment QA: ", mean_sentiment_qa)

    mean_accuracy_rating_faq = df_evaluation_results_faq['evaluation_accuracy'].mean()
    results['mean_accuracy_rating_faq'] = mean_accuracy_rating_faq
    print("Mean Accuracy Rating FAQ: ", mean_accuracy_rating_faq)
    mean_usefulness_rating_faq = df_evaluation_results_faq['evaluation_usefulness'].mean()
    results['mean_usefulness_rating_faq'] = mean_usefulness_rating_faq
    print("Mean Usefulness Rating FAQ: ", mean_usefulness_rating_faq)

    mean_accuracy_rating_jugglechat = df_evaluation_results_jugglechat['evaluation_accuracy'].mean()
    results['mean_accuracy_rating_jugglechat'] = mean_accuracy_rating_jugglechat
    print("Mean Accuracy Rating JuggleChat: ", mean_accuracy_rating_jugglechat)
    mean_usefulness_rating_jugglechat = df_evaluation_results_jugglechat['evaluation_usefulness'].mean()
    results['mean_usefulness_rating_jugglechat'] = mean_usefulness_rating_jugglechat
    print("Mean Usefulness Rating JuggleChat: ", mean_usefulness_rating_jugglechat)

    mean_accuracy_absum = df_evaluation_results_jugglechat[
        df_evaluation_results_jugglechat['worker_id'].isin(df_absum_ids)]['evaluation_accuracy'].mean()
    results['mean_accuracy_absum'] = mean_accuracy_absum
    print("Mean Accuracy Rating Absum: ", mean_accuracy_absum)
    mean_usefulness_absum = df_evaluation_results_jugglechat[
        df_evaluation_results_jugglechat['worker_id'].isin(df_absum_ids)]['evaluation_usefulness'].mean()
    results['mean_usefulness_absum'] = mean_usefulness_absum
    print("Mean Usefulness Rating Absum: ", mean_usefulness_absum)

    mean_accuracy_rating_qa = df_evaluation_results_qa['evaluation_accuracy'].mean()
    results['mean_accuracy_rating_qa'] = mean_accuracy_rating_qa
    print("Mean Accuracy Rating QA: ", mean_accuracy_rating_qa)
    mean_usefulness_rating_qa = df_evaluation_results_qa['evaluation_usefulness'].mean()
    results['mean_usefulness_rating_qa'] = mean_usefulness_rating_qa
    print("Mean Usefulness Rating QA: ", mean_usefulness_rating_qa)

    mean_quiz_score_control = pd.read_csv('data/quiz_results_control.csv')['score'].mean()
    results['mean_quiz_score_control'] = mean_quiz_score_control
    print("Mean Quiz Score Control: ", mean_quiz_score_control)
    mean_quiz_score_faq = pd.read_csv('data/quiz_results_faq.csv')['score'].mean()
    results['mean_quiz_score_faq'] = mean_quiz_score_faq
    print("Mean Quiz Score FAQ: ", mean_quiz_score_faq)
    mean_quiz_score_jugglechat = pd.read_csv('data/quiz_results_jugglechat.csv')['score'].mean()
    results['mean_quiz_score_jugglechat'] = mean_quiz_score_jugglechat
    print("Mean Quiz Score JuggleChat: ", mean_quiz_score_jugglechat)
    mean_quiz_score_qa = pd.read_csv('data/quiz_results_qa.csv')['score'].mean()
    results['mean_quiz_score_qa'] = mean_quiz_score_qa
    print("Mean Quiz Score QA: ", mean_quiz_score_qa)

    with open('results.json', 'w') as fp:
        json.dump(results, fp)


if __name__ == "__main__":
    # Not yielding useful datta
    # print("Top Words FAQ: ", get_top_words(df_evaluation_results_faq))
    # print("Top Words QA: ", get_top_words(df_evaluation_results_qa))
    # print("Top Words JuggleChat: ", get_top_words(df_evaluation_results_jugglechat))

    get_data_results()
