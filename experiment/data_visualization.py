import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud


def create_wordclouds():
    """
    Creating word clouds for each of the evaluation sentiment texts for QA, FAQ and JuggleChat. Results
    of word cloud experimentation unfortunately did not yield and useful visualizations.
    :return:
    """

    faq_eval = pd.read_csv('data/evaluation_results_faq.csv')['evaluation_sentiment'].tolist()
    faq_eval_text = ''.join(faq_eval)
    jugglechat_eval = pd.read_csv('data/evaluation_results_jugglechat.csv')['evaluation_sentiment'].tolist()
    jugglechat_eval_text = ''.join(jugglechat_eval)
    qa_eval = pd.read_csv('data/evaluation_results_qa.csv')['evaluation_sentiment'].tolist()
    qa_eval_text = ''.join(qa_eval)

    wordcloud = WordCloud(max_font_size=40, collocations=False).generate(faq_eval_text)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.savefig("faq_wordcloud.png", dpi=300)
    plt.cla()
    plt.close()

    wordcloud = WordCloud(max_font_size=40, collocations=False).generate(qa_eval_text)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig("qa_wordcloud.png", dpi=300)
    plt.cla()
    plt.close()

    wordcloud = WordCloud(max_font_size=40, collocations=False).generate(jugglechat_eval_text)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig("jugglechat_wordcloud.png", dpi=300)
    plt.cla()
    plt.close()


def plot_quiz_results():
    """
    Plots mean quiz scores for each experimental group.
    :return:
    """
    with open("results.json", "r") as results:
        results_json = json.load(results)

    mean_quiz_score_control = results_json['mean_quiz_score_control']
    mean_quiz_score_jugglechat = results_json['mean_quiz_score_jugglechat']
    mean_quiz_score_faq = results_json['mean_quiz_score_faq']
    mean_quiz_score_qa = results_json['mean_quiz_score_qa']
    mean_quiz_scores = [
        mean_quiz_score_control,
        mean_quiz_score_jugglechat,
        mean_quiz_score_faq,
        mean_quiz_score_qa]

    experimental_groups = ['Control', 'JuggleChat', 'FAQ', 'QA']

    mean_series = pd.Series(mean_quiz_scores)
    plt.figure()
    ax = mean_series.plot(kind='bar')

    for bar in ax.patches:
        y_location = bar.get_height() + .025
        ax.text(bar.get_x(), y_location,
                str(round(bar.get_height(), 5)), fontsize=10,
                color='black')

    plt.ylim(bottom=0, top=0.6)
    y_position = np.arange(len(experimental_groups))
    plt.bar(y_position, mean_quiz_scores, align='center', color='#4a4a4a')
    plt.xticks(y_position, experimental_groups, rotation='horizontal')
    plt.ylabel('Mean Quiz Scores')
    plt.savefig('images/quiz_scores.png')
    plt.close()


def plot_sentiment_results():
    """
    Plots mean sentiment for each experimental group.
    :return:
    """
    with open("results.json", "r") as results:
        results_json = json.load(results)

    mean_sentiment_faq = results_json['mean_sentiment_faq']
    mean_sentiment_jugglechat = results_json['mean_sentiment_jugglechat']
    mean_sentiment_qa = results_json['mean_sentiment_qa']
    mean_sentiments = [
        mean_sentiment_faq,
        mean_sentiment_jugglechat,
        mean_sentiment_qa]

    experimental_groups = ['FAQ', 'JuggleChat', 'QA']

    mean_series = pd.Series(mean_sentiments)
    plt.figure()
    ax = mean_series.plot(kind='bar')

    for bar in ax.patches:
        x_location = bar.get_x() + 0.125
        y_location = bar.get_height() + .025
        ax.text(x_location, y_location,
                str(round(bar.get_height(), 2)),
                fontsize=10, color='black')

    plt.ylim(bottom=0, top=0.5)
    y_position = np.arange(len(experimental_groups))
    plt.bar(y_position, mean_sentiments, align='center', color='grey')
    plt.xticks(y_position, experimental_groups, rotation='horizontal')
    plt.ylabel('Mean Sentiments')
    plt.savefig('images/sentiments.png')
    plt.close()


def plot_accuracy_and_usefulness():
    """
    Plots mean accuracy ratings and usefulness ratings for each experimental group. Contains code
    inspired from: https://matplotlib.org/gallery/api/barchart.html
    """
    with open("results.json", "r") as results:
        results_json = json.load(results)

    experimental_groups = ['FAQ', 'JuggleChat', 'QA']

    accuracy = [
        round(results_json['mean_accuracy_rating_faq']/10, 3),
        round(results_json['mean_accuracy_rating_jugglechat']/10, 3),
        round(results_json['mean_accuracy_rating_qa']/10, 3)
    ]
    usefulness = [
        round(results_json['mean_usefulness_rating_faq']/10, 3),
        round(results_json['mean_usefulness_rating_jugglechat']/10, 3),
        round(results_json['mean_usefulness_rating_qa']/10, 3)
    ]

    bar_width = 0.3

    accuracy_positions = np.arange(len(experimental_groups))
    usefulness_positions = [x + bar_width for x in accuracy_positions]

    # accuracy_series = pd.Series(accuracy)
    # usefulness_series = pd.Series(usefulness)
    # plt.figure()
    # ax1 = accuracy_series.plot(kind='bar')
    # ax2 = usefulness_series.plot(kind='bar')
    #
    # for bar in ax1.patches:
    #     x_location = bar.get_x() + 0.125
    #     y_location = bar.get_height() + .025
    #     ax1.text(x_location, y_location,
    #              str(round(bar.get_height(), 2)),
    #              fontsize=10, color='black')
    #
    # for bar in ax2.patches:
    #     x_location = bar.get_x() + 0.125
    #     y_location = bar.get_height() + .025
    #     ax2.text(x_location, y_location,
    #              str(round(bar.get_height(), 2)),
    #              fontsize=10, color='black')

    fig, ax = plt.subplots()
    # widths = [bar_width / 2, bar_width / 2, bar_width / 2]
    accuracy_bars = ax.bar(accuracy_positions, accuracy, bar_width,
                           color='#666666', label='Accuracy Rating', edgecolor='white')
    usefulness_bars = ax.bar(usefulness_positions, usefulness, bar_width,
                             color='#adadad', label='Usefulness Rating', edgecolor='white')

    ax.set_xticks([tick + bar_width/2 for tick in range(len(accuracy))])
    ax.set_xticklabels(experimental_groups)
    ax.legend()

    def autolabel(rects, xpos='center'):
        xpos = xpos.lower()
        ha = {'center': 'center', 'right': 'left', 'left': 'right'}
        offset = {'center': 0.5, 'right': 0.1, 'left': 0.9}

        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() * offset[xpos], 1.001 * height,
                    '{}'.format(height), ha=ha[xpos], va='bottom')

    autolabel(accuracy_bars, "left")
    autolabel(usefulness_bars, "right")

    # plt.bar(accuracy_positions, accuracy,
    #         label='Accuracy Rating', width=bar_width, color='#666666', edgecolor='white')
    # plt.bar(usefulness_positions, usefulness,
    #         label='Usefulness Rating', width=bar_width, color='#adadad', edgecolor='white')
    #
    # plt.xticks([tick + bar_width/2 for tick in range(len(accuracy))], experimental_groups)
    # plt.legend()

    plt.savefig("images/accuracy_usefulness.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    plot_quiz_results()
    plot_sentiment_results()
    plot_accuracy_and_usefulness()
