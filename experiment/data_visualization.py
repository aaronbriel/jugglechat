import json
import matplotlib.pyplot as plt
import pandas as pd
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


def create_accuracy_usefulness_plot():
    with open("results.json", "r") as results:
        results_json = json.load(results)

    values = []
    alpha_priors_column = []
    for alpha_priors_ in alpha_priors_list:
        values_ = get_expected_value(observations, alpha_priors_)
        values.append(values_)
        alpha_priors_column.append(", ".join(str(x) for x in alpha_priors_))

    df = pd.DataFrame(values, columns=sentiment)
    df['alpha_priors'] = alpha_priors_column

    # Plot expectation graphic
    data = pd.melt(
        df,
        var_name='sentiment',
        value_name='expected_percent',
        id_vars='alpha_priors'
    )
    plt.figure()
    sns.barplot(
        data=data,
        x='alpha_priors',
        y='expected_percent',
        palette='hls',
        hue='sentiment'
    )
    plt.savefig("accuracy_and_usefulness.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    create_accuracy_usefulness_plot()
