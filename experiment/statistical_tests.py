import matplotlib.pyplot as plt
import pandas as pd
import researchpy as rp
import scipy.stats as stats
import statsmodels.stats.multicomp as mc


def run_anova(exp_type=None, feature=None):
    """
    Calculates summary statistics and ANOVA one-way test for given feature across
    all experimental groups
    :param type: type of experiment
    :param feature: Measured experimental feature (ie test score, sentiment, etc)
    :return: None
    """
    print("======================================================================")
    print("Statistics for {} - {}".format(exp_type, feature))
    if exp_type == 'quiz_results':
        df_control = pd.read_csv('data/{}_control.csv'.format(exp_type))
    df_faq = pd.read_csv('data/{}_faq.csv'.format(exp_type))
    df_jugglechat = pd.read_csv('data/{}_jugglechat.csv'.format(exp_type))
    df_qa = pd.read_csv('data/{}_qa.csv'.format(exp_type))

    # Getting summary statistics for each group
    if exp_type == 'quiz_results':
        print("Control stats:\n{}".format(rp.summary_cont(df_control[feature])))
    print("FAQ stats:\n{}".format(rp.summary_cont(df_faq[feature])))
    print("JuggleChat stats:\n{}".format(rp.summary_cont(df_jugglechat[feature])))
    print("QA stats:\n{}\n".format(rp.summary_cont(df_qa[feature])))

    # Creating base dataframe with scores from all groups
    if exp_type == 'quiz_results':
        control_scores = df_control[feature].tolist()
    faq_scores = df_faq[feature].tolist()
    jugglechat_scores = df_jugglechat[feature].tolist()
    qa_scores = df_qa[feature].tolist()

    if exp_type == 'quiz_results':
        f_stat, p_value = stats.f_oneway(control_scores, faq_scores, jugglechat_scores, qa_scores)
    else:
        f_stat, p_value = stats.f_oneway(faq_scores, jugglechat_scores, qa_scores)

    print("F statistic: {}, pvalue: {}".format(f_stat, p_value))
    print("======================================================================")


def plot_tukey_post_hoc_test(df, feature=None, new_feature=None):
    """
    Runs Tukey post-hoc test on given feature for all experimental groups, printing and
    plotting the results.
    :param df: Flat dataframe (see create_tukey_df)
    :param feature: Measured experimental feature (ie test score, sentiment, etc)
    :param new_feature: Feature renamed for graph readability
    :return: None
    """
    df.rename(columns={feature: new_feature}, inplace=True)
    df = df.filter([new_feature, 'Experimental Group'])

    comp = mc.MultiComparison(df[new_feature], df['Experimental Group'])
    post_hoc_res = comp.tukeyhsd()

    print("===============================================================")
    print(new_feature)
    print(post_hoc_res.summary())

    post_hoc_res.plot_simultaneous(ylabel="Experimental Group", xlabel=new_feature)

    plt.savefig("images/tukey_{}.png".format(feature), dpi=300)
    plt.close()


def create_tukey_df():
    """
    Creates flat dataframe containing all calculations for all experimental groups
    """
    df_faq = pd.read_csv('data/evaluation_results_faq.csv')
    df_faq['Experimental Group'] = 'FAQ'
    df_jugglechat = pd.read_csv('data/evaluation_results_jugglechat.csv')
    df_jugglechat['Experimental Group'] = 'JuggleChat'
    df_qa = pd.read_csv('data/evaluation_results_qa.csv')
    df_qa['Experimental Group'] = 'Extractive-QA'

    df = df_faq.copy()
    df = df.append(df_jugglechat)
    df = df.append(df_qa)

    return df


if __name__ == "__main__":
    run_anova(exp_type='quiz_results', feature='score')
    run_anova(exp_type='evaluation_results', feature='evaluation_accuracy')
    run_anova(exp_type='evaluation_results', feature='evaluation_usefulness')
    run_anova(exp_type='evaluation_results', feature='sentiment')

    df_ = create_tukey_df()
    plot_tukey_post_hoc_test(df_, 'evaluation_accuracy', 'Perceived Accuracy')
    plot_tukey_post_hoc_test(df_, 'evaluation_usefulness', 'Perceived Usefulness')
    plot_tukey_post_hoc_test(df_, 'sentiment', 'Sentiment')
