import pandas as pd
import researchpy as rp
import scipy.stats as stats

def run_stats(type=None, feature=None):
    print("======================================================================")
    print("Statistics for {} - {}".format(type, feature))
    if type=='quiz_results':
        df_control = pd.read_csv('data/{}_control.csv'.format(type))
    df_faq = pd.read_csv('data/{}_faq.csv'.format(type))
    df_jugglechat = pd.read_csv('data/{}_jugglechat.csv'.format(type))
    df_qa = pd.read_csv('data/{}_qa.csv'.format(type))

    # Getting summary statistics for each group
    if type == 'quiz_results':
        print("Control stats:\n{}".format(rp.summary_cont(df_control[feature])))
    print("FAQ stats:\n{}".format(rp.summary_cont(df_faq[feature])))
    print("JuggleChat stats:\n{}".format(rp.summary_cont(df_jugglechat[feature])))
    print("QA stats:\n{}\n".format(rp.summary_cont(df_qa[feature])))

    # Creating base dataframe with scores from all groups
    if type == 'quiz_results':
        control_scores = df_control[feature].tolist()
    faq_scores = df_faq[feature].tolist()
    jugglechat_scores = df_jugglechat[feature].tolist()
    qa_scores = df_qa[feature].tolist()

    if type == 'quiz_results':
        f_stat, p_value = stats.f_oneway(control_scores, faq_scores, jugglechat_scores, qa_scores)
    else:
        f_stat, p_value = stats.f_oneway(faq_scores, jugglechat_scores, qa_scores)

    print("F statistic: {}, pvalue: {}".format(f_stat, p_value))
    print("======================================================================")


if __name__ == "__main__":
    run_stats(type='quiz_results', feature='score')
    run_stats(type='evaluation_results', feature='evaluation_accuracy')
    run_stats(type='evaluation_results', feature='evaluation_usefulness')
    run_stats(type='evaluation_results', feature='sentiment')
