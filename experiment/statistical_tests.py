import pandas as pd
import researchpy as rp
import scipy.stats as stats

df = pd.DataFrame()
df_control = pd.read_csv('data/quiz_results_control.csv')
df_faq = pd.read_csv('data/quiz_results_faq.csv')
df_jugglechat = pd.read_csv('data/quiz_results_jugglechat.csv')
df_qa = pd.read_csv('data/quiz_results_qa.csv')

# Getting summary statistics for each group
print("Control stats:\n", rp.summary_cont(df_control['score']))
print("FAQ stats:\n", rp.summary_cont(df_faq['score']))
print("JuggleChat stats:\n", rp.summary_cont(df_jugglechat['score']))
print("QA stats:\n", rp.summary_cont(df_qa['score']))

# Creating base dataframe with scores from all groups
control_scores = df_control['score'].tolist()
faq_scores = df_faq['score'].tolist()
jugglechat_scores = df_jugglechat['score'].tolist()
qa_scores = df_qa['score'].tolist()

print('\n', stats.f_oneway(control_scores, faq_scores, jugglechat_scores, qa_scores))

