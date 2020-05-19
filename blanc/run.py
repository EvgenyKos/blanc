from blanc import BlancHelp
import pandas as pd

"""
ru_en_train = pd.read_csv('ru-en/train.ruen.df.short.tsv', sep='\t', header=0)

ru_en_train['score'] = 0

for i in range(len(ru_en_train)):
    summary = ru_en_train.translation[i]
    document = ru_en_train.original[i]
    blanc_help = BlancHelp()
    blanc_score = blanc_help.eval_once(document, summary)
    print(blanc_score)
    ru_en_train.loc[i, 'score'] = blanc_score

ru_en_train.to_csv(r"ru_en_blanc.csv", index=False)
"""

parallel = pd.read_csv('parallel_ru_en.csv.gz', index_col=0)
par_sample = parallel.sample(1000)
par_sample = par_sample.reset_index(drop=True)

par_sample['score'] = 0

print(par_sample.head())
print(len(par_sample))

for i in range(len(par_sample)):
    summary = par_sample.english[i]
    document = par_sample.russian[i]
    blanc_help = BlancHelp()
    blanc_score = blanc_help.eval_once(document, summary)
    print(blanc_score)
    par_sample.loc[i, 'score'] = blanc_score

par_sample.to_csv(r"par_blanc.csv", index=False)
