from blanc import BlancHelp
import pandas as pd


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
