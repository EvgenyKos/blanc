from blanc import BlancHelp
import pandas as pd
import random
import nltk
import argparse
from utils import Defaults
nltk.download('punkt')
nltk.download('words')
from nltk.corpus import words

random.seed(1)

parser = argparse.ArgumentParser()
parser.add_argument(
                    '--type', type=str,
                    default='standard',
                    help='Type of analysis'
                    )
parser.add_argument(
                    '--gap',
                    type=int,
                    help='distance between words to mask during inference',
                    default=Defaults.gap,
                    )
parser.add_argument(
                    '--min_token_length_normal',
                    type=int,
                    help=(
                         'minimum number of chars in normal tokens to mask, where a normal token is '
                         'a whole word'
                          ),
                    default=Defaults.min_token_length_normal,
                    )
parser.add_argument(
                    '--min_token_length_lead',
                    type=int,
                    help='minimum number of chars in lead token to mask, where a lead token begins a word',
                    default=Defaults.min_token_length_lead,
                    )
parser.add_argument(
                    '--min_token_length_followup',
                    type=int,
                    help=(
                         'minimum number of chars in followup token to mask, where a followup token '
                         'continues a word'
                         ),
                    default=Defaults.min_token_length_followup,
                    )
parser.add_argument(
                    '--device',
                    type=str,
                    help='cpu or cuda device',
                    default=Defaults.device,
                    )
parser.add_argument(
                    '--model_name',
                    type=str,
                    help='Select model cased or uncased',
                    default=Defaults.model_name
                    )
parser.add_argument(
                    '--n_tokens',
                    type=int,
                    help='Number of replaced words in translation',
                    default=1
                    )
parser.add_argument(
                    '--output_name',
                    type=str,
                    default="ru_en_blanc",
                    help='Output file name')

parser.add_argument(
                    '--base',
                    type=int,
                    help='Base model or your pritrained',
                    default=Defaults.base,
                    )

args = parser.parse_args()


ru_en_train = pd.read_csv('train.ruen.df.short.tsv', sep='\t', header=0)
# ru_en_train = ru_en_train.sample(100)
# ru_en_train = ru_en_train.reset_index(drop=True)

print(ru_en_train.head())

blanc_help = BlancHelp(gap=args.gap,
                       base=args.base,
                       model_name=args.model_name,
                       min_token_length_normal=args.min_token_length_normal,
                       min_token_length_lead=args.min_token_length_lead,
                       min_token_length_followup=args.min_token_length_followup,
                       device=args.device
                       )

if args.type == 'standard':

    score, total_unks = blanc_help.eval_pairs(ru_en_train.original, ru_en_train.translation)

    print("Total number of UNKs: ", total_unks)

    ru_en_train['score'] = score

    ru_en_train.to_csv(args.output_name + ".csv", sep=',', index=False)


# For random shuffle
elif args.type == 'shuffle':
    t_unks = 0
    for i in range(len(ru_en_train)):

        # shuffle words in a sentences
        en_words = ru_en_train.translation[i].split()
        random.shuffle(en_words)
        new_translation = ' '.join(en_words)
        summary = new_translation

        document = ru_en_train.original[i]

        blanc_score, total_unks = blanc_help.eval_once(document, summary)
        ru_en_train.loc[i, 'score'] = blanc_score
        t_unks += total_unks
    print("Total number of UNKs: ", t_unks)
    ru_en_train.to_csv(args.output_name + ".csv", sep=',', index=False)

# For random words replacement
elif args.type == 'replace':

    word_list = words.words()
    t_unks = 0
    for i in range(len(ru_en_train)):

        # shuffle words in a sentences
        print(ru_en_train.translation[i])
        sen_words = ru_en_train.translation[i].split()

        cnt = 0
        attempt = 0
        status = True

        while cnt < args.n_tokens:

            random.shuffle(word_list)
            n = random.randint(0, len(sen_words)-1)
            attempt += 1
            # print(sen_words[n])
            if attempt < 100:
                if len(sen_words[n]) > 3:

                    for ran_word in word_list:

                        if len(ran_word) != len(sen_words[n]):
                            continue

                        sen_words[n] = ran_word

                        # print(sen_words[n])
                        cnt += 1
                        break
            else:
                break

        new_translation = ' '.join(sen_words)

        print(new_translation)

        summary = new_translation

        document = ru_en_train.original[i]

        blanc_score, total_unks = blanc_help.eval_once(document, summary)
        t_unks += total_unks
        print("Blanc score: ", blanc_score)
        ru_en_train.loc[i, 'score'] = blanc_score
    print("Total number of UNKs: ", t_unks)
    ru_en_train.to_csv(args.output_name + ".csv", sep=',', index=False)

# Set translation as document
elif args.type == 'oposite':
    score, total_unks = blanc_help.eval_pairs(ru_en_train.translation, ru_en_train.original)

    ru_en_train['score'] = score

    ru_en_train.to_csv(args.output_name + ".csv", sep=',', index=False)
    print("Total number of UNKs: ", total_unks)
else:
    print('Uknown option. Please read documentation')
