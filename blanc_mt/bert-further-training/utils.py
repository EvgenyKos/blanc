import datetime
import pandas as pd
import re
from polyglot.detect import Detector


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def remove_char_en(s):
    '''Replaces all characters in a English string, except specified in brackets, with a space.
    
    Args:
         s (str): a sequence of characters ( words or sentences).
         
    Returns:
         s (str): updated string without unwanted characters. 
    '''
    s = re.sub(r'[^a-zA-Z0-9]+', " ", s)
    return s


def remove_char_ru(s):
    '''Replaces all characters in a Russian string, except specified in brackets, with a space. 
    
     Args:
         s (str): a sequence of characters ( words or sentences).
         
    Returns:
         s (str): updated string without unwanted characters. 
    '''
    s = re.sub(r'[^А-яЁё0-9]+', " ", s)
    return s


def language_type(sentence):
    '''Detects what language the sentence and returns the code of the detected language.
    
    Args:
         sentence (str): a sentence or multiple sentences. 
         
    Returns:
         language code if detected. 
    '''
    try:
        detector = Detector(sentence)
    except:
        return 'unk'
    else:
        return detector.language.code


def prepare_data(english_corpus, russian_corpus):
    '''Takes two parallel corpuses, concatenates them and returns a dataframe.
    
    Args:
         english_corpus: text file with english corpus.
         russian_corpus: text file with russian corpus. 
         
    Returns:
         dataframe of concatenated corpuses. 
    '''

    f1 = open(english_corpus, encoding="utf8")
    d1 = f1.readlines()
    for index, line in enumerate(d1):
        d1[index] = line.strip()
    f1.close()
    f1 = open(russian_corpus, encoding="utf8")
    d2 = f1.readlines()
    for index, line in enumerate(d2):
        d2[index] = line.strip()
    f1.close()
    print(len(d1))
    print(len(d2))
    en_df = pd.DataFrame(d1)
    ru_df = pd.DataFrame(d2)
    data = pd.concat([ru_df, en_df], axis=1, sort=False)
    data.columns = ['russian', 'english']
    print(data.head())
    return data


def apply_filters(df, ru, en):
    '''Removes all characters but letters and numbers from sentences in the dataframe.
       Removes datapoints that do not match the expected language. 
    
    Args:
         df: dataframe 
         ru : dataframe column with russian sentences.
         en: dataframe column with english sentences. 
         
    Returns:
         modified and filtered dataframe. 
    '''

    df[en] = df[en].apply(remove_char_en)
    df[ru] = df[ru].apply(remove_char_ru)

    df = df.reset_index(drop=True)

    df["language_ru"] = df[ru].apply(language_type)
    df["language_en"] = df[en].apply(language_type)

    df = df[df['language_ru'] == 'ru']
    df = df[df['language_en'] == 'en']

    df = df.reset_index(drop=True)

    df['sentence_len_ru'] = df[ru].str.split().str.len()
    df['sentence_len_en'] = df[en].str.split().str.len()

    return df


def custom_replace(tensor):
    '''Replaces special tokens with value -100.
    
    Args:
         tensor: tensor of tokenized text. 
         
    Returns:
         res: tensor with replaced special tokens. 
    '''
    res = tensor.clone()
    res[tensor == 0] = -100
    res[tensor == 101] = -100
    res[tensor == 102] = -100
    return res
