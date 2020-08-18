import pandas as pd
import ast

def split_scores(data, column_name):
      """Splits a dataframe column of a list of scores into columns for each assessor
    Args:
        data (dataframe): WMT dataset with BLANC score
        column_name (str): the column name  with a list of assessments
    Returns:
        data (dataframe): WMT dataset with columns for each assessor
    """
  for idx, l in enumerate(data[column_name]):
    lis = ast.literal_eval(l)
    for i, value in enumerate(lis):
      data.loc[idx, f'assesor_{i+1}'] = lis[i]
  return data

def get_mean(data, col_name1, col_name2):
    """Calculates the mean of two dataframe columns
    Args:
        data (dataframe): WMT dataset with BLANC score
        column_name1 (str): the first column with values
        column_name2 (str): the second column with values
    Returns:
        data (dataframe): WMT dataset with mean values
    """
  cols = data.loc[: , col_name1 : col_name2]
  data[f"mean{col_name1[-1]}{col_name2[-1]}"] = cols.mean(axis=1)
  return data

def add_index_col(data):
    """Adds index column
    Args:
        data (dataframe): WMT dataset with BLANC score
    Returns:
        data (dataframe): WMT dataset with index column
    """
  data = data.reset_index(drop =True)
  data['index_col'] = pd.RangeIndex(start=1, stop=len(data)+1, step=1)
  return data



  
  
