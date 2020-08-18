def split_scores(data, column_name):
      """Splits a column of a list of scores into columns for each assessor
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


  
  
