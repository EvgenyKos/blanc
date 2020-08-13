def split_scores(data, column_name):
  for idx, l in enumerate(data[column_name]):
    lis = ast.literal_eval(l)
    for i, value in enumerate(lis):
      data.loc[idx, f'assesor_{i+1}'] = lis[i]
  return data
  
  
