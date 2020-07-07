## BLANC extention into machine translation quality estimation

This is the extenstion of BLANC-help into machine translation quality estimation. 

Steps:
1. Download the data https://www.quest.dcs.shef.ac.uk/wmt20_files_qe/ru-en.tar.gz and extract into directory.

2. Calculate the relative score by executing python run.py

Full documentation with python run.py 

    arguments:

    -- type TYPE          type of analysis( shuffle, replace or standard)
                          (default standard)
                        
    --model_name NAME     BERT model type (default: bert-base-multilingual-uncased)

    --gap GAP             distance between words to mask during inference
                          (default: 6)
                        
    --min_token_length_normal LEN
                          minimum number of chars in normal tokens to mask,
                          where a normal token is a whole word (default: 4)
                        
    --min_token_length_lead LEN
                          minimum number of chars in lead token to mask, where a
                          lead token begins a word (default: 2)
                        
    --min_token_length_followup LEN
                          minimum number of chars in followup token to mask,
                          where a followup token continues a word (default: 100)
                        
    --device DEVICE       cpu or cuda device (default: cpu)
  
    --n_tokens NUM        number of randomly replcaed tokens in translation
                          (default: 1)
                        
     --base BASE          use base model or further pretrained Bert model.
                          1 for base and 2 for further pretrained.
                          (default: 1)
                        
    -- output_name OUTPUT
                          the name of output file. (default: ru_en_blanc) 

  

 


