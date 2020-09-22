# BLANC extension to machine translation quality estimation

This is an extension of BLANC-help into machine translation quality estimation. The original BLANC version is applied on evaluating the quality of text summarization. I used the original sentence  (Russian language) as a document and translation (English language) as a summary to calculate the relative score. Our assumption was that a model should predict more accurately with the translation rather than with a filler is based on [this paper.](https://arxiv.org/pdf/1906.01502.pdf) To get the scores I tested 3 models: bert-base-multilingual-uncased, bert-base-multilingual-cased and further pretrained Bert base uncased model. Further pretrained Bert model did not improve the results beyond the base model. It was tested on multiple pretrained Bert models with different hyperparameters. At the same time, the "cased" base model predicted 12 times more [UNK] tokens. During the initial testing it was uncovered that Bert was masking [UNK] tokens. It does not give practical reasons to predict words that are unknown in masked sentences. Therefore, I removed [UNK] tokens from masking.    

## Instructions

1. Download the data https://www.quest.dcs.shef.ac.uk/wmt20_files_qe/ru-en.tar.gz and extract into `blanc_mt` directory.

2. Install `requirements.txt`

3. Calculate the relative score by executing `python run.py`.
   
   I used the following specification (dafault values) to achive the highest score:
   
   `model_name = 'bert-base-multilingual-uncased'`,
   
   `gap = 2`,
   
   `min_token_length_normal = 4`, 
   
   `min_token_length_lead = 5`. 
   
   To get the scores on the randomaly shuffled translations sentences and translations with randomly replaced words specify the 'type' (view the documentation below). 

4. Evaluate the results using notebook `blanc_data_analysis.ipynb`.
   
   The highest Pearson correlation was ~0.18 while original BLANC paper had ~0.36. Please, review the notebook for more results.  
   
   
Full documentation with `python run.py`: 

    arguments:

    -- type TYPE          type of analysis( shuffle, replace or standard)
                          (default standard)
                        
    --model_name NAME     BERT model type (default: bert-base-multilingual-uncased)

    --gap GAP             distance between words to mask during inference
                          (default: 2)
                        
    --min_token_length_normal LEN
                          minimum number of chars in normal tokens to mask,
                          where a normal token is a whole word (default: 4)
                        
    --min_token_length_lead LEN
                          minimum number of chars in lead token to mask, where a
                          lead token begins a word (default: 5)
                        
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

  

 


