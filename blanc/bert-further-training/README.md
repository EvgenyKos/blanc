# BERT-further-pretrainig 

The code is used to further pretrain Bert model on Russian-English corpus. For that Yandex dataset (https://translate.yandex.ru/corpus?lang=en) was used. You will also need to install the HuggingFace libraries for AutoModel: https://huggingface.co/transformers/model_doc/auto.html. The model was initialized with AutoModelWithLMHead instead of BertForMaskedLM (BMLM) on 'bert-base-multilingual-uncased'. I used AutoModel because BMLM was overfitting and predicting the same tokens even after training on 1 epoch. `encode_plus` [tokenizer](https://huggingface.co/transformers/main_classes/tokenizer.html) with inserting special tokens, generating padding and generating segment ids. For labeling, I used `input_ids` and assigned value -100 to special tokens. Without it, the model was predicting [PAD] tokens. Batch size of 64 was used for `DataLoader`. `AdamW` [optimizer](https://huggingface.co/transformers/main_classes/optimizer_schedules.html) with learning rate 2e-5 and epsilon 1e-8. The training was done on 1 epoch. After testing, there was no effect of the number of epochs on BLANC score.        


## Instructions

1. Download and extract corpuses into directory. Create new enviroment and install requirements.txt

   1.1. pip install -r requirements.txt

2. Training

   2.1. Specicy GPU (default 0). For CPU type --GPU -1
   
        python3 train.py --GPU 2
      
      
   2.2. Specify batch size (default 32)
      
        python3 train.py --batch 64
      
   2.3. Specify number of epochs (default 2)
      
        python3 dtrain.py --epochs 5

The code creates directory directory '/bert_model/' if not exists and saves model there. 


