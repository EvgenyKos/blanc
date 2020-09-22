# BERT-further-pretrainig 

The code is used to further pretrain Bert model on Russian-English corpus. For that Yandex dataset (https://translate.yandex.ru/corpus?lang=en) was used. You will also need to install the HuggingFace libraries for AutoModel: https://huggingface.co/transformers/model_doc/auto.html. The model was initialized with AutoModelWithLMHead instead of BertForMaskedLM (BMLM) on 'bert-base-multilingual-uncased'. I used AutoModel because BMLM was overfitting and predicting the same tokens even after training on 1 epoch. `encode_plus` [tokenizer](https://huggingface.co/transformers/main_classes/tokenizer.html) with inserting special tokens, generating padding and generating segment ids. For labeling, I used `input_ids` and assigned value -100 to special tokens. Without it, the model was predicting [PAD] tokens. Batch size of 64 was used for `DataLoader`. `AdamW` [optimizer](https://huggingface.co/transformers/main_classes/optimizer_schedules.html) with learning rate 2e-5 and epsilon 1e-8. The training was done on 1 epoch. After testing, there was no effect of the number of epochs on BLANC score. You can download pytorch model [here](https://storage.cloud.google.com/bert_qe/bert_model/pytorch_model.bin) and configuration file [here](https://storage.cloud.google.com/bert_qe/bert_model/config.json).        


## Instructions

1. Download and extract corpuses into bert-further-training directory. Create new enviroment and install requirements.txt. 

   1.1. `pip install -r requirements.txt`
   
   1.2. Prepare the data by executing `python3 data_prep.py`. A file `yandex_parallel.csv` will be saved into your directory. 

2. Further train Bert model by executing `python train.py`. View the documentation below. 

The code creates directory directory '/bert_model/' if not exists and saves model there. 

Full documentation with `python train.py`: 

    arguments:

    -- GPU NUMBER         cpu (-1) or cuda device (0 or more if multiple gpu's available)
                          (default: 0)
                        
    --smp SAMPLE          use data sample or not. (default: False)

    --size SAMPLE_SIZE    sample size if True for sample (default: 100)
                        
    --batch SIZE          batch size for training (default: 32)
                        
    --epochs EPOCHS       number of epochs to train (default: 2)
  


