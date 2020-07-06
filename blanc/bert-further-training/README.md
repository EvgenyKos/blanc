## BERT-further-pretrainig 

The code is used to pre train DistilBert model on Russian-English corpus. For that Yandex dataset (https://translate.yandex.ru/corpus?lang=en) was used. You will also need to install the HuggingFace libraries for AutoModel: https://huggingface.co/transformers/model_doc/auto.html. 


Steps:
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


