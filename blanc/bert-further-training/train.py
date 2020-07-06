import torch
from transformers import *
import pandas as pd
import os.path
import argparse

from torch.utils.data import TensorDataset, DataLoader
from utils import *
import time

parser = argparse.ArgumentParser()
parser.add_argument('--GPU', dest='gpu_id', type=int,
                    default=0, help='Select gpu id. Default 0')
parser.add_argument('--smp', dest='sample', type=bool,
                    default=False, help='Run on data sample?')
parser.add_argument('--size', dest='s_size', type=int,
                    default=100, help='Select data sample size. Default 100')
parser.add_argument('--batch', dest='batch_size', type=int,
                    default=32, help='Select batch size. Default 32')
parser.add_argument('--epochs', dest='n_epochs', type=int,
                    default=2, help='Select number of epochs. Default 2')

args = parser.parse_args()

# If there's a GPU available...
if args.gpu_id == -1:
    device = torch.device("cpu")
    print('\nUsing CPU.')

elif torch.cuda.is_available():
    device = torch.device(f'cuda: {args.gpu_id}')
    print("\nGPU is available and " + torch.cuda.get_device_name(args.gpu_id) + " in use")
else:
    print('\nNo GPU available, using CPU.')
    device = torch.device("cpu")

# Yandex data
df = pd.read_csv('yandex_parallel.csv', index_col=0)

print(df.head())

# Find lenght of sequence that covers 80% of data
df['total_len'] = df['sentence_len_ru'] + df['sentence_len_en']

max_seq_len = int(df['total_len'].quantile(0.8))
print("\nMaximum sequence lenght is ", max_seq_len)

df = df[df["total_len"] <= max_seq_len]

# Taking a sample to test the model
if args.sample:
    df = df.sample(args.s_size, random_state=1)

df = df.reset_index(drop=True)


model_name = "bert-base-multilingual-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelWithLMHead.from_pretrained(model_name)


input_ids = []
print('\n Tokenization...')

tokenizer.padding_side = 'right'

for i in range(len(df)):
    encoded_pair = tokenizer.encode_plus(df.russian[i], df.english[i],
                                        add_special_tokens=True,
                                        max_length=max_seq_len,
                                        pad_to_max_length=True,
                                        return_token_type_ids=True,
                                        return_attention_mask=True,
                                        )

    input_ids.append(encoded_pair)


batch_size = args.batch_size
all_input_ids = torch.tensor([f['input_ids'] for f in input_ids], dtype=torch.long)
all_token_type_ids = torch.tensor([f['token_type_ids'] for f in input_ids], dtype=torch.long)
all_attention_mask = torch.tensor([f['attention_mask'] for f in input_ids], dtype=torch.long)
all_label_ids = custom_replace(all_input_ids)

data = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label_ids)
paral_dataloader = DataLoader(data,  batch_size=batch_size)
print('\n done loading dataset...')


params = list(model.named_parameters())

n_epochs = args.n_epochs

model.to(device)

lr = 2e-5

optimizer = AdamW(model.parameters(),
                  lr=lr,
                  eps=1e-8
                  )

print('\n number of batches', len(paral_dataloader))

for epoch_i in range(0, n_epochs):

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, n_epochs))
    print('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()

    # Reset the total loss for this epoch.
    total_loss = 0

    training_loss = 0.0
    model.train()
    print('\nnumber of batches', len(paral_dataloader))

    for step, batch in enumerate(paral_dataloader):

        # Progress update every 40 batches.
        if step % 1000 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)

            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(paral_dataloader), elapsed))

        # print('Batch: ', batch)

        optimizer.zero_grad()

        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        token_type_ids = batch[2].to(device)
        label_ids = batch[3].to(device)


        loss, predictions = model(input_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids,
                                 masked_lm_labels=label_ids
                                 )

        # predicted_index = torch.argmax().item()
        # predicted_token = tokenizer.convert_ids_to_tokens(predicted_index)
        # print(predicted_token)

        loss.backward()

        optimizer.step()

        training_loss += loss.item() * input_ids.size(0)

    training_loss /= len(paral_dataloader.dataset)
    print('Epoch: {}, Training Loss: {:.10f}'.format(epoch_i+1, training_loss))

if not os.path.exists('./bert_model/'):
    os.makedirs('./bert_model/')

model.save_pretrained('./bert_model/')
print("saved Bert that was pre-trained")
