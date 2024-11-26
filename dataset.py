import random
import torch
from torch.utils.data import Dataset

class BERTDataset(Dataset):

    def __init__(self, data_pairs, tokenizer, max_len) -> None:
        super().__init__()
        self.data_pairs = data_pairs
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data_pairs)

    def _get_random_pair(self):
        random_idx =  random.randrange(0, len(self.data_pairs))
        return self.data_pairs[random_idx]
    
    def _next_sentence_prediction(self, pair):
        """
        50% times it will produce correct sentence and 50% times incorrect
        """
        if random.random()<0.5:
            return (pair[0], pair[1], 1)
        else:
            return (pair[0], self._get_random_pair()[1], 0)

    def _masked_sentence(self, token_ids):
        """
        15% of token will be MASKED
        """   
        output = []
        output_label = []

        # 15% of the tokens would be replaced
        for i, token_id in enumerate(token_ids):
            prob = random.random()


            if prob < 0.15:
                prob = prob/0.15

                # 80% chance change token to mask token
                if prob < 0.8:
                    output.append(self.tokenizer.mask_token_id) 

                # 10% chance change token to random token
                elif prob < 0.9:
                    output.append(random.randrange(len(self.tokenizer.vocab)))
                
                # 10% chance change token to current token
                else:
                    output.append(token_id)

                output_label.append(token_id)

            else:
                output.append(token_id)
                output_label.append(0)

        return output, output_label

    def __getitem__(self, index):
        
        ## Get pair
        pair = self.data_pairs[index]

        ## (NSP) : Next sentence prediction : (first, seocnd, 1 or 0)
        first_sent, second_sent, is_next_label = self._next_sentence_prediction(pair)

        ## Tokenizer the sentences
        first_token_ids = self.tokenizer(first_sent, add_special_tokens = False)['input_ids']
        second_token_ids = self.tokenizer(second_sent, add_special_tokens = False)['input_ids']

        ## (MLM) : MASK Language Model
        first_masked_ids, first_label_ids = self._masked_sentence(first_token_ids)
        second_masked_ids, second_label_ids = self._masked_sentence(second_token_ids)

        ## segment ids : in first there will be 2 more token [CLS], [SEP], for second only [SEP] will be added
        first_segment_ids = [1] * (len(first_masked_ids) + 2)
        second_segment_ids = [2] * (len(second_masked_ids) + 1)

    
        ## Add CLS and SEP token
        bert_input_ids = torch.cat([
            torch.tensor([self.tokenizer.cls_token_id], dtype = torch.long),
            torch.tensor(first_masked_ids, dtype= torch.long),
            torch.tensor([self.tokenizer.sep_token_id], dtype= torch.long),
            torch.tensor(second_masked_ids, dtype = torch.long),
            torch.tensor([self.tokenizer.sep_token_id], dtype= torch.long)
        ], dim = 0)

        bert_label_ids = torch.cat([
            torch.tensor([self.tokenizer.pad_token_id], dtype = torch.long),
            torch.tensor(first_label_ids, dtype= torch.long),
            torch.tensor([self.tokenizer.pad_token_id], dtype= torch.long),
            torch.tensor(second_label_ids, dtype = torch.long),
            torch.tensor([self.tokenizer.pad_token_id], dtype= torch.long)
        ], dim = 0)

        bert_segment_ids = torch.cat([
            torch.tensor(first_segment_ids, dtype = torch.long),
            torch.tensor(second_segment_ids, dtype = torch.long),
        ], dim = 0)

        ## Trucate if length is greater than max_len
        bert_input_ids = bert_input_ids[:self.max_len]
        bert_label_ids = bert_label_ids[:self.max_len]
        bert_segment_ids = bert_segment_ids[:self.max_len]

        ## check for padding
        num_of_pad_tokens = self.max_len - len(bert_input_ids)

        ## Expand everything to num of tokens
        padding = torch.tensor([self.tokenizer.pad_token_id] * num_of_pad_tokens, dtype = torch.long)
        bert_input_ids = torch.cat([bert_input_ids, padding])
        bert_label_ids = torch.cat([bert_label_ids, padding])
        bert_segment_ids = torch.cat([bert_segment_ids, padding])
        is_next_label = torch.tensor(is_next_label, dtype = torch.long)

        # print(bert_input_ids.shape, bert_input_ids.shape, bert_segment_ids.shape)

        assert bert_input_ids.size() == bert_label_ids.size(), "input ids shape != label ids shape"
        assert bert_input_ids.size() == bert_segment_ids.size(), "input ids shape != segment ids shape"


        inputs = {
            "bert_input": bert_input_ids,
            "segment_input": bert_segment_ids
        }

        labels = {
            "nsp_label": is_next_label,
            "mlm_label": bert_label_ids
        }

        raw = {
            "t1": first_sent,
            "t2": second_sent,
            "masked_t1": self.tokenizer.decode(first_masked_ids),
            "masked_t2": self.tokenizer.decode(second_masked_ids)
        }
        return inputs, labels, raw
        
        
        








