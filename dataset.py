from torch.utils.data import Dataset

class QAdataset(Dataset):

    """
    QA dataset for tokenized questions and corresponding answers to train question answering model

    """
    def __init__(self,enc_ids,enc_masks,dec_ids,dec_masks,is_training):

        assert len(enc_ids)==len(enc_masks)
        assert len(dec_ids)==len(dec_masks)

        self.enc_ids=enc_ids
        self.enc_masks=enc_masks

        self.dec_ids=dec_ids
        self.dec_masks=dec_masks

        self.is_training=is_training


    def __len__(self):
        return len(self.enc_ids)
    

    def __getitem__(self, index):
        if self.is_training:
            return self.enc_ids[index],self.enc_masks[index],self.dec_ids[index],self.dec_masks[index]
        else:
            return self.enc_ids[index],self.enc_mask[index]
        

class DprDataset(Dataset):

    """
    Dataset class to hold tokenized Questions and corresponding Answers

    """
    def __init__(self,ques_ids,ques_masks,ans_ids, ans_masks):
        assert len(ques_ids)==len(ques_masks)
        assert len(ans_ids)==len(ans_masks)

        self.ques_ids=ques_ids
        self.ques_mask=ques_masks

        self.ans_ids=ans_ids
        self.ans_masks=ans_masks

    def __len__(self):
        return len(self.ques_ids)
    
    def __getitem__(self, index):
        return self.ques_ids[index],self.ques_mask[index],self.ans_ids[index],self.ans_masks[index]
    

class PassageDataset(Dataset):
    """
    Dataset class to load and process the tokenized corpus containg the passages which act as context for different questions and answers

    Attributes: 
    Passage_ids: ids for identification of passage 
    input_ids: input_ids of the context post tokenization
    attention_masks: attention_mask of the context post tokenization

    """

    def __init__(self,passage_ids,input_ids,attention_masks):
        self.passage_ids = passage_ids
        self.input_ids = input_ids
        self.attention_masks = attention_masks

    
    def __len__(self):
        return len(self.passage_ids)
    
    def __getitem__(self, id):
        if id in self.passage_ids:
            return self.input_ids[id], self.attention_masks[id]
    
    def get_by_id(self,id):
        return self.__getitem__(id)