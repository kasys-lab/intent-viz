import torch
import torch.nn as nn
import torch.nn.functional as F

WORD_DIM = 100
STAT_DIM = 78
DROPOUT_RATE = 0.1

class NeuralBase(nn.Module):
    def __init__(self, fc1_dim):
        super(NeuralBase, self).__init__()
        self.fc1 = nn.Linear(fc1_dim, 500)
        self.fc2 = nn.Linear(500, 1000)
        self.fc3 = nn.Linear(1000, 3000)
        self.fc4 = nn.Linear(3000, 1000)
        self.fc5 = nn.Linear(1000, 500)
        self.fc6 = nn.Linear(500, 100)
        self.fc7 = nn.Linear(100, 8)

        self.dropout = nn.Dropout(p=DROPOUT_RATE)

    def _fnn(self, x):
        x = self.fc1(self.dropout(x)) # ReLU: max(x, 0)
        x = F.relu(self.fc2(self.dropout(x)))
        x = F.relu(self.fc3(self.dropout(x)))
        x = F.relu(self.fc4(self.dropout(x)))
        x = F.relu(self.fc5(self.dropout(x)))
        x = F.relu(self.fc6(self.dropout(x)))
        x = F.relu(self.fc7(self.dropout(x)))
        return x


class AttentionBase(NeuralBase):
    def __init__(self, fc1_dim):
        super(AttentionBase, self).__init__(fc1_dim)
        self.att_weight_c =  nn.Linear(WORD_DIM, 1)
        self.att_weight_q =  nn.Linear(WORD_DIM, 1)
        self.att_weight_cq = nn.Linear(WORD_DIM, 1)

    def _attention(self, title, headers):
        cq = []
        for i in range(len(title[0])):
            title_word = title.select(1, i).unsqueeze(1)
            seki = self.att_weight_cq(self.dropout(headers * title_word)).squeeze()
            cq.append(seki)

        cq = torch.stack(cq, dim=-1)

        c_dim = title.shape[1]
        q_dim = headers.shape[1]
        sim = self.att_weight_c(self.dropout(headers)).expand(-1, -1, c_dim) + \
            self.att_weight_q(self.dropout(title)).permute(0, 2, 1).expand(-1, q_dim, -1) + \
            cq

        return sim

    def _t2q_att(self, sim, column_feature):
        c_dim = column_feature.shape[1]
        column_w = F.softmax(torch.max(sim, dim=2)[0], dim=1).unsqueeze(1)
        t2q_att = torch.bmm(column_w, column_feature).squeeze(1)
        t2q_att = t2q_att.unsqueeze(1).expand(-1, c_dim, -1)
        t2q_att = torch.sum(t2q_att, 1)
        return t2q_att

    def _q2t_att(self, sim, title):
        q_dim = title.shape[1]
        title_w = F.softmax(torch.max(sim, dim=1)[0], dim=1).unsqueeze(1)
        q2t_att = torch.bmm(title_w, title).squeeze(1)
        q2t_att = q2t_att.unsqueeze(1).expand(-1, q_dim, -1)
        q2t_att = torch.sum(q2t_att, 1)
        return q2t_att

    def forward(self, batch):
        title = batch[0].unsqueeze(0)
        data = batch[1]
        headers, column_feature = data[:,:WORD_DIM], data[:,WORD_DIM:]
        headers = headers.unsqueeze(0)
        column_feature = column_feature.unsqueeze(0)        

        sim = self._attention(title, headers)
        t2q_att = self._t2q_att(sim, column_feature)
        q2t_att = self._q2t_att(sim, title)
        x = torch.cat((t2q_att, q2t_att), 1)
        return self._fnn(x)

class BiAttention(AttentionBase):
    def __init__(self):
        super(BiAttention, self).__init__(WORD_DIM+STAT_DIM)



class SingleColBertWithFeature(nn.Module):
    def __init__(self, bert):
        super(SingleColBertWithFeature, self).__init__()
        self.bert = bert
        self.fc1 = nn.Linear(846, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, 10)
        self.fc4 = nn.Linear(10, 1)

        self.dropout = nn.Dropout(p=DROPOUT_RATE)

    def forward(self, title_single_col, column_feature):
        bert = self.bert

        ## bertに入力
        outputs = bert(**title_single_col)
        ## outputs.last_hidden_state : (batch_size×30(列数)) × 30(単語数) × 768(BERT出力次元)
        outputs = outputs.last_hidden_state
        ## [CLS]の部分だけを抽出 (batch_size × 30) × 768
        outputs = outputs[:, 0]
        ## サイズを変更し、統計情報ベクトルと結合 x: batch_size × 30 × 846
        x = torch.cat((outputs.unsqueeze(0).float(), column_feature.unsqueeze(0)), 2)

        x = self.fc1(self.dropout(x)) # ReLU: max(x, 0)
        x = F.relu(self.fc2(self.dropout(x)))
        x = F.relu(self.fc3(self.dropout(x)))
        x = F.relu(self.fc4(self.dropout(x)))
        return x.view(-1, 30)
