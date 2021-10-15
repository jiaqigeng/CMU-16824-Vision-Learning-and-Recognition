import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.wx = nn.Linear(512, 512)
        self.wg = nn.Linear(512, 512)
        self.whx = nn.Linear(512, 1)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, X, g):
        if type(g) is int:
            H = self.wx(X)
        else:
            g = g.unsqueeze(1)
            H = self.wx(X) + self.wg(g)

        H = self.activation(H)
        H = self.dropout(H)
        alpha = F.softmax(self.whx(H), dim=1)
        x_hat = torch.sum(alpha * X, dim=1)
        return x_hat


class AlternatingCoAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_attention = Attention()
        self.question_attention_1 = Attention()
        self.question_attention_2 = Attention()

    def forward(self, question_features, image_features):
        s = self.question_attention_1(question_features, 0)
        v = self.image_attention(image_features, s)
        out = self.question_attention_2(question_features, v)
        return v, out


class CoattentionNet(nn.Module):
    """
    Predicts an answer to a question about an image using the Hierarchical Question-Image Co-Attention
    for Visual Question Answering (Lu et al, 2017) paper.
    """
    def __init__(self, question_word_list_length, answer_list_length):
        super().__init__()
        ############ 3.3 TODO
        self.word_fc = nn.Linear(question_word_list_length, 512)

        self.unigram_conv = nn.Conv1d(512, 512, kernel_size=1, padding=0)
        self.bigram_conv = nn.Conv1d(512, 512, kernel_size=2, padding=0)
        self.trigram_conv = nn.Conv1d(512, 512, kernel_size=3, padding=1)

        self.lstm = nn.LSTM(input_size=512, hidden_size=512, batch_first=True)

        self.word_level = AlternatingCoAttention()
        self.phrase_level = AlternatingCoAttention()
        self.question_level = AlternatingCoAttention()

        self.ww = nn.Linear(512, 512)
        self.wp = nn.Linear(1024, 512)
        self.ws = nn.Linear(1024, 1024)
        self.wh = nn.Linear(1024, answer_list_length)

        self.activation = nn.Tanh()
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        self.dropout3 = nn.Dropout(p=0.5)
        self.dropout4 = nn.Dropout(p=0.5)
        ############

    def forward(self, image, question_encoding):
        ############ 3.3 TODO
        image_features = image.reshape((image.shape[0], -1, image.shape[2]))
        word_features = self.word_fc(question_encoding)
        word_features = self.activation(word_features)
        # word_features = self.dropout1(word_features)

        word_features = word_features.reshape((word_features.shape[0],
                                               word_features.shape[2],
                                               word_features.shape[1]))

        unigram_features = self.unigram_conv(word_features)
        unigram_features = self.activation(unigram_features)
        bigram_features = self.bigram_conv(F.pad(word_features, (0, 1)))
        bigram_features = self.activation(bigram_features)
        trigram_features = self.trigram_conv(word_features)
        trigram_features = self.activation(trigram_features)

        combined_features = torch.stack((unigram_features, bigram_features, trigram_features))
        combined_features, _ = torch.max(combined_features, dim=0)
        combined_features = combined_features.reshape((word_features.shape[0], word_features.shape[2],
                                                       word_features.shape[1]))
        question_features = self.lstm(combined_features)[0]

        word_features = word_features.reshape((word_features.shape[0],
                                               word_features.shape[2],
                                               word_features.shape[1]))

        vw, qw = self.word_level(word_features, image_features)
        vp, qp = self.phrase_level(combined_features, image_features)
        vs, qs = self.question_level(question_features, image_features)

        hw = self.ww(qw + vw)
        hw = self.activation(hw)
        hw = self.dropout2(hw)
        hp = self.wp(torch.cat((qp + vp, hw), dim=1))
        hp = self.activation(hp)
        hp = self.dropout3(hp)
        hs = self.ws(torch.cat((qs + vs, hp), dim=1))
        hs = self.activation(hs)
        hs = self.dropout4(hs)

        return self.wh(hs)
        ############
