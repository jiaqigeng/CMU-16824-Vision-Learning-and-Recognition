import torch
import torch.nn as nn
from external.googlenet.googlenet import googlenet


class SimpleBaselineNet(nn.Module):
    """
    Predicts an answer to a question about an image using the Simple Baseline for Visual Question Answering (Zhou et al, 2017) paper.
    """
    def __init__(self, question_word_list_length, answer_list_length):
        super().__init__()
        ############ 2.2 TODO
        self.google_net = googlenet(pretrained=True)
        self.fc1 = nn.Linear(question_word_list_length, 1024)
        self.fc2 = nn.Linear(2024, answer_list_length)
        ############

    def forward(self, image, question_encoding):
        ############ 2.2 TODO
        question_bag_of_words, _ = torch.max(question_encoding, 1)

        google_net_output = self.google_net(image)
        if type(google_net_output) is tuple:
            image_features = google_net_output[0]
        else:
            image_features = google_net_output

        question_features = self.fc1(question_bag_of_words)
        combined_features = torch.cat((image_features, question_features), 1)
        ans_features = self.fc2(combined_features)
        return ans_features
        ############
