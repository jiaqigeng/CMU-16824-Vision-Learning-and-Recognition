from torch.utils.data import DataLoader
import torch
from torch.utils.tensorboard import SummaryWriter
import os
from PIL import Image
from torchvision import transforms
from collections import Counter
import torch.nn.functional as F


class ExperimentRunnerBase(object):
    """
    This base class contains the simple train and validation loops for your VQA experiments.
    Anything specific to a particular experiment (Simple or Coattention) should go in the corresponding subclass.
    """

    def __init__(self, train_dataset, val_dataset, model, batch_size, num_epochs, num_data_loader_workers=10, log_validation=False):
        self._model = model
        self._num_epochs = num_epochs
        self._log_freq = 10  # Steps
        self._test_freq = 250  # Steps

        self._train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_data_loader_workers)

        # If you want to, you can shuffle the validation dataset and only use a subset of it to speed up debugging
        self._val_dataset_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_data_loader_workers)

        # Use the GPU if it's available.
        self._cuda = torch.cuda.is_available()

        if self._cuda:
            self._model = self._model.cuda()

        self._log_validation = log_validation
        self.writer = SummaryWriter()
        self.questionIds = val_dataset.questionIds
        self._vqa = val_dataset._vqa
        self._image_filename_pattern = val_dataset._image_filename_pattern
        self._image_dir = val_dataset._image_dir
        self.answer_to_id_map = val_dataset.answer_to_id_map
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.log_idx = 0

    def _optimize(self, predicted_answers, true_answers):
        """
        This gets implemented in the subclasses. Don't implement this here.
        """
        raise NotImplementedError()

    def validate(self, idx):
        ############ 2.8 TODO
        # Should return your validation accuracy
        total = 0.
        correct = 0.

        ans = 0
        for batch_id, batch_data in enumerate(self._val_dataset_loader):
            if batch_id > 10:
                break
            if self._cuda:
                image = batch_data['image'].cuda()
                question_embedding = batch_data['question'].cuda()
                answer_embedding = batch_data['answers'].cuda()
            else:
                image = batch_data['image']
                question_embedding = batch_data['question']
                answer_embedding = batch_data['answers']

            _, answer_majority_voted = torch.max(answer_embedding, 2)
            answer_majority_voted, _ = torch.mode(answer_majority_voted, 1)

            outputs = self._model(image, question_embedding)
            outputs = F.softmax(outputs, dim=1)
            _, predicted_answer = torch.max(outputs.data, 1)
            if batch_id == 0:
                ans = predicted_answer[idx % image.shape[0]]

            ground_truth_answer = answer_majority_voted
            total += ground_truth_answer.shape[0]
            correct += torch.sum(torch.eq(predicted_answer, ground_truth_answer)).item()

        ############
        # if self._log_validation:
        #     ############ 2.9 TODO
        #     # you probably want to plot something here
        #     print()
        #     ############
        return correct / total, ans

    def train(self):
        for epoch in range(self._num_epochs):
            num_batches = len(self._train_dataset_loader)
            for batch_id, batch_data in enumerate(self._train_dataset_loader):
                self._model.train()  # Set the model to train mode
                current_step = epoch * num_batches + batch_id

                ############ 2.6 TODO
                # Run the model and get the ground truth answers that you'll pass to your optimizer
                # This logic should be generic; not specific to either the Simple Baseline or CoAttention.
                if self._cuda:
                    image = batch_data['image'].cuda()
                    question_embedding = batch_data['question'].cuda()
                    answer_embedding = batch_data['answers'].cuda()
                else:
                    image = batch_data['image']
                    question_embedding = batch_data['question']
                    answer_embedding = batch_data['answers']

                _, answer_majority_voted = torch.max(answer_embedding, 2)
                answer_majority_voted, _ = torch.mode(answer_majority_voted, 1)

                predicted_answer = self._model(image, question_embedding)
                ground_truth_answer = answer_majority_voted # TODO
                ############

                # Optimize the model according to the predictions
                loss = self._optimize(predicted_answer, ground_truth_answer)

                if current_step % self._log_freq == 0:
                    print("Epoch: {}, Batch {}/{} has loss {}".format(epoch, batch_id, num_batches, loss))
                    ############ 2.9 TODO
                    # you probably want to plot something here
                    self.writer.add_scalar('train_loss', loss.item(), current_step)
                    ############

                if current_step % self._test_freq == 0:
                    self._model.eval()
                    val_accuracy, predict_ans_idx = self.validate(self.log_idx)
                    idx = self.questionIds[self.log_idx]
                    self.log_idx += 1

                    img_id = self._vqa.loadQA(idx)[0]['image_id']
                    img_name = self._image_filename_pattern.replace("{}", str(img_id).zfill(12))
                    img_path = os.path.join(self._image_dir, img_name)
                    question = self._vqa.qqa[idx]['question']
                    img = Image.open(img_path).convert('RGB')
                    img = self.transform(img)
                    answers = []
                    for ans_dict in self._vqa.loadQA(idx)[0]['answers']:
                        answers.append(ans_dict['answer'])

                    best_answer = Counter(answers).most_common(1)[0][0]
                    predict_ans = "unknown"
                    for answer in self.answer_to_id_map:
                        if self.answer_to_id_map[answer] == int(predict_ans_idx):
                            predict_ans = answer
                            break

                    print("Epoch: {} has val accuracy {}".format(epoch, val_accuracy))
                    ############ 2.9 TODO
                    # you probably want to plot something here
                    self.writer.add_scalar('val accuracy', val_accuracy, current_step)
                    self.writer.add_image("image", img, current_step)
                    self.writer.add_text("question", question, current_step)
                    self.writer.add_text("ground truth answers", best_answer, current_step)
                    self.writer.add_text("predicted answers", predict_ans, current_step)
                    ############

    def save(self):
        torch.save(self._model.state_dict(), "coatt_model.pth")
