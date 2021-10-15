import torch.optim

from student_code.simple_baseline_net import SimpleBaselineNet
from student_code.experiment_runner_base import ExperimentRunnerBase
from student_code.vqa_dataset import VqaDataset
from torchvision import datasets, models, transforms
from PIL import Image


class SimpleBaselineExperimentRunner(ExperimentRunnerBase):
    """
    Sets up the Simple Baseline model for training. This class is specifically responsible for creating the model and optimizing it.
    """

    def __init__(self, train_image_dir, train_question_path, train_annotation_path,
                 test_image_dir, test_question_path, test_annotation_path, batch_size, num_epochs,
                 num_data_loader_workers, cache_location, lr, log_validation):

        ############ 2.3 TODO: set up transform
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        ############

        train_dataset = VqaDataset(image_dir=train_image_dir,
                                   question_json_file_path=train_question_path,
                                   annotation_json_file_path=train_annotation_path,
                                   image_filename_pattern="COCO_train2014_{}.jpg",
                                   transform=transform,
                                   ############ 2.4 TODO: fill in the arguments
                                   # question_word_to_id_map='change this argument',
                                   # answer_to_id_map='change this argument',
                                   ############
                                   )

        val_dataset = VqaDataset(image_dir=test_image_dir,
                                 question_json_file_path=test_question_path,
                                 annotation_json_file_path=test_annotation_path,
                                 image_filename_pattern="COCO_val2014_{}.jpg",
                                 transform=transform,
                                 ############ 2.4 TODO: fill in the arguments
                                 question_word_to_id_map=train_dataset.question_word_to_id_map,
                                 answer_to_id_map=train_dataset.answer_to_id_map,
                                 ############
                                 )

        self._model = SimpleBaselineNet(train_dataset.question_word_list_length, train_dataset.answer_list_length)

        super().__init__(train_dataset, val_dataset, self._model, batch_size, num_epochs,
                         num_data_loader_workers, log_validation)

        ############ 2.5 TODO: set up optimizer
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD([
            {'params': self._model.fc1.parameters(), 'lr': 0.8},
        ], lr=0.01, momentum=0.9)
        ############

    def _optimize(self, predicted_answers, true_answer_ids):
        ############ 2.7 TODO: compute the loss, run back propagation, take optimization step.
        loss = self.criterion(predicted_answers, true_answer_ids)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._model.fc1.weight.data = self._model.fc1.weight.data.clamp(-1500, 1500)
        self._model.fc2.weight.data = self._model.fc2.weight.data.clamp(-20, 20)
        return loss
        ############
