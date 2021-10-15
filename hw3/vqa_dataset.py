import torch
from torch.utils.data import Dataset
from external.vqa.vqa import VQA
import os
import re
from PIL import Image
from torchvision import transforms


class VqaDataset(Dataset):
    """
    Load the VQA dataset using the VQA python API. We provide the necessary subset in the External folder, but you may
    want to reference the full repo (https://github.com/GT-Vision-Lab/VQA) for usage examples.
    """

    def __init__(self, image_dir, question_json_file_path, annotation_json_file_path, image_filename_pattern,
                 transform=None, question_word_to_id_map=None, answer_to_id_map=None, question_word_list_length=5746, answer_list_length=5216,
                 pre_encoder=None, cache_location=None):
        """
        Args:
            image_dir (string): Path to the directory with COCO images
            question_json_file_path (string): Path to the json file containing the question data
            annotation_json_file_path (string): Path to the json file containing the annotations mapping images, questions, and
                answers together
            image_filename_pattern (string): The pattern the filenames of images in this dataset use (eg "COCO_train2014_{}.jpg")
        """
        self._vqa = VQA(annotation_file=annotation_json_file_path, question_file=question_json_file_path)
        self._image_dir = image_dir
        self._image_filename_pattern = image_filename_pattern
        self._transform = transform
        self._max_question_length = 26
        self.questionIds = self._vqa.getQuesIds()

        # Publicly accessible dataset parameters
        self.question_word_list_length = question_word_list_length + 1
        self.unknown_question_word_index = question_word_list_length
        self.answer_list_length = answer_list_length + 1
        self.unknown_answer_index = answer_list_length
        self._pre_encoder = pre_encoder
        self._cache_location = cache_location
        if self._cache_location is not None:
            try:
                os.makedirs(self._cache_location)
            except OSError:
                pass

        # Create the question map if necessary
        if question_word_to_id_map is None:
            ############ 1.6 TODO
            question_sentence_list = []
            for idx in self._vqa.qqa:
                question_sentence_list.append(self._vqa.qqa[idx]['question'])
            question_word_list = self._create_word_list(question_sentence_list)
            question_word_to_id_map = self._create_id_map(question_word_list, self.question_word_list_length-1)
            self.question_word_to_id_map = question_word_to_id_map
            ############
        else:
            self.question_word_to_id_map = question_word_to_id_map

        # Create the answer map if necessary
        if answer_to_id_map is None:
            ############ 1.7 TODO
            answer_sentence_list = []
            for idx in self._vqa.qqa:
                for ans_dict in self._vqa.loadQA(idx)[0]['answers']:
                    answer_sentence_list.append(ans_dict['answer'])

            answer_to_id_map = self._create_id_map(answer_sentence_list, self.answer_list_length-1)
            self.answer_to_id_map = answer_to_id_map
            ############
        else:
            self.answer_to_id_map = answer_to_id_map

    def _create_word_list(self, sentences):
        """
        Turn a list of sentences into a list of processed words (no punctuation, lowercase, etc)
        Args:
            sentences: a list of str, sentences to be splitted into words
        Return:
            A list of str, words from the split, order remained.
        """
        ############ 1.4 TODO
        word_list = []
        for sentence in sentences:
            s = re.sub("[^\w\s]", "", str(sentence).lower())
            word_list.extend(s.split())

        return word_list
        ############

    def _create_id_map(self, word_list, max_list_length):
        """
        Find the most common str in a list, then create a map from str to id (its rank in the frequency)
        Args:
            word_list: a list of str, where the most frequent elements are picked out
            max_list_length: the number of strs picked
        Return:
            A map (dict) from str to id (rank)
        """
        ############ 1.5 TODO
        word_count_map = {}
        id_map = {}

        for word in word_list:
            if word in word_count_map:
                word_count_map[word] += 1
            else:
                word_count_map[word] = 1

        word_count_sorted = sorted(word_count_map, key=word_count_map.get, reverse=True)
        word_selected = word_count_sorted[:max_list_length]

        for id, word in enumerate(word_selected):
            id_map[word] = id

        return id_map
        ############

    def __len__(self):
        ############ 1.8 TODO
        return len(self.questionIds)
        ############

    def __getitem__(self, idx):
        """
        Load an item of the dataset
        Args:
            idx: index of the data item
        Return:
            A dict containing multiple torch tensors for image, question and answers.
        """

        ############ 1.9 TODO
        # figure out the idx-th item of dataset from the VQA API
        idx = self.questionIds[idx]
        img_id = self._vqa.loadQA(idx)[0]['image_id']
        img_name = self._image_filename_pattern.replace("{}", str(img_id).zfill(12))
        img_path = os.path.join(self._image_dir, img_name)
        ############

        if self._cache_location is not None and self._pre_encoder is not None:
            ############ 3.2 TODO
            # implement your caching and loading logic here
            image_feature_cache_path = os.path.join(self._cache_location, img_name.replace(".jpg", ".pt"))

            if os.path.exists(image_feature_cache_path):
                image_features = torch.load(image_feature_cache_path).cuda()
            else:
                img = Image.open(img_path).convert('RGB')
                if self._transform is not None:
                    img_transformed = self._transform(img)
                else:
                    img_transformed = transforms.ToTensor()(img)

                if torch.cuda.is_available():
                    img_transformed = img_transformed.cuda()
                    self._pre_encoder.cuda()

                self._pre_encoder.eval()
                image_features = self._pre_encoder(img_transformed.unsqueeze(0))
                torch.save(image_features, image_feature_cache_path)

            image_input = image_features
            ############
        else:
            ############ 1.9 TODO
            # load the image from disk, apply self._transform (if not None)
            img = Image.open(img_path).convert('RGB')
            if self._transform is not None:
                img_transformed = self._transform(img)
            else:
                img_transformed = transforms.ToTensor()(img)

            image_input = img_transformed
            ############

        ############ 1.9 TODO
        # load and encode the question and answers, convert to torch tensors
        question_embedding = torch.zeros((self._max_question_length, self.question_word_list_length))
        question = self._vqa.qqa[idx]['question']
        question_word_list = self._create_word_list([question])[:self._max_question_length]

        for i, word in enumerate(question_word_list):
            if word in self.question_word_to_id_map:
                question_embedding[i, self.question_word_to_id_map[word]] = 1
            else:
                question_embedding[i, self.unknown_question_word_index] = 1

        answers_embedding = torch.zeros((10, self.answer_list_length))
        for i, ans_dict in enumerate(self._vqa.loadQA(idx)[0]['answers']):
            if ans_dict['answer'] in self.answer_to_id_map:
                answers_embedding[i, self.answer_to_id_map[ans_dict['answer']]] = 1
            else:
                answers_embedding[i, self.unknown_answer_index] = 1

        return {'image': image_input, 'question': question_embedding, 'answers': answers_embedding}
        ############
