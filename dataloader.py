from torchvision.transforms.functional import pil_to_tensor
from torch.utils.data import Dataset
from datasets import load_dataset


class DatasetParser(Dataset):
    def __init__(self, mode):

        # super init
        super().__init__()

        # load the finance data for vision language model - QA pairs
        data = load_dataset("sujet-ai/Sujet-Finance-QA-Vision-100k")

        if mode == 'train':
            self.data = data['train']
        else:
            self.data = data['test']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        if 'image' in self.data[index].keys():

            # img_path / instruction
            img_pil = self.data[index]['image']
            conversations = eval(self.data[index]['qa_pairs'])

            # img path -> img
            img_tensor = pil_to_tensor(img_pil)

            return {'image': img_tensor, 'conversations': conversations}

        else:
            return {'conversations': self.data[index]['conversations']}
