'''
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-30

The code is designed for scenarios such as disentanglement-based methods where it is necessary to ensure an equal number of positive and negative samples.
'''

import torch
import random
import numpy as np
from dataset.abstract_dataset import DeepfakeAbstractBaseDataset


class pairDataset(DeepfakeAbstractBaseDataset):
    def __init__(self, config=None, mode='train', indices = None):
        super().__init__(config, mode, indices)
        # self.indices = indices
        self.filterd_image_list = self.image_list
        self.filterd_label_list = self.label_list
        # use the subset indices to extract the new image_list and label_list
        # if self.indices is not None:
        #     # print("before indices")
        #     print("len(self.image_list):", len(self.image_list))
        #     print("len(self.label_list):", len(self.label_list))

        #     self.filterd_image_list = [self.image_list[i] for i in self.indices]
        #     self.filterd_label_list = [self.label_list[i] for i in self.indices]

        #     print("len(self.filterd_image_list):", len(self.filterd_image_list))
        #     print("len(self.filterd_label_list):", len(self.filterd_label_list))


        self.fake_imglist = [(img, label, 1) for img, label in zip(self.filterd_image_list, self.filterd_label_list) if label != 0]
        self.real_imglist = [(img, label, 0) for img, label in zip(self.filterd_image_list, self.filterd_label_list) if label == 0]

        # debug: check the number of fake and real images (real should be circa 3 times less than fake)
        print("len(fake_imglist):", len(self.fake_imglist)) # 9000 (1st iteration), 7200 (2nd iteration)
        print("len(real_imglist):", len(self.real_imglist)) # 3000 (1st iteration), 1800 (2n iteration)

        # print("fake_imglist[0:10]: ", self.fake_imglist[0:10])
        # print("real_imglist[0:10]: ", self.real_imglist[0:10])

        print("len(fake_imglist + real_imglist):", self.__len__())


        self.fake_indices = [i for i, label in enumerate(self.label_list) if label != 0]
        self.real_indices = [i for i, label in enumerate(self.label_list) if label == 0]
        print("len(fake_indices):", len(self.fake_indices))
        print("len(real_indices):", len(self.real_indices))
        # Print fake and real image indices
        # print("fake_imglist indices:", self.fake_indices)
        # print("real_imglist indices:", self.real_indices)

        # breakpoint()
        # len(fake_imglist): 7237
        # len(real_imglist): 2363
        # tot real: 9600
        # len(fake_imglist): 1763
        # len(real_imglist): 637
        # tot fake: 2400
        # tot imgs: 12000 -> train dataset size

    def __getitem__(self, index, norm=True):
        # print("index: ", index)
       
        if index in self.fake_indices:
            # print("Fake image")
            # get the position in the list of the fake_index
            index = self.fake_indices.index(index) # get the index of the fake image in the fake_imglist
            # print("corresponding index in fake_imglist:", index)
            fake_image_path, fake_spe_label, fake_label = self.fake_imglist[index]

            # Randomly select a real image
            real_index = random.randint(0, len(self.real_imglist) - 1)
            real_image_path, real_spe_label, real_label = self.real_imglist[real_index]

        elif index in self.real_indices:
            # print("Real image")
            # get the position in the list of the real_index
            real_index = self.real_indices.index(index)
            # print("corresponding index in real_imglist:", index)
            real_image_path, real_spe_label, real_label = self.real_imglist[real_index]

            # Randomly select a fake image
            fake_index = random.randint(0, len(self.fake_imglist) - 1)
            fake_image_path, fake_spe_label, fake_label = self.fake_imglist[fake_index]
        else: 
            raise IndexError(f"index {index} is not in fake_indices or real_indices")

        # Get the fake and real image paths and labels
        # fake_image_path, fake_spe_label, fake_label = self.fake_imglist[index]
        # real_index = random.randint(0, len(self.real_imglist) - 1)  # Randomly select a real image
        # real_image_path, real_spe_label, real_label = self.real_imglist[real_index]

        # Get the mask and landmark paths for fake and real images
        fake_mask_path = fake_image_path.replace('frames', 'masks')
        fake_landmark_path = fake_image_path.replace('frames', 'landmarks').replace('.png', '.npy')
        
        real_mask_path = real_image_path.replace('frames', 'masks')
        real_landmark_path = real_image_path.replace('frames', 'landmarks').replace('.png', '.npy')

        # Load the fake and real images
        fake_image = self.load_rgb(fake_image_path)
        real_image = self.load_rgb(real_image_path)

        fake_image = np.array(fake_image)  # Convert to numpy array for data augmentation
        real_image = np.array(real_image)  # Convert to numpy array for data augmentation

        # Load mask and landmark (if needed) for fake and real images
        if self.config['with_mask']:
            fake_mask = self.load_mask(fake_mask_path)
            real_mask = self.load_mask(real_mask_path)
        else:
            fake_mask, real_mask = None, None

        if self.config['with_landmark']:
            fake_landmarks = self.load_landmark(fake_landmark_path)
            real_landmarks = self.load_landmark(real_landmark_path)
        else:
            fake_landmarks, real_landmarks = None, None

        # Do transforms for fake and real images
        fake_image_trans, fake_landmarks_trans, fake_mask_trans = self.data_aug(fake_image, fake_landmarks, fake_mask)
        real_image_trans, real_landmarks_trans, real_mask_trans = self.data_aug(real_image, real_landmarks, real_mask)

        if not norm:
            return {"fake": (fake_image_trans, fake_label), 
                    "real": (real_image_trans, real_label)}

        # To tensor and normalize for fake and real images
        fake_image_trans = self.normalize(self.to_tensor(fake_image_trans))
        real_image_trans = self.normalize(self.to_tensor(real_image_trans))

        # Convert landmarks and masks to tensors if they exist
        if self.config['with_landmark']:
            fake_landmarks_trans = torch.from_numpy(fake_landmarks_trans)
            real_landmarks_trans = torch.from_numpy(real_landmarks_trans)
        if self.config['with_mask']:
            fake_mask_trans = torch.from_numpy(fake_mask_trans)
            real_mask_trans = torch.from_numpy(real_mask_trans)

        return {"fake": (fake_image_trans, fake_label, fake_spe_label, fake_landmarks_trans, fake_mask_trans), 
                "real": (real_image_trans, real_label, real_spe_label, real_landmarks_trans, real_mask_trans)}

    def __len__(self):
        # return len(self.fake_imglist)
        if len(self.fake_imglist) == len(self.real_imglist):
            return len(self.fake_imglist) # Return the number of fake images which is equal to the number of real images
        else: 
            # return len(self.fake_imglist) + len(self.real_imglist) # in my case the number of fake images is 3 times the number of real images
            # print(f"len(fake_imglist): {len(self.fake_imglist)}, len(real_imglist): {len(self.real_imglist)}, len_total = {len(self.fake_imglist) + len(self.real_imglist)}")
            return len(self.fake_imglist) + len(self.real_imglist)

    @staticmethod
    def collate_fn(batch):
        """
        Collate a batch of data points.

        Args:
            batch (list): A list of tuples containing the image tensor, the label tensor,
                        the landmark tensor, and the mask tensor.

        Returns:
            A tuple containing the image tensor, the label tensor, the landmark tensor,
            and the mask tensor.
        """
        # Separate the image, label, landmark, and mask tensors for fake and real data
        fake_images, fake_labels, fake_spe_labels, fake_landmarks, fake_masks = zip(*[data["fake"] for data in batch])
        real_images, real_labels, real_spe_labels, real_landmarks, real_masks = zip(*[data["real"] for data in batch])

        # Stack the image, label, landmark, and mask tensors for fake and real data
        fake_images = torch.stack(fake_images, dim=0)
        fake_labels = torch.LongTensor(fake_labels)
        fake_spe_labels = torch.LongTensor(fake_spe_labels)
        real_images = torch.stack(real_images, dim=0)
        real_labels = torch.LongTensor(real_labels)
        real_spe_labels = torch.LongTensor(real_spe_labels)

        # Special case for landmarks and masks if they are None
        if fake_landmarks[0] is not None:
            fake_landmarks = torch.stack(fake_landmarks, dim=0)
        else:
            fake_landmarks = None
        if real_landmarks[0] is not None:
            real_landmarks = torch.stack(real_landmarks, dim=0)
        else:
            real_landmarks = None

        if fake_masks[0] is not None:
            fake_masks = torch.stack(fake_masks, dim=0)
        else:
            fake_masks = None
        if real_masks[0] is not None:
            real_masks = torch.stack(real_masks, dim=0)
        else:
            real_masks = None

        # Combine the fake and real tensors and create a dictionary of the tensors
        images = torch.cat([real_images, fake_images], dim=0)
        labels = torch.cat([real_labels, fake_labels], dim=0)
        spe_labels = torch.cat([real_spe_labels, fake_spe_labels], dim=0)
        
        if fake_landmarks is not None and real_landmarks is not None:
            landmarks = torch.cat([real_landmarks, fake_landmarks], dim=0)
        else:
            landmarks = None

        if fake_masks is not None and real_masks is not None:
            masks = torch.cat([real_masks, fake_masks], dim=0)
        else:
            masks = None

        data_dict = {
            'image': images,
            'label': labels,
            'label_spe': spe_labels,
            'landmark': landmarks,
            'mask': masks
        }
        return data_dict

