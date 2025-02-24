import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import cv2.ximgproc as ximgproc
class EndoVIS15Dataset(Dataset):
    def __init__(self, dataset_id, transform=None):
        self.dataset_id = dataset_id
        self.transform = transform
        self.vid_path = f'../data/endovis15/Segmentation_Robotic_Testing/Segmentation/Dataset{str(dataset_id)}/Video.avi'
        self.gt_path = f'../data/endovis15/Segmentation_Robotic_Testing_GT/Dataset{str(dataset_id)}/0.avi'
        self.images, self.masks, self.skeletons = self._load_data()
    def _load_data(self):
        images = []
        masks = []
        skeletons = []
        vid_cap = cv2.VideoCapture(self.vid_path)
        gt_cap = cv2.VideoCapture(self.gt_path)
        while True:
            ret_img, frame = vid_cap.read()
            ret_gt, gt_frame = gt_cap.read()
            if not ret_img or not ret_gt:
                break
            image = frame.copy()
            mask = cv2.cvtColor(gt_frame, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
            skeleton = ximgproc.thinning(mask)
            images.append(image)
            masks.append(mask)
            skeletons.append(skeleton)
        vid_cap.release()
        gt_cap.release()
        return images, masks, skeletons
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        skeleton = self.skeletons[idx]
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)
        skeleton = cv2.resize(skeleton, (224, 224), interpolation=cv2.INTER_NEAREST)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
        skeleton_tensor = torch.from_numpy(skeleton).unsqueeze(0).float()
        if self.transform:
            image_tensor = self.transform(image_tensor)
            mask_tensor = self.transform(mask_tensor)
            skeleton_tensor = self.transform(skeleton_tensor)
        return image_tensor, mask_tensor, skeleton_tensor