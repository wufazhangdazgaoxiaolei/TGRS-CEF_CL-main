import random
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

random.seed(1)
# dict_label:类别对应表
# dict_label = {"airplane": 0, "automobile": 1, "bird": 2, "cat": 3, "deer": 4,"dog": 5,
#               "frog": 6, "horse": 7, "ship": 8, "truck": 9}
dict_label = {"airplane": 0, "airport": 1, "baseball_diamond": 2, "basketball_court": 3, "beach": 4,"bridge": 5,
              "chaparral": 6, "church": 7, "circular_farmland": 8, "cloud": 9,
              "commercial_area": 10, "dense_residential": 11, "desert": 12, "forest": 13, "freeway": 14,"golf_course": 15,
              "ground_track_field": 16, "harbor": 17, "industrial_area": 18, "intersection": 19,
              "island": 20, "lake": 21, "meadow": 22, "medium_residential": 23, "mobile_home_park": 24,"mountain": 25,
              "overpass": 26, "palace": 27, "parking_lot": 28, "railway": 29,
              "railway_station": 30, "rectangular_farmland": 31, "river": 32, "roundabout": 33, "runway": 34,"sea_ice": 35,
              "ship": 36, "snowberg": 37, "sparse_residential": 38, "stadium": 39,
              "storage_tank": 40, "tennis_court": 41, "terrace": 42, "thermal_power_station": 43, "wetland": 44
              }  # 如果改了分类目标，这里需要修改



def get_img_info(data_dir):
    data_info = list()
    for root, dirs, _ in os.walk(data_dir):
        # 遍历类别
        for sub_dir in dirs:
            img_names = os.listdir(os.path.join(root, sub_dir))
            img_names = list(filter(lambda x: x.endswith('.jpg'), img_names)) # 过滤，剩下.png结尾的文件名
            # 遍历图片
            for i in range(len(img_names)):
                img_name = img_names[i]
                path_img = os.path.join(root, sub_dir, img_name) # 完整图片路径
                label = dict_label[sub_dir] # 获取当前图片的标签
                data_info.append((path_img, int(label))) # 返回 [(path_img1,label1),(path_img2,label2),...]

    return data_info

class MyDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.label_name = dict_label
        self.data_info = get_img_info(data_dir)  # data_info存储所有图片路径和标签
        self.transform = transform

    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')     # 0~255
        if self.transform is not None:
            img = self.transform(img)   # 在这里做transform，转为tensor等等
        return img, label

    def __len__(self):
        return len(self.data_info)


#  指定计算mean和std的图像数据集路径
train_dir = os.path.join('.', 'data/NWPU-RESISC45', "train")

#  图像预处理
train_transform = transforms.Compose([
    transforms.Resize((32, 32)), # 可以改成你图片近似大小或者模型要求大小
    transforms.ToTensor(),
])

train_data = MyDataset(data_dir=train_dir, transform=train_transform)
train_loader = DataLoader(dataset=train_data, batch_size=3000, shuffle=True) # 3000张图片的mean std
train = iter(train_loader).next()[0]  # 3000张图片的mean、std
train_mean = np.mean(train.numpy(), axis=(0, 2, 3))
train_std = np.std(train.numpy(), axis=(0, 2, 3))

print("train_mean:",train_mean)
print("train_std:",train_std)



