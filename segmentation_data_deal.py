import os
import torchvision
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.utils import shuffle
class My_Segmentation_data:
    def __init__(self, step_size = 64, crop_size = (224, 224), imagenet_mean = torch.tensor([[0.485, 0.456, 0.406]]), imagenet_std = torch.tensor([0.229, 0.224, 0.225]),
                 test_num = 100, Augmentation_num=4, random_seed=42):
        self.step_size = step_size
        self.crop_size = crop_size
        self.Augmentation_num = Augmentation_num
        assert self.step_size % self.Augmentation_num == 0, f"请提供为 Augmentation_num:{self.Augmentation_num} 的倍数的step_size"
        self.pic_num_perbatch = self.step_size // self.Augmentation_num
        self.imagenet_mean = imagenet_mean
        self.imagenet_std = imagenet_std
        self.test_num = test_num
        self.current_pic_ord = 0
        self.test_current_pic_ord = 0
        self.random_seed = random_seed

        
    def reset(self):
        self.current_pic_ord = 0
        self.test_current_pic_ord = 0
    def load_data(self, voc_dir='data/VOCdevkit/VOC2012', 
                  VOC_COLORMAP = [[0,0,0],[128,0,0],[0,128,0],[128,128,0],
               [0,0,128],[128,0,128],[0,128,128],[128,128,128],
               [64,0,0],[192,0,0],[64,128,0],[192,128,0],
               [64,0,128],[192,0,128],[64,128,128],[192,128,128],
               [0,64,0],[128,64,0],[0,192,0],[128,192,0],
               [0,64,128]],
               VOC_CLASSES = ['background','aeroplane','bicycle','bird','boat','bottle','bus',
              'car','cat','chair','cow','diningtable','dog','horse','motorbike',
              'person','potted plant','sheep','sofa','train','tv/monitor']  ):
        txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation', 'train.txt')
        # 设定图像的读取模式为RGB
        mode = torchvision.io.image.ImageReadMode.RGB
        # 读取txt_fname文件，并将文件中的内容分割成一个个的文件名，然后存储在images列表中   
        with open(txt_fname,'r') as f:
            images = f.read().split()
        # 创建两个空列表，分别用于存储特征和标签
        self.train_features, self.train_labels = [],[]
        # 对于images中的每个文件名
        for i, fname in enumerate(images):
            # 使用torchvision.io.read_image读取对应的图像文件，然后添加到features列表中
            self.train_features.append(torchvision.io.read_image(os.path.join(voc_dir,'JPEGImages',f'{fname}.jpg')))  
            # 使用torchvision.io.read_image读取对应的标签图像文件，然后添加到labels列表中
            self.train_labels.append(torchvision.io.read_image(os.path.join(voc_dir,'SegmentationClass',f'{fname}.png'),mode))  

        txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation', 'val.txt')
        with open(txt_fname,'r') as f:
            images = f.read().split()

        self.test_features, self.test_labels = [],[]
        # 对于images中的每个文件名
        for i, fname in enumerate(images):
            # 使用torchvision.io.read_image读取对应的图像文件，然后添加到features列表中
            self.test_features.append(torchvision.io.read_image(os.path.join(voc_dir,'JPEGImages',f'{fname}.jpg')))  
            # 使用torchvision.io.read_image读取对应的标签图像文件，然后添加到labels列表中
            self.test_labels.append(torchvision.io.read_image(os.path.join(voc_dir,'SegmentationClass',f'{fname}.png'),mode))  
        
        ALL_FEATRUE = self.train_features+self.test_features
        ALL_LABELS = self.train_labels+self.test_labels
        # indices = np.arange(len(ALL_FEATRUE))
        # np.random.shuffle(indices)  # 打乱索引
        # ALL_FEATRUE = ALL_FEATRUE[indices]
        # ALL_LABELS = ALL_LABELS[indices]
        ALL_FEATRUE, ALL_LABELS = shuffle(ALL_FEATRUE, ALL_LABELS, random_state=self.random_seed)

        self.train_features = ALL_FEATRUE[:-self.test_num]
        self.train_labels = ALL_LABELS[:-self.test_num]
        self.test_features = ALL_FEATRUE[-self.test_num:]
        self.test_labels = ALL_LABELS[-self.test_num:]



        self.VOC_COLORMAP = VOC_COLORMAP
        self.VOC_CLASSES = VOC_CLASSES
        self.train_sample_sum = len(self.train_features)*self.Augmentation_num
        self.test_sample_sum = len(self.test_features)
        print(f"数据已载入， 训练集（增强）一共有： {self.train_sample_sum} 个sample； 测试集（增强）：{self.Augmentation_num*self.test_sample_sum} 个sample。")
        self.colormap2label = torch.zeros(256**3, dtype=torch.long)
        # 对于VOC_COLORMAP中的每个颜色值（colormap）
        for i, colormap in enumerate(self.VOC_COLORMAP):
            # 计算颜色值的一维索引，并将这个索引对应的位置设为i。这样，给定一个颜色值，我们就可以通过这个映射找到对应的类别索引
            self.colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i  
        self.current_pic_ord = 0
        self.test_current_pic_ord = 0
    
    def img_norm_process(self, img):
        img = img.permute(1,2,0)
        img = img/255
        img = img - self.imagenet_mean
        img = img / self.imagenet_std
        return img
    def voc_rand_crop(self, feature, label, height, width):
        """随即裁剪特征和标签图像"""
        # 调用RandomCrop的get_params方法，随机生成一个裁剪框。裁剪框的大小是(height, width)
        # rect拿到特征的框
        rect = torchvision.transforms.RandomCrop.get_params(feature,(height,width))   
        # 根据生成的裁剪框，对特征图像进行裁剪
        # 拿到框中的特征和标号
        feature = torchvision.transforms.functional.crop(feature, *rect)
        # 根据生成的裁剪框，对标签图像进行裁剪。注意，我们是在同一个裁剪框下裁剪特征图像和标签图像，以保证它们对应的位置仍然是对齐的
        label = torchvision.transforms.functional.crop(label,*rect)
        # 返回裁剪后的特征图像和标签图像
        return feature, label

    def show_normed_image(self, image, title=''):
        # image = self.img_norm_process(image)
        if image.shape[2] != 3:
            image = image.permute(1,2,0)
        # image is [H, W, 3]
        plt.rcParams['figure.figsize'] = [5, 5]
        assert image.shape[2] == 3
        plt.imshow(torch.clip((image * self.imagenet_std + self.imagenet_mean) * 255, 0, 255).int())
        plt.title(title, fontsize=16)
        plt.axis('off')
    def show_orin_image(self, image, title=''):
        # image = self.img_norm_process(image)
        if image.shape[2] != 3:
            image = image.permute(1,2,0)
        # image is [H, W, 3]
        plt.rcParams['figure.figsize'] = [5, 5]
        assert image.shape[2] == 3
        plt.imshow(image)
        plt.title(title, fontsize=16)
        plt.axis('off')
    def show_target_img(self, target):
        target = target.unsqueeze(0)
        # target = target/255
        mask = target!=0
        target = (target+15)*5 *mask
        target = target.expand(3, -1, -1)
        self.show_orin_image(target)


    def label_2_target(self, label):
        label = label.permute(1,2,0).numpy().astype('int32')
        # 计算colormap中每个像素的颜色值对应的一维索引。这里的索引计算方式和上一个函数中的是一致的
        idx = ((label[:,:,0] * 256 + label[:,:,1]) * 256 + label[:,:,2])  
        # 使用colormap2label这个映射将索引映射为对应的类别索引，并返回
        return self.colormap2label[idx]
    
        
    
    def next_batch(self):
        if self.current_pic_ord+self.pic_num_perbatch > len(self.train_features):
            self.current_pic_ord = 0
        
        image = torch.empty(self.step_size, 3, self.crop_size[0], self.crop_size[1])
        target = torch.empty(self.step_size, self.crop_size[0], self.crop_size[1],dtype=torch.long)
        for p_ in range(self.pic_num_perbatch):
            p_o = self.current_pic_ord + p_

            while self.train_features[p_o].shape[1] <224 or self.train_features[p_o].shape[2]<224:
                self.current_pic_ord += 1
                p_o = self.current_pic_ord + p_
                if p_o>=len(self.train_features):
                    self.current_pic_ord = 0
                    p_o = self.current_pic_ord + p_
            
            feature =  self.img_norm_process(self.train_features[p_o]).permute(2,0,1)
            label = self.train_labels[p_o]
            for i_ in range(self.Augmentation_num):
                img_, lab_ = self.voc_rand_crop(feature=feature, label=label, height=self.crop_size[0], width=self.crop_size[1])
                image[p_*self.Augmentation_num + i_], target[p_*self.Augmentation_num + i_] = img_,self.label_2_target(lab_).squeeze(0)
        self.current_pic_ord += self.pic_num_perbatch
        return image, target
    
    def test_next_batch(self):

        if self.test_current_pic_ord+self.pic_num_perbatch > len(self.test_features):
            return None, None, False
        
        image = torch.empty(self.step_size, 3, self.crop_size[0], self.crop_size[1])
        target = torch.empty(self.step_size, self.crop_size[0], self.crop_size[1],dtype=torch.long)
        for p_ in range(self.pic_num_perbatch):
            p_o = self.test_current_pic_ord + p_
            while self.test_features[p_o].shape[1] <224 or self.test_features[p_o].shape[2]<224:
                self.test_current_pic_ord += 1
                p_o = self.test_current_pic_ord + p_
                if p_o>=len(self.test_features):
                    return None, None, False
            feature =  self.img_norm_process(self.test_features[p_o]).permute(2,0,1)
            label = self.test_labels[p_o]
            for i_ in range(self.Augmentation_num):
                img_, lab_ = self.voc_rand_crop(feature=feature, label=label, height=self.crop_size[0], width=self.crop_size[1])
                image[p_*self.Augmentation_num + i_], target[p_*self.Augmentation_num + i_] = img_,self.label_2_target(lab_).squeeze(0)
        self.test_current_pic_ord += self.pic_num_perbatch
        return image, target, True


def test():
    Data_ = My_Segmentation_data(step_size=8, Augmentation_num=4, crop_size=(224,224))
    Data_.load_data('data/VOCdevkit/VOC2012')
    
    # img_, tar_ = Data_.next_batch()
    while True:
        img_, tar_ = Data_.next_batch()
    c = 1



if '__main__'==__name__:
    test()