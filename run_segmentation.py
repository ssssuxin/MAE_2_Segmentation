from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
from MAE_For_Segmtation_model import MAE_For_Segmentation
import segmentation_data_deal 
# def show_image(image, title=''):
#     # image is [H, W, 3]
#     assert image.shape[2] == 3
#     plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
#     plt.title(title, fontsize=16)
#     plt.axis('off')
#     return
def load_model_weight(model, model_dir):
    ## todo加载模型参数
    sd_hf = torch.load(model_dir, weights_only=True)
    # sd = model.state_dict()
    msg = model.load_state_dict(sd_hf, strict=False)
    print(msg)


batch_size = 16
step_size = 4
data_dir = "data/VOCdevkit/VOC2012"
Data_ = segmentation_data_deal.My_Segmentation_data(step_size=step_size, Augmentation_num=4, crop_size=(224,224))#可视化
Data_.load_data(data_dir)
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
img_dir = "ped2.jpg"
model_dir = "model/model_weights_20250424_115333_epoch2_valloss0.5402.pth"
img_size = (224, 224)
VOC_COLORMAP = [[0,0,0],[128,0,0],[0,128,0],[128,128,0],
               [0,0,128],[128,0,128],[0,128,128],[128,128,128],
               [64,0,0],[192,0,0],[64,128,0],[192,128,0],
               [64,0,128],[192,0,128],[64,128,128],[192,128,128],
               [0,64,0],[128,64,0],[0,192,0],[128,192,0],
               [0,64,128]]
VOC_COLORMAP = torch.tensor(VOC_COLORMAP, dtype=torch.uint8).to(device=device)

img = Image.open(img_dir)
img = img.resize(img_size)
or_pic = torch.tensor(np.array(img)).unsqueeze(0).to(device=device)
img = np.array(img) / 255.

assert img.shape == (224, 224, 3)
imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])
img = img - imagenet_mean
img = img / imagenet_std
# plt.rcParams['figure.figsize'] = [5, 5]
# show_image(torch.tensor(img))
# img.unsqueeze(0)
img = torch.tensor(img)

img = img.to(device=device)
img = img.unsqueeze(0).permute(0, 3, 1, 2)
num_classes = 21 #VOC2012里面分了20个类— + 1背景
loss_use_bg = True #False表示不用背景参与loss构建 True表示使用
model = MAE_For_Segmentation(num_classes=num_classes, loss_use_bg=loss_use_bg)
load_model_weight(model=model, model_dir=model_dir)

model.to(device=device)
model.eval()
# model.requires_grad_
# 回来把这个写一下，读test图片数据，生成照片
# sample_num = Data_.test_sample_sum
# for i_ in range(sample_num):
# while True:
#         X,Y,flag = test_data.test_next_batch()
#         if not flag:
#             break
#         X, Y = X.to(device), Y.to(device)
#         loss, _ = model(X, Y)
#         mask = loss != 0
#         loss_sum += loss.sum().item()
#         loss_num += mask.sum().item()
#         num_sample_count += X.shape[0]

count = 0
while True:
    X,Y,flag = Data_.test_next_batch()
    if not flag:
        break
    X, Y = X.to(device), Y.to(device)
    _, logits = model(X.float())
    result = torch.argmax(logits, dim=1)

    result = VOC_COLORMAP[result]

    # or_pic = or_pic.permute( 2, 0, 1)
    # result = result.permute(0, 3, 1, 2)
    alpha = 0.4
    X = torch.clip((X.permute(0, 2, 3, 1) * Data_.imagenet_std.to(device) + Data_.imagenet_mean.to(device)) * 255, 0, 255).int()
    final = (alpha*X + (1-alpha)*result).int().permute(0,3,1,2).to("cpu")
    # Data_.show_orin_image(final[0])
    img_np = final.permute(0, 2, 3, 1).numpy().astype(np.uint8)
    for i in range(img_np.shape[0]):
        image = Image.fromarray(img_np[i])
        image.save(f"Result_img/{count * X.shape[0] + i}_img.jpg")
    count += 1

x=1



def main():
    pass

if "__main__"==__name__:
    main()