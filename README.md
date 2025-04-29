## 本PJ是用MAE训练的VIT做语义分割迁移学习的练手项目
使用方式如下

# 环境
python 3.8 

    pip install -r requirement.txt

# 数据集下载

    python download_data.py
等待数据下载  并且  将被下载到上一个目录的data拷贝到工程目录当中

# 下载MAE预训练模型
    !wget -nc https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_large_ganloss.pth

# 训练
    python segmtation_finetune.py
# 加载和跑模型
在文件“run_segmentation.py” 中修改变量 model_dir 换成被训练模型权重 

    python run_segmentation.py

在文件夹Result_img中可以看到语义分割推理结果
