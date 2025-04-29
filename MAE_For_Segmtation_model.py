# import sys 
# sys.path.append("../") 
from models_mae import MaskedAutoencoderViT
import torch.nn as nn
import torch

class MAE_For_Segmentation(MaskedAutoencoderViT):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, num_classes=1,use_decoder=1,loss_use_bg=False): #use_decoder 1是用CNN的模型 2是用transformer的模型
        super().__init__(img_size, patch_size, in_chans,
                 embed_dim, depth, num_heads,
                 decoder_embed_dim, decoder_depth, decoder_num_heads,
                 mlp_ratio, norm_layer, norm_pix_loss)
        
        self.use_decoder = use_decoder
        self.num_classes = num_classes
        self.patch_num_sqrt = img_size/patch_size
        self.embed_dim = embed_dim

        if not loss_use_bg:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='none')# 背景损失就别搞了
        else:
            self.criterion = torch.nn.CrossEntropyLoss(reduction='none')

        if use_decoder == 1:
            self.init_CNN_decoder()
        elif use_decoder == 2:
            self.transformer_decoder()
        
        self.delete_unnecessary_layer()

    def delete_unnecessary_layer(self):
        del self.decoder_embed
        del self.decoder_pos_embed
        del self.decoder_blocks
        del self.decoder_norm
        del self.decoder_pred

        
    def init_CNN_decoder(self):
        self.decoder_reduce = nn.Conv2d(self.embed_dim, 256, kernel_size=1)
        
        # 渐进上采样
        self.decoder_up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.decoder_up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        assert 64 >= self.num_classes , " 类别数量太多，请调整网络结构 "
        self.decoder_up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, max(32, self.num_classes), kernel_size=3, padding=1),
            nn.ReLU()
        )
        # 最终输出
        self.decoder_final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(max(32, self.num_classes), self.num_classes, kernel_size=1)
        )
    def transformer_decoder():
        pass

    def forwad_encoder(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1) #expand在这里是复制  -1是表示该维度不变
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
            
        x = self.norm(x) #特征提取完毕
        return x

    def forward_decoder_CNN(self, x):
        # 输入x: [B, 196, 1024]
        x = x[:, 1:, :]
        B = x.shape[0]
        x = x.view(B, 14, 14, 1024).permute(0, 3, 1, 2)  # [B, 1024, 14, 14]
        x = self.decoder_reduce(x)  # [B, 256, 14, 14]
        x = self.decoder_up1(x)     # [B, 128, 28, 28]
        
        x = self.decoder_up2(x)     # [B, 64, 56, 56]
        x = self.decoder_up3(x)     # [B, max(32, num_classes), 112, 112]
        # x =  # [B, num_classes, 224, 224]
        # prob = torch.softmax(x, dim=1)  # 转换为概率（如果output是logits）
        # pred_mask = torch.argmax(prob, dim=1)  # [B, 224, 224]
        return self.decoder_final(x)
        # return x
    def forward_decoder_Transformer(self, x):
        # pass
        return x
    

    def forward_loss(self, pred, target):
        return self.criterion(pred, target)
        #  loss
        # pass
        # mask = #用于描述


    def forward(self, x, y=None):
        x = self.forwad_encoder(x)
        if self.use_decoder == 1:
            x = self.forward_decoder_CNN(x)
        elif self.use_decoder == 2:
            x = self.forward_decoder_Transformer(x)

        if y == None: # 为预测
            return None, x
        
        loss = self.forward_loss(x, y)  # 为训练
        return loss, None



