from MAE_For_Segmtation_model import MAE_For_Segmentation
import segmentation_data_deal as Data_deal
import torch
import time
# import tqdm



MAE_pretrain_model = ""
unfreeze_layer_num = 5  # MAE中解封的block数

image_size = (224, 224)





num_classes = 21 #VOC2012里面分了20个类— + 1背景
loss_use_bg = True #False表示不用背景参与loss构建 True表示使用

data_dir = "data/VOCdevkit/VOC2012"
init_pth = "mae_visualize_vit_large_ganloss.pth"
lastest_test_loss = 0

from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_loss_file_name_train = f"train_loss_{timestamp}.txt"
log_loss_file_name_test = f"test_loss_{timestamp}.txt"
print(f"log loss file name:  {log_loss_file_name_train}     {log_loss_file_name_test}")
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
random_seed = 42



def load_pretrain_weight(model):
    ## todo加载模型参数
    sd_hf = torch.load(init_pth, weights_only=True)["model"]
    # sd = model.state_dict()
    msg = model.load_state_dict(sd_hf, strict=False)
    print(msg)


    freeze_block_list = [f"blocks.{i}" for i in range(19,24)]
    for name, param in model.named_parameters():
        if not "decoder" in name and not any(block_num in name for block_num in freeze_block_list):
            param.requires_grad = False
            continue
        param.requires_grad = True
        print(f"unfreeze!!: {name}")
            

import matplotlib.pyplot as plt
import os
plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
curve, = ax1.plot([], [], 'b-')  # 创建空线条
curve2, = ax2.plot([], [], 'r-')  # 创建空线条
ax1.set_xlabel('sample_or')
ax1.set_ylabel('loss')
ax1.set_title('Dynamic Loss Plot')
ax2.set_xlabel('sample_or')
ax2.set_ylabel('Test_loss')
ax2.set_title('Dynamic Loss Plot')
def plotting():
    sample_or = []
    loss = []
    sample_or_test = []
    loss_test = []

    with open(f"log/{log_loss_file_name_train}", 'r') as f:
        for line in f:
            if 'sample_or:' in line and 'loss:' in line:
                so = int(line.split('sample_or:')[1].split()[0])
                l = float(line.split('loss:')[1].strip())
                sample_or.append(so)
                loss.append(l)


    with open(f"log/{log_loss_file_name_test}", 'r') as f:
        for line in f:
            if 'sample_or:' in line and 'loss:' in line:
                so = int(line.split('sample_or:')[1].split()[0])
                l = float(line.split('loss:')[1].strip())
                sample_or_test.append(so)
                loss_test.append(l)

        curve.set_xdata(sample_or)
        curve.set_ydata(loss)
        ax1.relim()        # 重新计算坐标轴范围
        ax1.autoscale_view()  # 自动调整


        curve2.set_xdata(sample_or_test)
        curve2.set_ydata(loss_test)
        ax2.relim()        # 重新计算坐标轴范围
        ax2.autoscale_view()  # 自动调整


        fig.canvas.draw()
        fig.canvas.flush_events()  # 刷新显示



def test_validation(test_data, model):
    model.eval()
    test_data.test_current_pic_ord = 0
    loss_sum = 0
    loss_num = 0

    num_sample_count = 0
    with torch.no_grad():
        while True:
            X,Y,flag = test_data.test_next_batch()
            if not flag:
                break
            X, Y = X.to(device), Y.to(device)
            loss, _ = model(X, Y)
            mask = loss != 0
            loss_sum += loss.sum().item()
            loss_num += mask.sum().item()
            num_sample_count += X.shape[0]
        if ddp:
            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_num, op=dist.ReduceOp.SUM)

    if master_process:
        print(f"loss in test: {loss_sum/loss_num:.4f}   一共有 {num_sample_count} 个sample参与测试")
        with open(f"log/{log_loss_file_name_test}", "a") as f:
            f.write(f"loss:{loss_sum/loss_num:.4f}\n")
        global lastest_test_loss 
        lastest_test_loss = loss_sum/loss_num




from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)


def main():



    model = MAE_For_Segmentation(num_classes=num_classes, loss_use_bg=loss_use_bg)
    load_pretrain_weight(model)

    model.to(device)
    # model = torch.compile(model)

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model # always contains the "raw" unwrapped model
    epoch = 2
    batch_size = 16     # 一个batch（更新一次反向传播）需要累计的图片量
    step_size = 8       # 单个GPU的每轮要跑的图片数量
    grad_accumulate_steps = batch_size // (step_size*ddp_world_size)
    assert batch_size == step_size * ddp_world_size * grad_accumulate_steps, "请保证step_size和batch_size相匹配"


    Data_ = Data_deal.My_Segmentation_data(step_size=step_size, Augmentation_num=4, crop_size=(224,224))
    Data_.load_data(data_dir)



    batches_ = Data_.train_sample_sum // batch_size #Data_.train_sample_sum 是 图片总数（增广后）； batch_size是一个batch要处理的图片数
    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=6e-4)

    
    for e_ in range(epoch):
        for b_ in range(batches_):
            t_start = time.time()
            loss_accu = []
            loss_num = 0
            model.train()
            optimizer.zero_grad()
            for step_ in range(grad_accumulate_steps):
                X, Y = Data_.next_batch()
                X, Y = X.to(device), Y.to(device)
                loss, _ = model(X.float(), Y)
                mask = Y != 0
                if loss_use_bg:
                    mask.fill_(True)
                loss_accu.append(loss[mask].sum()) 
                loss_num += mask.sum().item()


            final_loss = sum(loss_accu) / loss_num
            loss_val = final_loss.item()
            final_loss.backward()
            optimizer.step()

            time_duration =  time.time()-t_start
            if master_process:
                print(f"loss: {loss_val:.4f}  |  time_cost/picture: {time_duration/batch_size:.4f}s  |  process: { (b_+1) * batch_size }/{batches_*batch_size}  |  timeleft: {((epoch- (e_+1))*batches_ + (batches_-(b_+1) ))*time_duration/60:.1f}min")
                with open(f"log/{log_loss_file_name_train}", "a") as f:
                    f.write(f"sample_or:{e_ * Data_.train_sample_sum + b_*batch_size}   loss:{loss_val:.4f}\n")

            if(b_ % 20 == 0):
                if master_process:
                    with open(f"log/{log_loss_file_name_test}", "a") as f:
                        f.write(f"sample_or:{e_ * Data_.train_sample_sum + b_*batch_size}   ")
                test_validation(Data_, model)
            if master_process:
                plotting()
        if master_process:
            torch.save(raw_model.state_dict(), f"model/model_weights_{timestamp}_epoch{e_+1}_valloss{lastest_test_loss:.4f}.pth")
    
    if ddp:
        destroy_process_group()
    


if '__main__' == __name__:
    main()
