import matplotlib.pyplot as plt
import os
log_loss_file_name_train = "train_loss_20250423_215936.txt"
log_loss_file_name_test = "test_loss_20250423_215936.txt"

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
while os.path.exists(f"log/{log_loss_file_name_train}") and os.path.exists(f"log/{log_loss_file_name_test}"):
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
    plt.pause(0.1)
plt.ioff()
plt.show()

