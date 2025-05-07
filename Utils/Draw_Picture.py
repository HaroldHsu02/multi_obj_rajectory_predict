import matplotlib
import seaborn as sns
# import matplotlib.pyplot as plt
from matplotlib import pyplot as plt, font_manager
from matplotlib import font_manager

# matplotlib.use('TkAgg')


def plot_rewards(rewards, ma_rewards, plot_cfg, tag='train'):
    sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title("learning curve on {} of {} for {}".format(
        plot_cfg.device, plot_cfg.algo_name, plot_cfg.env_name))
    plt.xlabel('episodes')
    plt.plot(rewards, label='rewards')
    plt.plot(ma_rewards, label='ma_rewards')
    plt.legend()
    if plot_cfg.save:
        plt.savefig(plot_cfg.result_path+"{}_rewards_curve".format(tag))
    plt.show(block=True)


def plot_location(UE_location, MEC_location):
    """画出用户以及服务器的位置散点图"""
    my_font = font_manager.FontProperties(fname="C:\Windows\Fonts\MSYHL.TTC")
    UE_location_x = []
    UE_location_y = []
    MEC_location_x = []
    MEC_location_y = []
    for i in range(len(UE_location)):
        UE_location_x.append(UE_location[i, 0])
        UE_location_y.append(UE_location[i, 1])
    for i in range(len(MEC_location)):
        MEC_location_x.append(MEC_location[i, 0])
        MEC_location_y.append(MEC_location[i, 1])
    # 设置图形大小
    plt.figure(figsize=(20, 20), dpi=80)
    # 使用scatter绘制散点图,和之前绘制折线图一样只用将plot更改成scatter
    plt.scatter(UE_location_x, UE_location_y, label='用户位置')
    plt.scatter(MEC_location_x, MEC_location_y, label='边缘服务器位置')
    # 添加描述信息
    plt.xlabel('场地长度', fontproperties=my_font)
    plt.ylabel('场地宽度', fontproperties=my_font)
    plt.title('位置散点图', fontproperties=my_font)
    # 添加图例
    plt.legend(prop=my_font, loc='upper left')  # 要在绘制图像那一步添加标签
