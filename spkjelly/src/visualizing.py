import numpy as np
import matplotlib
from matplotlib import pyplot as plt

def plot_one_neuron_v_I_s(v: np.ndarray, I: np.ndarray, s: np.ndarray, v_threshold=1.0, v_reset=0.0,
                        title='$V[t]$ $I[t]$ and $S[t]$ of the neuron', figsize=(12, 8), dpi=200):
    '''
    :param v: shape=[T], 存放神经元不同时刻的电压
    :param s: shape=[T], 存放神经元不同时刻释放的脉冲
    :param v_threshold: 神经元的阈值电压
    :param v_reset: 神经元的重置电压。也可以为 ``None``
    :param title: 图的标题
    :param dpi: 绘图的dpi
    :return: 一个figure

    绘制单个神经元的电压、脉冲随着时间的变化情况。示例代码：

    .. code-block:: python

        import torch
        from spikingjelly.activation_based import neuron
        from spikingjelly import visualizing
        from matplotlib import pyplot as plt

        lif = neuron.LIFNode(tau=100.)
        x = torch.Tensor([2.0])
        T = 150
        s_list = []
        v_list = []
        for t in range(T):
            s_list.append(lif(x))
            v_list.append(lif.v)
        visualizing.plot_one_neuron_v_s(v_list, s_list, v_threshold=lif.v_threshold, v_reset=lif.v_reset,
                                        dpi=200)
        plt.show()

    .. image:: ./_static/API/visualizing/plot_one_neuron_v_s.*
        :width: 100%
    '''
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax0 = plt.subplot2grid((5, 1), (0, 0), rowspan=2)
    ax0.set_title(title)
    T = s.shape[0]
    t = np.arange(0, T)
    ax0.plot(t, I)
    ax0.set_xlim(-0.5, T - 0.5)
    ax0.set_ylabel('current')
    
    ax1 = plt.subplot2grid((5, 1), (2, 0), rowspan=2)
    ax1.plot(t, v)
    ax1.set_xlim(-0.5, T - 0.5)
    ax1.set_ylabel('voltage')
    ax1.axhline(v_threshold, label='$V_{threshold}$', linestyle='-.', c='r')
    if v_reset is not None:
        ax1.axhline(v_reset, label='$V_{reset}$', linestyle='-.', c='g')
    ax1.legend(frameon=True)
    
    t_spike = s * t
    mask = (s == 1)  # eventplot中的数值是时间发生的时刻，因此需要用mask筛选出
    ax2 = plt.subplot2grid((5, 1), (4, 0))
    ax2.eventplot(t_spike[mask], lineoffsets=0, colors='r')
    ax2.set_xlim(-0.5, T - 0.5)

    ax2.set_xlabel('simulating step')
    ax2.set_ylabel('spike')
    ax2.set_yticks([])

    ax2.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    return fig, ax0, ax1, ax2