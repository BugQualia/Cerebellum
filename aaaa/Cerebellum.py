from collections import deque

import numba as nb
import numpy as np

from aaaa import frequently_used as fr

# Hyperparameters:
EXC_W_MAX = 1
INH_W_MAX = -1

LTP_VAL = 0.05
LTD_VAL = 0.001


class Synapse:
    def __init__(self, cell_count, rf_size, max_syn_count):
        self.cell_count = cell_count
        self.rf_size = rf_size
        self.max_syn_count = max_syn_count
        self.w = np.random.normal(0, 1, (self.cell_count, self.rf_size))


class Cell:
    def __init__(self, cell_count, threshold,
                 excitatory_in_size, inhibitory_in_size,
                 max_excitatory_count, max_inhibitory_count):
        self.cell_count = cell_count
        self.threshold = threshold
        self.cell_output = np.zeros(cell_count, 'int32')

        self.excitatory = Synapse(cell_count, excitatory_in_size, max_excitatory_count)
        self.inhibitory = Synapse(cell_count, inhibitory_in_size, max_inhibitory_count)

        self.input_hist = deque([np.zeros((cell_count, 2), dtype='int32') for _ in range(5)], maxlen=5)
        self.cell_output_history = deque([np.zeros(cell_count, dtype='int32') for _ in range(5)], maxlen=5)

    def update(self, excitatory_in, inhibitory_in):
        # Linear sum
        exc_scr = np.dot(excitatory_in, self.excitatory.w)
        inh_scr = np.dot(inhibitory_in, self.inhibitory.w)

        # update self.cell_output
        active_ind = np.where((exc_scr - inh_scr) > self.threshold)[0]
        self.cell_output = np.zeros(self.cell_count, 'int32')
        self.cell_output[active_ind] = 1

        # update history
        self.input_hist.rotate(1)
        self.cell_output_history.rotate(1)
        self.input_hist[0] = np.asarray([excitatory_in, inhibitory_in])
        self.cell_output_history[0] = self.cell_output

        # STDP
        ltp = self.cell_output_history[2]
        self.excitatory.w[ltp] = fr.nb_clip((self.excitatory.w[ltp] + self.input_hist[2][0] * LTP_VAL), 0, EXC_W_MAX)
        self.inhibitory.w[ltp] = fr.nb_clip((self.inhibitory.w[ltp] + self.input_hist[2][1] * LTP_VAL), 0, INH_W_MAX)

        ltd = self.cell_output_history[2]
        self.excitatory.w[ltd] = fr.nb_clip((self.excitatory.w[ltd] - self.input_hist[3][0] * LTD_VAL), 0, EXC_W_MAX)
        self.inhibitory.w[ltd] = fr.nb_clip((self.inhibitory.w[ltd] - self.input_hist[3][1] * LTD_VAL), 0, INH_W_MAX)


class ION:
    def __init__(self, cell_count, in_size, freq):
        self.cell_count = cell_count
        self.freq = freq
        self.v = np.random.randint(0, 2, cell_count)*2-1
        self.cell_output = np.zeros(cell_count)
        self.dv = np.zeros(cell_count)
        self.map = np.random.uniform(0, 0.003, (cell_count, in_size))

    def update(self, elec_in):
        curr = np.dot(self.map, elec_in)
        self.dv = self.dv - self.v * self.freq
        self.v = np.arctan(self.v + self.dv + curr)
        self.cell_output = np.zeros(self.cell_count)
        self.cell_output[np.where(self.v > 0.5)] = 1
        self.v[np.where(self.v > 0.5)] = -0.5
        return self.v + self.cell_output


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    hist = np.zeros((1000, 10))
    ion = ION(20, 20, np.concatenate([np.full(10, 0.02), np.full(10, 0.026)]))
    tmp = np.zeros(20)
    for i in range(1000):
        hist[i] = ion.update(tmp)[0:10]
        tmp = ion.v
        if i == 200:
            tmp[5] = 1
        if i == 300:
            tmp[6] = 1
    plt.subplot(1, 2, 1)
    plt.plot(hist)
    plt.subplot(1, 2, 2)
    plt.plot(np.sum(hist, axis=1))
    plt.show()

    # aa = (4 + 1j)/np.sqrt(17)
    # bb = (4 - 1j)/np.sqrt(17)
    # y = []
    # for i in range(200):
    #     y.append(aa**i + bb**i)
    # print(np.real(y))
    # plt.plot(np.real(y))
    # plt.show()

    # y1 = [0.23162855960858492, -0.07944159167794967, -0.2546310706197991, -0.05573934925323282, 0.2441876099420363]
    # y2 = [-0.25563021183483203, -0.03603669780124944, 0.25331225900236054, 0.21104476680487766, -0.10959098648913836]
    #
    # for i in range(10000):
    #     i1_in = np.dot(np.array(y1[-5:]), np.array([0.7, 0.2, 0, 0, 0]))
    #     i2_in = np.dot(np.array(y1[-5:]), np.array([0, 0, 0, 0.2, 0.7]))
    #     dy = i1_in - i2_in
    #     y1.append(np.arctan(y1[-1] + dy))
    # print(y1)
    # plt.subplot(1, 2, 1)
    # plt.plot(y1)
    # plt.subplot(1, 2, 2)
    # plt.plot(np.real(np.fft.fft(y1)))
    # plt.show()
