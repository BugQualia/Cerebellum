from collections import deque

import numpy as np

# Hyperparameters:
EXC_W_MAX = 1
INH_W_MAX = -1

LTP_VAL = 0.05
LTD_VAL = 0.001


class Cell:
    def __init__(self, cell_count, excitatory_in_size, inhibitory_in_size, e_syn_count, i_syn_count):
        self.cell_count = cell_count
        self.cell_output = np.zeros(cell_count, 'int32')

        self.e_potential_syn = np.array([np.random.choice(excitatory_in_size, e_syn_count, False) for _ in range(cell_count)])
        self.i_potential_syn = np.array([np.random.choice(inhibitory_in_size, i_syn_count, False) for _ in range(cell_count)])

        self.excitatory_w = np.random.uniform(0, 1, (cell_count, e_syn_count))
        self.inhibitory_w = np.random.uniform(0, 1, (cell_count, i_syn_count))
        for i in range(cell_count):
            self.excitatory_w[i] /= np.sum(self.excitatory_w[i])
            self.inhibitory_w[i] /= np.sum(self.inhibitory_w[i])

        self.input_hist = deque([np.zeros((cell_count, 2), dtype='int32') for _ in range(5)], maxlen=5)
        self.cell_output_history = deque([np.zeros(cell_count, dtype='int32') for _ in range(5)], maxlen=5)

    def update(self, excitatory_in, inhibitory_in, learning=False):
        exc_scr = np.zeros(self.cell_count)
        inh_scr = np.zeros(self.cell_count)
        for i in range(self.cell_count):
            exc_scr[i] = np.dot(excitatory_in[self.e_potential_syn[i]], self.excitatory_w[i])
            inh_scr[i] = np.dot(inhibitory_in[self.i_potential_syn[i]], self.inhibitory_w[i])

        # update self.cell_output
        self.cell_output = np.clip(exc_scr - inh_scr, 0, None)


class PurkinjeCell:
    def __init__(self, cell_count, excitatory_in_size, inhibitory_in_size, e_syn_count, i_syn_count):
        self.cell_count = cell_count
        self.cell_output = np.zeros(cell_count, 'int32')

        self.e_potential_syn = np.array([np.random.choice(excitatory_in_size, e_syn_count, False) for _ in range(cell_count)])
        self.i_potential_syn = np.array([np.random.choice(inhibitory_in_size, i_syn_count, False) for _ in range(cell_count)])

        self.excitatory_w = np.random.uniform(0, 1, (cell_count, e_syn_count))
        self.inhibitory_w = np.random.uniform(0, 1, (cell_count, i_syn_count))
        for i in range(cell_count):
            self.excitatory_w[i] /= np.sum(self.excitatory_w[i])
            self.inhibitory_w[i] /= np.sum(self.inhibitory_w[i])

        self.input_hist = deque([np.zeros((cell_count, 2), dtype='int32') for _ in range(5)], maxlen=5)
        self.cell_output_history = deque([np.zeros(cell_count, dtype='int32') for _ in range(5)], maxlen=5)

    def update(self, cf_in, excitatory_in, inhibitory_in, learning=True):
        exc_scr = np.zeros(self.cell_count)
        inh_scr = np.zeros(self.cell_count)
        for i in range(self.cell_count):
            exc_scr[i] = np.dot(excitatory_in[self.e_potential_syn[i]], self.excitatory_w[i])
            inh_scr[i] = np.dot(inhibitory_in[self.i_potential_syn[i]], self.inhibitory_w[i])

        # update self.cell_output
        self.cell_output = cf_in + np.clip(exc_scr - inh_scr, 0, None)

        # update history
        self.input_hist.rotate(1)
        self.cell_output_history.rotate(1)
        self.input_hist[0] = np.asarray([excitatory_in, inhibitory_in])
        self.cell_output_history[0] = self.cell_output

        if not learning: return
        for i in range(self.cell_count):
            if cf_in[i] != 0:
                tmp = excitatory_in[self.e_potential_syn[i]]
                print(np.sum(tmp > 1))
                self.excitatory_w[i] -= (tmp > 1) * tmp * 0.01
                self.excitatory_w[i] = np.clip(self.excitatory_w[i], 0, None)
                # print(np.sum(self.excitatory_w[i]))
                # self.excitatory_w[i] /= np.sum(self.excitatory_w[i])


class ION:
    def __init__(self, cell_count, in_size, freq):
        self.cell_count = cell_count
        self.freq = freq
        self.v = np.random.randint(0, 2, cell_count)*0.8-0.4
        self.cell_output = np.zeros(cell_count)
        self.dv = np.zeros(cell_count)
        self.map = np.random.uniform(0, 0.003, (cell_count, in_size))
        self.map_chem = np.random.uniform(0, 0.4, (cell_count, in_size))

    def update(self, elec_in, chem_in):
        curr = np.dot(self.map, elec_in) + chem_in
        self.dv = self.dv - self.v * self.freq
        self.v = np.tanh(self.v + self.dv + curr)
        self.cell_output = np.zeros(self.cell_count)
        self.cell_output[np.where(self.v > 0.5)] = 1
        self.v[np.where(self.v > 0.5)] = -0.3
        return self.v + self.cell_output*3


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    hist = np.zeros((300, 12))
    cell_group_1 = Cell(4, 4, 0, 2, 0)
    cell_group_2 = Cell(4, 8, 0, 2, 0)
    cell_group_3 = Cell(4, 8, 0, 2, 0)

    cg1_in = np.concatenate([np.full((100, 4), 0), np.full((100, 4), 1), np.full((100, 4), 0)], axis=0)
    cg2_in = np.zeros(8)
    cg3_in = np.zeros(8)

    for ii in range(300):
        print(ii)

        cg2_in[0:4] = cell_group_1.cell_output
        cg2_in[4:8] = cell_group_3.cell_output
        cg3_in[0:4] = cell_group_1.cell_output
        cg3_in[4:8] = cell_group_2.cell_output
        cell_group_1.update(cg1_in[ii], np.empty(0), learning=True)
        cell_group_2.update(cg2_in, np.empty(0), learning=True)
        cell_group_3.update(cg3_in, np.empty(0), learning=True)
        hist[ii][:4] = cg1_in[ii]
        hist[ii][4:8] = cell_group_1.cell_output
        hist[ii][8:] = cell_group_2.cell_output
    plt.plot(hist)
    plt.show()
    ####################################################################################################################
    hist = np.zeros((2000, 10))
    ion = ION(20, 20, np.random.uniform(0.01, 0.03, 20))
    tmp = np.zeros(20)
    chem = np.zeros(20)
    for ii in range(2000):
        if ii == 1200:
            chem[5] = 1
        if ii == 1500:
            chem[6] = 1
        hist[ii] = ion.update(tmp, chem)[0:10]
        tmp = ion.v + ion.cell_output
        chem = np.zeros(20)
    plt.subplot(1, 2, 1)
    plt.plot(hist)
    plt.subplot(1, 2, 2)
    plt.plot(np.sum(hist, axis=1))
    plt.show()
