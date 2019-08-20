import numpy as np


class Lif:
    def __init__(self, cell_count, threshold=0.5, leak=0.9, adaption_rate=0.03, max_adaption=1.0):
        self.cell_count = cell_count

        self.threshold = threshold
        self.leak = leak
        self.adaption_rate = adaption_rate
        self.max_adaption = max_adaption

        self.adaption = np.zeros(self.cell_count, dtype='float32')
        self.membrane_potential = np.zeros(self.cell_count, dtype='float32')
        self.cell_output = np.zeros(self.cell_count, dtype='int32')

    def update(self, scalar_in):
        self.membrane_potential = self.membrane_potential + scalar_in - self.adaption*self.max_adaption
        self.adaption = self.adaption * (1 - self.adaption_rate) + scalar_in * self.adaption_rate
        fired = np.where(self.membrane_potential > self.threshold)
        self.membrane_potential[fired] = 0
        self.membrane_potential *= self.leak
        self.cell_output = np.zeros(self.cell_count, dtype='int32')
        self.cell_output[fired] = 1


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    single_cell = Lif((2, 2), threshold=0.6, leak=0.8, adaption_rate=0.05, max_adaption=0.9)
    test_current = np.concatenate((np.full(100, 0),
                                   np.full(150, 0.5),
                                   np.full(100, 0),
                                   np.full(100, -0.5),
                                   np.full(150, 0),
                                   # np.linspace(0.5, 0.4, 100),
                                   # np.full(100, 0),
                                   # np.linspace(0, 0.5, 100),
                                   # np.full(100, 0)
                                   )).astype('float32')
    test_duration = test_current.size

    activation_history = np.zeros(test_duration)
    mem_potential_history = np.zeros(test_duration)
    for i in range(test_duration):
        single_cell.update(test_current[i])
        activation_history[i] = single_cell.cell_output[0][0]
        mem_potential_history[i] = single_cell.membrane_potential[0][0]

    plt.plot(test_current)
    plt.plot(activation_history + mem_potential_history)
    plt.show()
