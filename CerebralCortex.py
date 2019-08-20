import numpy as np

from body import _frequently_used as fr

# Hyperparameters:
MEM_MAX = 2100000000
MEM_MIN = -2100000000
L4_PROX_TOP_MUL = 1.9
LEARNED_CELL_RATIO = 0.8
MAX_SPARSITY = 0.024
MIN_SPARSITY = 0.004
NOT_LEARN_SPARSITY = 0.006
REMAP_RATIO = 0.001
HISTORY_DEC = 0.7
BURST_SCR_MUL = 10
BURST_LRN_MUL = 5

APICAL_ACTIVATE_THRESHOLD = 0.1
L5_PROX_TOP_MUL = 4
L5_BURST_DEPRESS_MUL = 3


class Synapse:
    def __init__(self, cell_count, rf_size, max_syn_count):
        self.cell_count = cell_count
        self.rf_size = rf_size
        self.max_syn_count = max_syn_count
        self.map = np.full((self.cell_count, self.max_syn_count), -1, dtype='int32')
        self.memory = np.zeros((self.cell_count, self.rf_size), dtype='int32')
        self.pref = np.array([np.random.permutation(self.rf_size) for _ in range(self.cell_count)])
        self.remap(np.arange(self.cell_count, dtype='int32'))

    def change_max_syn_count(self, new_max_syn_count):
        self.max_syn_count = new_max_syn_count
        self.map = np.full((self.cell_count, self.max_syn_count), -1, dtype='int32')

    def remap(self, lc):
        for i in lc:
            mid = np.partition(self.memory[i], -self.max_syn_count)[-self.max_syn_count]
            poss = np.where(self.memory[i] > mid)[0]
            maybe = np.where(self.memory[i] == mid)[0]
            if self.max_syn_count < poss.size + maybe.size:
                maybe = self.pref[i][np.where(fr.nb_isin(self.pref[i], maybe))[0][:self.max_syn_count - poss.size]]
            self.map[i][:maybe.size] = maybe
            self.map[i][maybe.size:maybe.size+poss.size] = poss
            self.map[i][maybe.size+poss.size:] = -1

    def memory_normalization(self):
        # TODO
        pass

    def memory_noise(self, cell):
        self.memory[cell] += np.random.normal(0, 1, (cell.size, self.rf_size)).astype('int32')

    def store_synapse(self, save_loc='default'):
        np.savez(save_loc,
                 cell_count=self.cell_count,
                 in_size=self.rf_size,
                 max_syn_count=self.max_syn_count,
                 map=self.map,
                 memory=self.memory,
                 pref=self.pref)

    def load_synapse(self, save_loc='default'):
        npz_file = np.load(save_loc + '.npz')
        self.cell_count = npz_file['cell_count']
        self.rf_size = npz_file['in_size']
        self.max_syn_count = npz_file['max_syn_count']
        self.map = npz_file['map']
        self.memory = npz_file['memory']
        self.pref = npz_file['pref']


class EntropyReducer:
    """
    Cortical layer 4 pyramidal cells
    """
    def __init__(self, cell_count, prox_in_size, dist_in_size, max_prox_count, max_dist_count):
        self.cell_count = cell_count
        self.cell_output = np.zeros(self.cell_count, 'int32')
        self.cell_history = np.zeros(self.cell_count, 'float64')
        self.prox = Synapse(self.cell_count, prox_in_size, max_prox_count)
        self.dist = Synapse(self.cell_count, dist_in_size, max_dist_count)

    def update(self, prox_in, dist_in, learning=True):
        prox_scr = fr.sum_isin(self.prox.map, np.where(prox_in >= 1)[0]) \
                   + fr.sum_isin(self.prox.map, np.where(prox_in == 2)[0]) * BURST_SCR_MUL

        dist_scr = fr.sum_isin(self.dist.map, np.where(dist_in >= 1)[0]) \
                   + fr.sum_isin(self.dist.map, np.where(dist_in == 2)[0]) * BURST_SCR_MUL

        # update self.cell_output
        active_cell_count = int(np.clip(a=self.cell_count*np.sum(prox_in)/self.prox.rf_size,
                                        a_min=self.cell_count * MIN_SPARSITY,
                                        a_max=self.cell_count * MAX_SPARSITY))

        p_top = fr.select_top(prox_scr, int(L4_PROX_TOP_MUL * active_cell_count))
        d_top = fr.select_top(dist_scr[p_top], active_cell_count)
        active_ind = p_top[d_top]
        self.cell_output = np.zeros(self.cell_count, 'int32')
        self.cell_output[active_ind] = 1
        if active_cell_count <= int(self.cell_count * NOT_LEARN_SPARSITY) or not learning: return

        # update Synapse.memory
        learned_cell_count = int(active_cell_count * LEARNED_CELL_RATIO)
        # TODO: Replace with threshold
        # TODO: Implement metabotropic receptor
        self.cell_history = self.cell_history * HISTORY_DEC + self.cell_output * 100
        ltp = active_ind[fr.select_top(self.cell_history[active_ind], learned_cell_count)]
        self.prox.memory[ltp] = fr.nb_clip((self.prox.memory[ltp] + prox_in + (prox_in == 2) * BURST_LRN_MUL), MEM_MIN, MEM_MAX)
        self.dist.memory[ltp] = fr.nb_clip((self.dist.memory[ltp] + dist_in + (dist_in == 2) * BURST_LRN_MUL), MEM_MIN, MEM_MAX)

        ltd_val = np.partition(self.cell_history, learned_cell_count)[learned_cell_count]
        ltd = np.random.choice(np.where(self.cell_history <= ltd_val)[0], learned_cell_count, False)
        self.prox.memory[ltd] = fr.nb_clip((self.prox.memory[ltd] - (prox_in == 2) * BURST_LRN_MUL), MEM_MIN, MEM_MAX)
        self.dist.memory[ltd] = fr.nb_clip((self.dist.memory[ltd] - (dist_in == 2) * BURST_LRN_MUL), MEM_MIN, MEM_MAX)

        # TODO: Reduce learning rate
        self.prox.remap(np.random.choice(self.cell_count, int(self.cell_count * REMAP_RATIO), False))
        self.dist.remap(np.random.choice(self.cell_count, int(self.cell_count * REMAP_RATIO), False))

    def clear(self):
        self.cell_output = np.zeros(self.cell_count, 'int32')
        self.cell_history = np.zeros(self.cell_count, 'float64')

    def store_model(self, save_loc='default'):
        self.prox.store_synapse(save_loc+'_prox')
        self.dist.store_synapse(save_loc+'_dist')
        np.savez(save_loc,
                 cell_count=self.cell_count,
                 cell_output=self.cell_output,
                 cell_history=self.cell_history)

    def load_model(self, save_loc='default'):
        npz_file = np.load(save_loc+'.npz')
        self.cell_count = npz_file['cell_count']
        self.cell_output = npz_file['cell_output']
        self.cell_history = npz_file['cell_history']
        self.prox.load_synapse(save_loc+'_prox')
        self.dist.load_synapse(save_loc+'_dist')


class ER2D:
    """
    Cortical layer 4 pyramidal cells
    """
    def __init__(self, column_count, cpc,
                 prox_rf_size, dist_rf_size,
                 max_prox_count, max_dist_count):
        self.cell_count = column_count * cpc
        self.cpc = cpc
        self.cell_output = np.zeros(self.cell_count, 'int32')
        self.cell_history = np.zeros(self.cell_count, 'float64')
        self.prox = Synapse(self.cell_count, prox_rf_size, max_prox_count)
        self.dist = Synapse(self.cell_count, dist_rf_size, max_dist_count)

    def update(self, prox_in, dist_in, learning=True):
        prox_scr = fr.sum_isin2d_col(self.prox.map, prox_in, 1, self.cpc) \
                   + fr.sum_isin2d_col(self.prox.map, prox_in, 2, self.cpc) * BURST_SCR_MUL
        dist_scr = fr.sum_isin2d_col(self.dist.map, dist_in, 1, self.cpc) \
                   + fr.sum_isin2d_col(self.dist.map, dist_in, 2, self.cpc) * BURST_SCR_MUL

        # update self.cell_output
        # TODO: Column inhibition
        active_cell_count = int(np.clip(a=self.cpc * np.sum(prox_in)/self.prox.rf_size,
                                        a_min=self.cell_count * MIN_SPARSITY,
                                        a_max=self.cell_count * MAX_SPARSITY))

        p_top = fr.select_top(prox_scr, int(L4_PROX_TOP_MUL * active_cell_count))
        d_top = fr.select_top(dist_scr[p_top], active_cell_count)
        active_ind = p_top[d_top]
        self.cell_output = np.zeros(self.cell_count, 'int32')
        self.cell_output[active_ind] = 1
        if active_cell_count <= int(self.cell_count * NOT_LEARN_SPARSITY) or not learning: return

        # update Synapse.memory
        learned_cell_count = int(active_cell_count * LEARNED_CELL_RATIO)
        # TODO: Replace with threshold
        # TODO: Implement metabotropic receptor
        self.cell_history = self.cell_history * HISTORY_DEC + self.cell_output * 100
        ltp = active_ind[fr.select_top(self.cell_history[active_ind], learned_cell_count)]
        self.prox.memory[ltp] = fr.nb_clip((self.prox.memory[ltp]
                                            + prox_in[ltp//self.cpc]
                                            + (prox_in[ltp//self.cpc] == 2) * BURST_LRN_MUL), MEM_MIN, MEM_MAX)
        self.dist.memory[ltp] = fr.nb_clip((self.dist.memory[ltp]
                                            + dist_in[ltp//self.cpc]
                                            + (dist_in[ltp//self.cpc] == 2) * BURST_LRN_MUL), MEM_MIN, MEM_MAX)

        ltd_val = np.partition(self.cell_history, learned_cell_count)[learned_cell_count]
        ltd = np.random.choice(np.where(self.cell_history <= ltd_val)[0], learned_cell_count, False)
        self.prox.memory[ltd] = fr.nb_clip((self.prox.memory[ltd]
                                            - (prox_in[ltp//self.cpc] == 2) * BURST_LRN_MUL), MEM_MIN, MEM_MAX)
        self.dist.memory[ltd] = fr.nb_clip((self.dist.memory[ltd]
                                            - (dist_in[ltp//self.cpc] == 2) * BURST_LRN_MUL), MEM_MIN, MEM_MAX)

        # TODO: Reduce learning rate
        self.prox.remap(np.random.choice(self.cell_count, int(self.cell_count * REMAP_RATIO), False))
        self.dist.remap(np.random.choice(self.cell_count, int(self.cell_count * REMAP_RATIO), False))

    def clear(self):
        self.cell_output = np.zeros(self.cell_count, 'int32')
        self.cell_history = np.zeros(self.cell_count, 'float64')

    def store_model(self, save_loc='default'):
        self.prox.store_synapse(save_loc+'_prox')
        self.dist.store_synapse(save_loc+'_dist')
        np.savez(save_loc,
                 cell_count=self.cell_count,
                 cpc=self.cpc,
                 cell_output=self.cell_output,
                 cell_history=self.cell_history)

    def load_model(self, save_loc='default'):
        npz_file = np.load(save_loc+'.npz')
        self.cell_count = npz_file['cell_count']
        self.cpc = npz_file['cpc']
        self.cell_output = npz_file['cell_output']
        self.cell_history = npz_file['cell_history']
        self.prox.load_synapse(save_loc+'_prox')
        self.dist.load_synapse(save_loc+'_dist')


class Predictor:
    """
    Cortical layer 2/3, 5, layer 6 cortico-thalamic pyramidal cells
    """
    def __init__(self, cell_count,
                 prox_in_size, dist_in_size, apic_in_size,
                 max_prox_count, max_dist_count, max_apic_count):
        self.cell_count = cell_count
        self.cell_output = np.zeros(self.cell_count, 'int32')
        self.cell_history = np.zeros(self.cell_count, 'float64')
        self.prox = Synapse(self.cell_count, prox_in_size, max_prox_count)
        self.dist = Synapse(self.cell_count, dist_in_size, max_dist_count)
        self.apic = Synapse(self.cell_count, apic_in_size, max_apic_count)

    def update(self, prox_in, dist_in, apic_in, learning=True):
        prox_scr = fr.sum_isin(self.prox.map, np.where(prox_in >= 1)[0]) \
                   + fr.sum_isin(self.prox.map, np.where(prox_in == 2)[0]) * BURST_SCR_MUL

        dist_scr = fr.sum_isin(self.dist.map, np.where(dist_in >= 1)[0]) \
                   + fr.sum_isin(self.dist.map, np.where(dist_in == 2)[0]) * BURST_SCR_MUL

        apic_scr = fr.sum_isin(self.apic.map, np.where(apic_in >= 1)[0]) \
                   + fr.sum_isin(self.apic.map, np.where(apic_in == 2)[0]) * BURST_SCR_MUL

        # update self.cell_output
        active_cell_count = int(np.clip(a=self.cell_count*np.sum(prox_in)/self.prox.rf_size,
                                        a_min=self.cell_count * MIN_SPARSITY,
                                        a_max=self.cell_count * MAX_SPARSITY))

        p_top = fr.select_top(prox_scr, int(L5_PROX_TOP_MUL * active_cell_count))
        d_top = fr.select_top(dist_scr[p_top], active_cell_count)
        a_top = apic_scr > self.apic.max_syn_count*APICAL_ACTIVATE_THRESHOLD
        active_ind = p_top[d_top]
        burst_ind = np.intersect1d(active_ind, a_top)

        self.cell_output = np.zeros(self.cell_count, 'int32')
        self.cell_output[active_ind] = 1
        self.cell_output[burst_ind] = 2
        if active_cell_count <= int(self.cell_count * NOT_LEARN_SPARSITY) or not learning: return

        # update Synapse.memory
        learned_cell_count = int(active_cell_count * LEARNED_CELL_RATIO)
        # TODO: Implement metabotropic receptor
        self.cell_history = self.cell_history * HISTORY_DEC + self.cell_output * 100

        ltp = fr.select_top(self.cell_history[active_ind], learned_cell_count)
        self.prox.memory[ltp] = fr.nb_clip((self.prox.memory[ltp] + prox_in + (prox_in == 2) * BURST_LRN_MUL), MEM_MIN, MEM_MAX)
        self.dist.memory[ltp] = fr.nb_clip((self.dist.memory[ltp] + dist_in + (dist_in == 2) * BURST_LRN_MUL), MEM_MIN, MEM_MAX)
        self.apic.memory[ltp] = fr.nb_clip((self.apic.memory[ltp] + apic_in + (apic_in == 2) * BURST_LRN_MUL), MEM_MIN, MEM_MAX)

        not_fired = fr.select_bottom(dist_scr, int(L5_BURST_DEPRESS_MUL * active_cell_count))
        ltd = np.intersect1d(not_fired, a_top)
        self.prox.memory[ltd] = fr.nb_clip((self.prox.memory[ltd] - (prox_in == 2) * BURST_LRN_MUL), MEM_MIN, MEM_MAX)
        self.dist.memory[ltd] = fr.nb_clip((self.dist.memory[ltd] - (dist_in == 2) * BURST_LRN_MUL), MEM_MIN, MEM_MAX)
        self.apic.memory[ltd] = fr.nb_clip((self.apic.memory[ltd] - apic_in * BURST_LRN_MUL), MEM_MIN, MEM_MAX)

        self.prox.remap(np.random.choice(self.cell_count, int(self.cell_count * REMAP_RATIO), False))
        self.dist.remap(np.random.choice(self.cell_count, int(self.cell_count * REMAP_RATIO), False))
        self.apic.remap(np.random.choice(self.cell_count, int(self.cell_count * REMAP_RATIO), False))

    def clear(self):
        self.cell_output = np.zeros(self.cell_count, 'int32')
        self.cell_history = np.zeros(self.cell_count, 'float64')

    def store_model(self, save_loc='default'):
        self.prox.store_synapse(save_loc+'_prox')
        self.dist.store_synapse(save_loc+'_dist')
        self.apic.store_synapse(save_loc+'_dist')
        np.savez(save_loc,
                 cell_count=self.cell_count,
                 cell_output=self.cell_output,
                 cell_history=self.cell_history)

    def load_model(self, save_loc='default'):
        npz_file = np.load(save_loc+'.npz')
        self.cell_count = npz_file['cell_count']
        self.cell_output = npz_file['cell_output']
        self.cell_history = npz_file['cell_history']
        self.prox.load_synapse(save_loc+'_prox')
        self.dist.load_synapse(save_loc+'_dist')
        self.apic.load_synapse(save_loc+'_apic')


class L6CT:
    """
    Cortical layer 6 cortico-thalamic neuron
    """
    def __init__(self):
        pass


class Hippocampus:
    """
    Hippocampus
    """
    def __init__(self):
        pass
