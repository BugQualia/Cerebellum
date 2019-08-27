from aaaa import Cerebellum as cb
import numpy as np

unknown = 100

Glomeruli = np.array([])

GrC = cb.Cell(cell_count=88158, threshold=1,
              excitatory_in_size=4, inhibitory_in_size=4)

GoC = cb.Cell(cell_count=219, threshold=1,
              excitatory_in_size=9000, inhibitory_in_size=0)  # TODO: lugaro cell

PC = cb.Cell(cell_count=69, threshold=1,
             excitatory_in_size=88158, inhibitory_in_size=603)

SC = cb.Cell(cell_count=603, threshold=1,
             excitatory_in_size=1000, inhibitory_in_size=0)
BC = cb.Cell(cell_count=603, threshold=1,
             excitatory_in_size=1, inhibitory_in_size=1)
DCN = cb.Cell(cell_count=603, threshold=1,
              excitatory_in_size=1, inhibitory_in_size=1)

ION = cb.ION(cell_count=69, in_size=1, freq=np.random.uniform(0.02, 0.03, 69))


while 1:
    PC.update(GrC.cell_output, BC.cell_output)





