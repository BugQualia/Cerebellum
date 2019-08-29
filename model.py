from aaaa import Cerebellum as cb
import numpy as np
import matplotlib.pyplot as plt


hist = np.zeros((300, 16))
cell_group_1 = cb.Cell(4, 4, 4, 2, 2)
cell_group_2 = cb.Cell(4, 8, 4, 2, 2)
cell_group_3 = cb.Cell(4, 8, 4, 2, 2)
cell_group_4 = cb.Cell(4, 8, 4, 2, 2)

# cg1_in = np.concatenate([np.full((100, 4), 0), np.full((100, 4), 1), np.full((100, 4), 0)], axis=0)
cg1_in = np.full((300, 4), 0)
cg1_in[100:200, 0:2] = 1
cg1_in[50:150, 2:4] = 1
print(np.sum(cg1_in))
cg2_e_in = np.zeros(8)
cg3_e_in = np.zeros(8)
cg3_i_in = np.zeros(4)
cg4_e_in = np.zeros(8)

for ii in range(300):
    cg2_e_in[0:4] = cell_group_1.cell_output
    cg2_e_in[4:8] = cell_group_3.cell_output
    cg3_e_in[0:4] = cell_group_1.cell_output
    cg3_e_in[4:8] = cell_group_2.cell_output
    cg3_i_in = cell_group_2.cell_output*0.5
    cg4_e_in[0:4] = cell_group_2.cell_output
    cg4_e_in[4:8] = cell_group_4.cell_output
    print(cg1_in[ii])
    cell_group_1.update(cg1_in[ii], np.zeros(4), learning=True)
    cell_group_2.update(cg2_e_in, np.zeros(4), learning=True)
    cell_group_3.update(cg3_e_in, cg3_i_in, learning=True)
    cell_group_4.update(cg4_e_in, np.zeros(4), learning=True)

    hist[ii][:4] = cell_group_1.cell_output
    hist[ii][4:8] = cell_group_2.cell_output
    hist[ii][8:12] = cell_group_3.cell_output
    hist[ii][12:] = cell_group_4.cell_output


plt.subplot(1, 4, 1)
plt.plot(hist[:, :4])
plt.subplot(1, 4, 2)
plt.plot(hist[:, 4:8])
plt.subplot(1, 4, 3)
plt.plot(hist[:, 8:12])
plt.subplot(1, 4, 4)
plt.plot(hist[:, 12:])
plt.show()





