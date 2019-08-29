from aaaa import Cerebellum as cb
import numpy as np
import matplotlib.pyplot as plt

mf_in_size = 30000
grc_count = 100000


mf_in = np.zeros((mf_in_size, 600), dtype='float32')
mf_in[0:10000, 100:200] = np.sin(np.linspace(0, np.pi, 100))*2
mf_in[10000:20000, 50:150] = np.sin(np.linspace(0, np.pi, 100))*2
mf_in[20000:30000, 150:250] = np.sin(np.linspace(0, np.pi, 100))*2
mf_in[0:7000, 300:400] = np.sin(np.linspace(0, np.pi, 100))*2
mf_in[15000:25000, 375:425] = np.sin(np.linspace(0, np.pi, 50))*2
mf_in[20000:30000, 450:550] = np.sin(np.linspace(0, np.pi, 100))*2
mf_in = mf_in.T

ion_out = np.concatenate([np.zeros((70, 10)), np.full((10, 10), 1),
                          np.zeros((40, 10)), np.full((10, 10), 1),
                          np.zeros((40, 10)), np.full((10, 10), 1),
                          np.zeros((50, 10)), np.full((10, 10), 1),
                          np.zeros((55, 10)), np.full((5, 10), 1),
                          np.zeros((55, 10)), np.full((5, 10), 1),
                          np.zeros((55, 10)), np.full((5, 10), 1),
                          np.zeros((55, 10)), np.full((5, 10), 1),
                          np.zeros((55, 10)), np.full((5, 10), 1),
                          np.zeros((60, 10))], axis=0)
ion_out.T[0] = 0
ion_out.T[1][0:150] = 0


grc = cb.Cell(100000, mf_in_size, 10000, 4, 4)
goc = cb.Cell(10000, mf_in_size, 0, 4, 0)
pc = cb.PurkinjeCell(10, 100000, 0, 10000, 0)
dcn_Glu = cb.Cell(40, 30000, 10, 100, 10)
dcn_Gaba = cb.Cell(40, 30000, 10, 100, 10)

hist = np.zeros((600, 16))

for ii in range(600):
    print(ii)
    dcn_Glu.update(mf_in[ii], pc.cell_output, learning=False)
    dcn_Gaba.update(mf_in[ii], pc.cell_output, learning=False)
    pc.update(ion_out[ii], grc.cell_output, np.empty(0), learning=True)
    grc.update(mf_in[ii], goc.cell_output*0.5, learning=False)
    goc.update(mf_in[ii], np.empty(0), learning=False)

    hist[ii][:4] = grc.cell_output[0:4]
    hist[ii][4:8] = pc.cell_output[0:4]
    hist[ii][8:12] = dcn_Glu.cell_output[0:4]
    hist[ii][12:] = ion_out[ii][0:4]


plt.subplot(1, 5, 1)
plt.plot(mf_in[:, [10, 6000, 13000, 20000]])
plt.subplot(1, 5, 2)
plt.plot(hist[:, :4])
plt.subplot(1, 5, 3)
plt.plot(hist[:, 4:8])
plt.subplot(1, 5, 4)
plt.plot(hist[:, 8:12])
plt.subplot(1, 5, 5)
plt.plot(hist[:, 12:])
plt.show()
