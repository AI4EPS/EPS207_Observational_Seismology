# %%
import json
import os
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import signal
from tqdm import tqdm

# %%
nx = 11
nt = 1000
base_vec = signal.ricker(nt, nt // 30)

# %%
plt.figure()
plt.plot(base_vec)
plt.show()

# %%
vec = []
dt = []
for i in range(nx):
    vec.append(np.roll(base_vec, i)[::10])
    # dt.append(np.argmax(np.roll(base_vec, i))/10 - nt//20)

# %%
vec = np.array(vec)
plt.figure()
plt.matshow(vec)
plt.show()

# %%
nx, nt = vec.shape
template_shape = (nx, 1, 1, nt)
vec = vec[:, np.newaxis, np.newaxis, :]
fp = np.memmap("test_dtcc.dat", dtype="float32", mode="w+", shape=template_shape)
fp[:] = vec[:]
fp.flush()

# %%

# evid1 = range(0, 9)
# evid2 = range(1, 10)

pairs = []
for i in range(0, nx - 1):
    pairs.append((i, i + 1))

with open("test_dtcc_pair.txt", "w") as f:
    for p in pairs:
        f.write(f"{p[0]},{p[1]}\n")

with open("test_dtcc_list.txt", "w") as f:
    for i in range(nx):
        f.write(f"{i}\n")

# %%
config = {
    "template_shape": template_shape,
    "min_cc_score": 0.6,
    "min_cc_diff": 0.0,
}
with open("test_dtcc.json", "w") as f:
    json.dump(config, f)


# %%
# os.system("rm -rf ccpairs")
# os.system("python ../run.py --pair-list=test_dtcc_pair.txt  --data-path=test_dtcc.dat --data-format=memmap --config=test_dtcc.json  --batch-size=1  --result-path=ccpairs")
# os.system("torchrun --standalone --nproc_per_node=2 ../run.py --pair-list=test_dtcc.txt  --data-path=test_dtcc.dat --data-format=memmap --config=test_dtcc.json  --batch-size=1  --result-path=ccpairs")

os.system("rm -rf ccpairs")
os.system(
    "python ../run.py --data-list1=test_dtcc_list.txt  --data-path=test_dtcc.dat --data-format=memmap --config=test_dtcc.json  --batch-size=1  --result-path=ccpairs --auto-xcorr"
)

# %%
dt = 0.01
dt_cubic = dt / 100
x = np.linspace(0, 1, 2 + 1)
xs = np.linspace(0, 1, 2 * int(dt / dt_cubic) + 1)

h5_path = Path("ccpairs/")
h5_list = sorted(list(h5_path.rglob("*.h5")))
data = {}
for h5 in h5_list:
    with h5py.File(h5, "r") as fp:
        for id1 in tqdm(sorted(fp.keys(), key=lambda x: int(x))):
            gp1 = fp[id1]
            for id2 in sorted(gp1.keys(), key=lambda x: int(x)):
                cc_score = gp1[id2]["cc_score"][:]
                cc_index = gp1[id2]["cc_index"][:]
                cc_diff = gp1[id2]["cc_diff"][:]
                neighbor_score = gp1[id2]["neighbor_score"][:]

                cubic_score = scipy.interpolate.interp1d(x, neighbor_score, axis=-1, kind="quadratic")(xs)
                cubic_index = np.argmax(cubic_score, axis=-1, keepdims=True) - (len(xs) // 2)
                dt_cc = cc_index * dt + cubic_index * dt_cubic

                key = (id1, id2)
                # if int(id1) > int(id2):
                #     continue

                nch, nsta, npick = cc_score.shape
                records = []
                for i in range(nch):
                    for j in range(nsta):
                        if cc_score[i, j, 0] > config["min_cc_score"]:
                            # records.append([f"{j:05d}", dt_ct + dt_cc[best, j, 0], cc_score[best, j, 0]*cc_diff[best, j], phase_list[i]])
                            # if cc_diff[i, j] > config["min_cc_diff"]:
                            records.append(
                                [
                                    f"sta{j}",
                                    np.round(dt_cc[i, j, 0], 4),
                                    np.round(cc_score[i, j, 0], 4),
                                    np.round(cc_diff[i, j], 4),
                                    f"phase{i}",
                                ]
                            )

                data[key] = records

print(f"{len(data) = }")
for k in data:
    print(k, data[k])
# %%
