# %%
import glob
import os

import h5py
import pandas as pd

# %%
root_path = "tests"
data_path = "data"

# %%
# wget to data_path https://github.com/AI4EPS/CCTorch/releases/download/test_ambient_noise/Ridgecrest_ODH3-2021-06-15.183838Z.h5
file_path = f"{root_path}/{data_path}/Ridgecrest_ODH3-2021-06-15.183838Z.h5"
if not os.path.exists(file_path):
    os.system(
        f"wget -P {root_path}/{data_path} https://github.com/AI4EPS/CCTorch/releases/download/test_ambient_noise/Ridgecrest_ODH3-2021-06-15.183838Z.h5"
    )

# %%
files = glob.glob(f"{root_path}/{data_path}/*.h5")

# %%
pair_list = []
data_list = []

# %%
for file in files:
    with h5py.File(file, "r") as f:
        nx, nt = f["Data"].shape
        for i in range(nx):
            data_list.append([f"{file}", i])

data_list = pd.DataFrame(data_list, columns=["file_name", "channel_index"])
data_list.to_csv("data_list.txt", index=False)

# %%
select_chn = 500
ii = data_list[data_list["channel_index"] == select_chn].index[0]
pair_list = []
for ij, row in data_list.iterrows():
    if row["channel_index"] != select_chn:
        pair_list.append(f"{ii},{ij}")

# %%
with open("pair_list.txt", "w") as f:
    for pair in pair_list:
        f.write(pair + "\n")
