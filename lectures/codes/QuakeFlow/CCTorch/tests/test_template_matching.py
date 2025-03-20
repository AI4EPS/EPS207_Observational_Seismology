# %%
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# %%
path_origin = Path("../templates_raw/ccpairs")
df_origin = []
for txt in path_origin.glob("*.txt"):
    print(txt.name)
    tmp = pd.read_csv(txt, names=["id1", "id2", "P", "S"])
    df_origin.append(tmp)
df_origin = pd.concat(df_origin)
origin_rate = np.zeros(len(df_origin))

# %%
path_compressed = Path("../templates_compressed/ccpairs")
df_compressed = []
for txt in path_compressed.glob("*.txt"):
    print(txt.name)
    tmp = pd.read_csv(txt, names=["id1", "id2", "P", "S"])
    df_compressed.append(tmp)
df_compressed = pd.concat(df_compressed)

compress_rate = np.ones(len(df_compressed))
# %%
fig, ax = plt.subplots(1, 1, squeeze=False)
ax[0, 0].scatter(origin_rate, df_origin["P"], c="r", marker="+", s=50, label="P")
ax[0, 0].scatter(origin_rate, df_origin["S"], c="b", marker="+", s=50, label="S")
ax[0, 0].scatter(compress_rate, df_compressed["P"], c="r", marker="+", s=50)
ax[0, 0].scatter(compress_rate, df_compressed["S"], c="b", marker="+", s=50)
ax[0, 0].legend()
# %%
