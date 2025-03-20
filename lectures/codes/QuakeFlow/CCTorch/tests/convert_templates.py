# %%
import h5py
from pathlib import Path
from tqdm import tqdm
import numpy as np
import json

# %%
h5_templates = sorted(list(Path("../templates_hdf5").glob("*.h5")))

# %%
output_path = Path("../event_data")
if not output_path.exists():
    output_path.mkdir()

# %%
data = []
for h5_template in h5_templates:
    with h5py.File(h5_template, "r") as f1:
        with h5py.File(output_path / (h5_template.stem+"_P.h5"), "w") as f2:
            f2.create_dataset("data", data=f1["data/P/data"][()])
        with h5py.File(output_path / (h5_template.stem+"_S.h5"), "w") as f2:
            f2.create_dataset("data", data=f1["data/S/data"][()])

# %%
h5_templates = sorted(list(Path("../event_data").glob("*.h5")))

output_path = Path("../templates_raw")
if not output_path.exists():
    output_path.mkdir()

# %%
def convert_template(h5_template, output_path):
    # %%
    data_p = []
    data_s = []
    for h5_template in h5_templates:
        with h5py.File(h5_template, "r") as f:
            # print(h5_template.name)
            # for key in f.keys():
            #     print(f"  {key}: {f[key].shape}")
            # print(f["data"].shape)
            # print(f["data"].shape)
            # data.append(np.stack([f["data"][()], f["data"][()]])) # (nx, nt)
            if h5_template.name.endswith("_P.h5"):
                data_p.append(f["data"][()])
            elif h5_template.name.endswith("_S.h5"):
                data_s.append(f["data"][()])
            else:
                raise ValueError("Unknown template type")
            # data.append(f["data"][()]) # (nx, nt)

    # data = np.array(data)
    data = np.stack([np.array(data_p), np.array(data_s)], axis=1)
    print(data.shape)

    fp = np.memmap(output_path / "template.dat", dtype='float32', mode='w+', shape=data.shape)
    fp[:] = data[:]
    fp.flush()
    # np.savez(output_path / "templates.npz", data=data)

    # %%
    config = {"template_shape": data.shape}
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f)

    # %%
    with open(output_path / "event_pair.txt", "w") as f:
        for i in range(data.shape[0]):
            for j in range(i+1, data.shape[0]):
                f.write(f"{i},{j}\n")

# %%

h5_templates = sorted(list(Path("../event_data").glob("*.h5")))

output_path = Path("../templates_raw")
if not output_path.exists():
    output_path.mkdir()

convert_template(h5_templates, output_path)

# %%
h5_templates = sorted(list(Path("../../decompressed_template/wavelet").glob("*.h5")))

output_path = Path("../templates_compressed")
if not output_path.exists():
    output_path.mkdir()

convert_template(h5_templates, output_path)
