import os
import json

def prefix_dict_keys(d: dict, prefix):
    outdict = dict()
    for key, val in d.items():
        outdict[prefix + key] = val
    return outdict

def read_json(p: str, *args, **kwargs):
    with open(p, "r") as f:
        out = json.load(f, *args, **kwargs)
    return out


def write_json(o: object, p: str, *args, **kwargs):
    with open(p, "w") as f:
        json.dump(o, f, *args, **kwargs)


def copy_python_files(src: str, dest: str):
    for dirpath, dirnames, fnames in os.walk(src):
        src_files = [os.path.join(dirpath, f) for f in fnames if f.endswith(".py")]
        if src_files:
            dest_dir = os.path.join(dest, dirpath.strip(src).strip("/"))
            os.makedirs(dest_dir, exist_ok=True)
            os.system(f"cp {' '.join(src_files)} {dest_dir}")
