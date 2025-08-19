# utils/load_save_common.py
import os, re, json, datetime
import pandas as pd

def find_next_version(base_dir: str, prefix: str = "model") -> str:
    """
    Scannt base_dir nach Ordnern 'prefix_XX' und gibt die nächste 'XX' (zweistellig) zurück.
    Legt base_dir an, falls nicht existent.
    """
    os.makedirs(base_dir, exist_ok=True)
    pat = re.compile(rf"^{re.escape(prefix)}_(\d{{2}})$")
    nums = []
    for name in os.listdir(base_dir):
        m = pat.match(name)
        if m:
            try:
                nums.append(int(m.group(1)))
            except ValueError:
                pass
    return f"{(max(nums)+1) if nums else 1:02d}"

def prepare_run_dirs(models_root: str, family: str, version: str,
                     prefix: str = "model") -> dict:
    """
    Legt die Run-Verzeichnisstruktur an und gibt Pfade zurück.
    family z.B. 'k1a1', version z.B. '01'
    """
    family_dir = os.path.join(models_root, family)
    run_dir    = os.path.join(family_dir, f"{prefix}_{version}")
    plots_dir  = os.path.join(run_dir, "plots")
    weights_dir= os.path.join(run_dir, "models")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    return {
        "family_dir": family_dir,
        "run_dir": run_dir,
        "plots_dir": plots_dir,
        "weights_dir": weights_dir,
        "config_csv": os.path.join(run_dir, "config.csv"),
        "timings_csv": os.path.join(run_dir, "timings.csv"),
        "run_meta_json": os.path.join(run_dir, "run_meta.json"),
    }

def save_config_csv(config_dict: dict, path: str):
    pd.DataFrame([config_dict]).to_csv(path, index=False)

def save_run_meta(meta: dict, path: str):
    meta = dict(meta)
    meta.setdefault("timestamp", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
