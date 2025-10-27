import pandas as pd
from pathlib import Path
import os
import logging

log = logging.getLogger(__name__)

def save_traces_to_pickle(project_dir: Path):
    project_dir = Path(project_dir)
    if not project_dir.is_dir():
        raise ValueError(f"Not a directory: {project_dir}")

    files = sorted(project_dir.glob("objective_trace*.csv"))
    if not files:
        raise SystemExit(f"No files matched 'objective_trace*.csv' in {project_dir}")

    frames = []
    for f in files:
        df = pd.read_csv(f)
        #expecting columns: time_sec, incumbent_cost

        df = df.rename(columns={"time": "time_sec"})
        df["time_sec"] = pd.to_numeric(df["time_sec"], errors="coerce")
        df["incumbent_cost"] = pd.to_numeric(df["incumbent_cost"], errors="coerce")

        df = df.dropna(subset=["time_sec", "incumbent_cost"])

        df["source"] = f.stem
        frames.append(df[["time_sec", "incumbent_cost", "source"]])


    combined = pd.concat(frames, ignore_index=True, copy=False)

    combined = combined.sort_values(["source", "time_sec"]).reset_index(drop=True)
   
    output_path = project_dir / "traces.pkl"
    combined.to_pickle(output_path)

    msg = f"Saved {len(combined)} rows from {len(files)} files to {output_path}"
    print(msg)
    log.info(msg)
