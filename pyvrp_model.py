import argparse
import os
import time
import pyvrp

def solve_batch(batch_dir: str, max_time: float):
    assert os.path.isdir(batch_dir)
    instances = []
    sum_of_distances = []
    
    filenames = os.listdir(batch_dir)
    filenames = [f for f in filenames if os.path.isfile(f)]


def load_vrplib_dir(dirname: str):
    exts = {".vrp", ".vrplib", ".mdvrp"}  # be liberal about extensions
    items = []
    for fn in sorted(os.listdir(dirname)):
        path = os.path.join(dirname, fn)
        if not os.path.isfile(path):
            continue
        _, ext = os.path.splitext(fn)
        if ext.lower() in exts:
            try:
                items.append((fn, pyvrp.read(path)))
            except Exception as e:
                print(f"[WARN] Skipping {fn}: cannot read as VRPLIB ({e})")
    if not items:
        raise RuntimeError(f"No VRPLIB files found in: {dirname}")
    return items


def solve_instance(name: str, instance, max_time: float):
    start = time.perf_counter()
    res = pyvrp.solve(instance, stop=pyvrp.stop.MaxRuntime(max_time))
    elapsed = time.perf_counter() - start

    cost = None
    try:
        cost = res.cost()          
    except Exception:
        try:
            cost = res.objective()
        except Exception:
            pass

    summary = {
        "instance": name,
        "time_s": round(elapsed, 3),
        "cost": cost,
        "repr": str(res),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="PyVRP model execution")
    parser.add_argument(
        "--mode",
        default="eval_batch",
        choices=["eval_single", "eval_batch"],
        help="Run a single file or a whole folder of VRPLIB instances",
    )
    parser.add_argument(
        "--dataset_dir",
        default=None,
        type=str,
        help="[eval_batch] Folder containing VRPLIB files",
    )
    parser.add_argument(
        "--instance_path",
        default=None,
        type=str,
        help="[eval_single] Path to one VRPLIB file",
    )
    parser.add_argument(
        "--max_time",
        type=float,
        default=10.0,
        help="Max runtime per instance in seconds (float).",
    )

    args = parser.parse_args()

    if args.mode == "eval_single":
        if not args.instance_path or not os.path.isfile(args.instance_path):
            raise SystemExit("Provide a valid --instance_path to a VRPLIB file.")
        inst = pyvrp.read(args.instance_path)
        name = os.path.basename(args.instance_path)
        summary = solve_instance(name, inst, args.max_time)
        print(f"{summary['instance']}: cost={summary['cost']} time_s={summary['time_s']}")
        print(summary["repr"])
        return

    if not args.dataset_dir or not os.path.isdir(args.dataset_dir):
        raise SystemExit("Provide a valid --dataset_dir pointing to a folder of VRPLIB files.")

    items = load_vrplib_dir(args.dataset_dir)
    results = []
    for name, inst in items:
        print(f"[INFO] Solving {name} (limit {args.max_time}s) ...")
        summary = solve_instance(name, inst, args.max_time)
        results.append(summary)
        print(f"  -> cost={summary['cost']} time_s={summary['time_s']}")

    print("\n=== Batch summary ===")
    width = max(len(r["instance"]) for r in results)
    print(f"{'instance'.ljust(width)}  {'cost':>12}  {'time_s':>8}")
    for r in results:
        cost_str = "None" if r["cost"] is None else f"{r['cost']:,.6f}"
        print(f"{r['instance'].ljust(width)}  {cost_str:>12}  {r['time_s']:>8.3f}")


if __name__ == "__main__":
    main()

