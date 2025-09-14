import argparse
import os
import time
import pyvrp
from pyvrp import plotting

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
    return summary, res


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
    parser.add_argument(
        "--plot_coordinates", "--plot-coordinates",
        dest='plot_coordinates',
        action='store_true',
        help="Plot instance coordinates for eval_single mode.",
    )
    parser.add_argument(
        "--plot_demands", "--plot-demands",
        dest='plot_demands',
        action='store_true',
        help="Plot instance demands for eval_single mode.",
    )
    parser.add_argument(
        "--plot_diversity", "--plot-diversity",
        dest='plot_diversity',
        action='store_true',
        help="Plot instance diversity for eval_single mode.",
    )
    parser.add_argument(
        "--plot_result", "--plot-result",
        dest='plot_result',
        action='store_true',
        help="Plot instance result for eval_single mode.",
    )
    parser.add_argument(
        "--plot_solution", "--plot-solution",
        dest='plot_solution',
        action='store_true',
        help="Plot instance solution for eval_single mode.",
    )
    args = parser.parse_args()

    if args.mode == "eval_single":
        if not args.instance_path or not os.path.isfile(args.instance_path):
            raise SystemExit("Provide a valid --instance_path to a VRPLIB file.")
        inst = pyvrp.read(args.instance_path)
        if args.plot_coordinates:
            pyvrp.plotting.plot_coordinates(data=inst, title=f"{args.instance_path} instance")
        if args.plot_demands:
            pyvrp.plotting.plot_demands(data=inst, title=f"{args.instance_path} instance")
        name = os.path.basename(args.instance_path)
        summary, res = solve_instance(name, inst, args.max_time)
        print(f"{summary['instance']}: cost={summary['cost']} time_s={summary['time_s']}")
        print(summary["repr"])
        if args.plot_diversity:
            pyvrp.plotting.plot_diversity(res)
        if args.plot_result:
            pyvrp.plotting.plot_result(result=res, data=inst)
        if args.plot_solution:
            pyvrp.plotting.plot_solution(solution=res.best, data=inst)
        return

    # batch eval
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

