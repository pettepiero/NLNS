#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

def distribute_vehicles(num_depots: int, total: int):
    """
    Return a list of depot IDs (1..num_depots) repeated to sum to `total`,
    distributing as evenly as possible.
    """
    base = total // num_depots
    rem = total % num_depots
    result = []
    for d in range(1, num_depots + 1):
        count = base + (1 if d <= rem else 0)
        result.extend([d] * count)
    return result

def process_file(src_path: Path, dst_path: Path, total_vehicles: int):
    lines = src_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    # Find indices
    idx_num_depots = next((i for i, ln in enumerate(lines) if ln.strip().startswith("NUM_DEPOTS")), None)
    idx_node_section = next((i for i, ln in enumerate(lines) if ln.strip() == "NODE_COORD_SECTION"), None)
    idx_depot_section = next((i for i, ln in enumerate(lines) if ln.strip() == "DEPOT_SECTION"), None)

    if idx_node_section is None:
        raise ValueError(f"'NODE_COORD_SECTION' not found in {src_path.name}")

    # Parse NUM_DEPOTS
    if idx_num_depots is not None:
        try:
            num_depots = int(lines[idx_num_depots].split(":")[1].strip())
        except Exception:
            num_depots = 1
    else:
        num_depots = 1

    veh_by_depot = distribute_vehicles(num_depots, total_vehicles)

    # Add VEHICLES line
    vehicles_line = f"VEHICLES : {total_vehicles}"
    insert_idx_for_vehicles = idx_num_depots + 1 if idx_num_depots is not None else idx_node_section
    already_has_vehicles = any(l.strip().startswith("VEHICLES") for l in lines)
    if not already_has_vehicles:
        lines.insert(insert_idx_for_vehicles, vehicles_line)
        if insert_idx_for_vehicles <= idx_node_section:
            idx_node_section += 1
        if idx_depot_section is not None and insert_idx_for_vehicles <= idx_depot_section:
            idx_depot_section += 1

    # Add VEHICLES_DEPOT_SECTION
    vds_block = ["VEHICLES_DEPOT_SECTION"] + [str(d) for d in veh_by_depot]
    for i, blk_line in enumerate(vds_block):
        lines.insert(idx_node_section + i, blk_line)

    # Ensure DEPOT_SECTION ends with -1
    if idx_depot_section is not None:
        # find the next "EOF" line
        try:
            idx_eof = next(i for i, ln in enumerate(lines) if ln.strip() == "EOF")
        except StopIteration:
            idx_eof = len(lines)
        # check if -1 already present just before EOF
        if lines[idx_eof - 1].strip() != "-1":
            lines.insert(idx_eof, "-1")

    # Write out
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Copy and modify MDVRP instances by adding VEHICLES and VEHICLES_DEPOT_SECTION.")
    ap.add_argument("src_dir", type=Path, help="Source directory containing .mdvrp files")
    ap.add_argument("dst_dir", type=Path, help="Destination directory for modified files")
    ap.add_argument("--vehicles", "-v", type=int, default=15, help="Total number of vehicles to add (default: 15)")
    args = ap.parse_args()

    for src_file in sorted(args.src_dir.glob("*.mdvrp")):
        dst_file = args.dst_dir / src_file.name
        process_file(src_file, dst_file, args.vehicles)

if __name__ == "__main__":
    main()

