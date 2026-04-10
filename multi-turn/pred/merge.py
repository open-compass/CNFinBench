import json
import re
import argparse
from pathlib import Path
from collections import OrderedDict

def extract_round_index(round_value):
    """
    extract round index from round_num field:
    """
    if round_value is None:
        return None
    s = str(round_value)
    m = re.findall(r'\d+', s)
    if not m:
        return None
    return m[0] 


def process_jsonl_file(file_path: Path, data_by_id: OrderedDict):
    """
    process a jsonl file, merge data to data_by_id

    Args:
        file_path: jsonl file path
        data_by_id: data dictionary organized by id
    """
    print(f"  processing file: {file_path.name}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                obj = json.loads(line)

                # 1) delete row_index field
                obj.pop("row_index", None)

                # 2) process round related fields
                round_val = obj.get("round_num")
                round_idx = extract_round_index(round_val)

                if round_idx is not None:
                    suffix = f"_{round_idx}"

                    # rename fields: round_num, estimated_risk_escalation, strategy_chosen, reason
                    for key in ["round_num", "estimated_risk_escalation", "strategy_chosen", "reason"]:
                        if key in obj:
                            new_key = key + suffix
                            obj[new_key] = obj.pop(key)

                # 3) merge to the same row by id
                _id = obj.get("id")
                if _id is None:
                    # skip rows without id
                    continue

                if _id not in data_by_id:
                    data_by_id[_id] = OrderedDict()

                # merge fields: the same field name will be overwritten by the later one (usually the same value)
                for k, v in obj.items():
                    data_by_id[_id][k] = v
                    
            except json.JSONDecodeError as e:
                print(f"    warning: JSON parse failed, skip this line: {e}")
                continue
            except Exception as e:
                print(f"    warning: processing failed, skip this line: {e}")
                continue


def process_subdirectory(subdir: Path):
    """
    process a subdirectory, merge all jsonl files in it
    
    Args:
        subdir: subdirectory path
    """
    print(f"\nprocessing subdirectory: {subdir.name}")
    print("-" * 60)
    
    # find all jsonl files
    jsonl_files = sorted(subdir.glob("*.jsonl"))
    
    if not jsonl_files:
        print(f"  warning: no jsonl files found, skip")
        return
    
    print(f"  found {len(jsonl_files)} jsonl files")
    
    # merge data by id
    data_by_id = OrderedDict()
    
    # process each jsonl file
    for jsonl_file in jsonl_files:
        # skip output file itself, avoid duplicate processing
        if jsonl_file.name == "merged.jsonl":
            print(f"  warning: skip output file: {jsonl_file.name}")
            continue
            
        process_jsonl_file(jsonl_file, data_by_id)
    
    if not data_by_id:
        print(f"  warning: no valid data, skip")
        return
    
    # output to merged.jsonl
    output_path = subdir / "merged.jsonl"
    print(f"\n  merging {len(data_by_id)} records to: {output_path.name}")
    
    with open(output_path, "w", encoding="utf-8") as f_out:
        for _id, merged_obj in data_by_id.items():
            f_out.write(json.dumps(merged_obj, ensure_ascii=False) + "\n")
    
    print(f"  ✓ done")


def main():
    """
    main function: process all subdirectories in output directory
    """
    parser = argparse.ArgumentParser(description="merge all jsonl files in all subdirectories in output directory")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="output directory path (default: current script directory)",
    )
    
    args = parser.parse_args()
    
    # set output directory path
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # if current file is in output directory, use current directory
        current_file = Path(__file__).resolve()
        if current_file.parent.name == "output":
            output_dir = current_file.parent
        else:
            # otherwise use default path
            output_dir = Path(r"C:\Users\jiang\Desktop\dig_4rounds\output")
    
    output_dir = output_dir.resolve()
    
    if not output_dir.exists():
        print(f"error: directory not exists: {output_dir}")
        return
    
    print("=" * 60)
    print("start processing all subdirectories in output directory")
    print(f"directory: {output_dir}")
    print("=" * 60)
    
    # get all subdirectories
    subdirs = [d for d in output_dir.iterdir() if d.is_dir()]
    
    if not subdirs:
        print("warning: no subdirectories found")
        return
    
    print(f"found {len(subdirs)} subdirectories\n")
    
    # process each subdirectory
    for subdir in sorted(subdirs):
        process_subdirectory(subdir)
    
    print("\n" + "=" * 60)
    print("all processing done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
