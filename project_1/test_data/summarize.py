import json

def count_lines(path):
    try:
        with open(path, 'r') as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0

def load_controller_snap(path="/opt/app/snap.json"):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return {}

def summarize_run(patterns_path, truth_count, controller_snap="/opt/app/snap.json"):
    found = count_lines(patterns_path)
    snap = load_controller_snap(controller_snap)
    p95  = snap.get("p95")
    recall = (found / truth_count) if truth_count > 0 else None
    return {
        "found": found,
        "truth": truth_count,
        "recall": recall,
        "controller_p95": p95
    }

if __name__ == "__main__":
    import sys
    pat = sys.argv[1]
    truth = int(sys.argv[2])
    print(summarize_run(pat, truth))
