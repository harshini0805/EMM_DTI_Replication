"""Quick script to explore data directory structure."""

from pathlib import Path
import os

root = Path(__file__).parent / "human_random"

print("=" * 80)
print("Data Structure Explorer")
print("=" * 80)

for split in ["train", "test", "valid"]:
    split_dir = root / split
    print(f"\n{split}/ contents:")
    if split_dir.exists():
        files = sorted(split_dir.iterdir())
        for f in files:
            size = f.stat().st_size if f.is_file() else "DIR"
            print(f"  {f.name:30s} {size}")

            # Show first 3 lines if text file
            if f.is_file() and f.suffix in ['.csv', '.txt']:
                try:
                    with open(f, 'r') as file:
                        lines = [next(file) for _ in range(3)]
                    print(f"    Preview:")
                    for i, line in enumerate(lines, 1):
                        print(f"      Line {i}: {line[:80]}")
                except:
                    pass
    else:
        print(f"  [Directory not found]")

print("\n" + "=" * 80)
