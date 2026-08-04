from research.cross_asset_structure_engine import run

if __name__ == "__main__":
    run("ATH-XHIERARCHY-CONFLICT-001", "hierarchy_conflict", (("1h", "4h"), ("1w", "1m")),
        {"h1_vs_long": (("1h",), ("1w", "1m")),
         "include_d1": (("1h", "4h", "1d"), ("1w", "1m"))})
