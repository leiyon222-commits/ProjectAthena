from research.cross_asset_structure_engine import run

if __name__ == "__main__":
    run("ATH-XHORIZON-ALIGN-001", "horizon_align", ("1h", "4h", "1d"),
        {"short_pair": ("1h", "4h"), "include_week": ("1h", "4h", "1d", "1w")})
