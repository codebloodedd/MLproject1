import pandas as pd
from pathlib import Path

INPUTS = {
    "support": "top15_support.csv",
    "confidence": "top15_confidence.csv",
    "lift": "top15_lift.csv",
}

OUT_DIR = Path("website/assets/data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def clean_set_str(x: str) -> str:
    # Handles strings like: "frozenset({'A', 'B'})"
    x = str(x)
    x = x.replace("frozenset(", "").replace(")", "")
    x = x.strip()
    # Remove braces/quotes
    x = x.replace("{", "").replace("}", "").replace("'", "").replace('"', "")
    # Split nicely
    parts = [p.strip() for p in x.split(",") if p.strip()]
    return ", ".join(parts) if parts else "∅"

def format_rules(df: pd.DataFrame) -> str:
    lines = []
    for i, row in df.iterrows():
        ant = clean_set_str(row.get("antecedents", ""))
        cons = clean_set_str(row.get("consequents", ""))
        sup = row.get("support", "")
        conf = row.get("confidence", "")
        lift = row.get("lift", "")

        lines.append(
            f"{len(lines)+1}. IF {ant} THEN {cons} "
            f"(support={sup:.3f}, confidence={conf:.3f}, lift={lift:.3f})"
        )
    return "\n".join(lines)

def main():
    for key, fname in INPUTS.items():
        df = pd.read_csv(fname)
        # Keep only the columns we need, safely
        for col in ["support", "confidence", "lift"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        text = format_rules(df)
        out_path = OUT_DIR / f"top15_{key}.txt"
        out_path.write_text(text, encoding="utf-8")
        print("Wrote:", out_path)

if __name__ == "__main__":
    main()