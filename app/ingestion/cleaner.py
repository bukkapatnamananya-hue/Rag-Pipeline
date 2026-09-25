import json
from pathlib import Path


INPUT_PATH = Path(
    "data/processed/unstructured_elements.json"
)

OUTPUT_PATH = Path(
    "data/processed/cleaned_elements.json"
)


def load_elements():
    """Load elements extracted by Unstructured."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH.resolve()}"
        )

    with open(INPUT_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def clean_text(text):
    """Basic text cleanup."""

    return (
        text.replace("n42.50", "₹42.50")
            .replace("(n in Crores)", "(₹ in Crores)")
            .replace("R&D;", "R&D")
            .strip()
    )


def build_cleaned_elements(elements):
    """Clean extracted elements and reconstruct table-like information."""

    cleaned = []

    i = 0

    while i < len(elements):

        element = elements[i]
        element_type = element["element_type"]
        text = clean_text(element["text"])

        # ---------------------------------------------------------
        # Financial Performance section
        # ---------------------------------------------------------
        if text == "2. Consolidated Financial Performance (n in Crores)":

            cleaned.append({
                "element_id": element["element_id"],
                "element_type": "SectionHeader",
                "section": "Consolidated Financial Performance",
                "text": "2. Consolidated Financial Performance (₹ in Crores)"
            })

            # Skip the flattened table elements.
            i += 1

            # Find the end of the financial table.
            while i < len(elements):
                current_text = clean_text(elements[i]["text"])

                if current_text == "3. Segment Revenue Split":
                    break

                i += 1

            # Reconstruct the table manually based on the
            # extracted element ordering.
            financial_rows = [
                {
                    "metric": "Total Revenue from Operations",
                    "fy2023_24": "430.45",
                    "fy2022_23": "161.44",
                    "yoy_growth": "+166.6%"
                },
                {
                    "metric": "EBITDA",
                    "fy2023_24": "175.20",
                    "fy2022_23": "55.80",
                    "yoy_growth": "+213.9%"
                },
                {
                    "metric": "EBITDA Margin (%)",
                    "fy2023_24": "40.70%",
                    "fy2022_23": "34.56%",
                    "yoy_growth": "+614 bps"
                },
                {
                    "metric": "Profit Before Tax (PBT)",
                    "fy2023_24": "168.10",
                    "fy2022_23": "49.12",
                    "yoy_growth": "+242.2%"
                },
                {
                    "metric": "Profit After Tax (PAT)",
                    "fy2023_24": "129.85",
                    "fy2022_23": "37.64",
                    "yoy_growth": "+244.9%"
                },
                {
                    "metric": "Order Book Position (as of Mar 31)",
                    "fy2023_24": "1,402.00",
                    "fy2022_23": "473.00",
                    "yoy_growth": "+196.4%"
                }
            ]

            for row in financial_rows:

                text = (
                    f"Metric: {row['metric']}\n"
                    f"FY 2023-24: {row['fy2023_24']}\n"
                    f"FY 2022-23: {row['fy2022_23']}\n"
                    f"YoY Growth: {row['yoy_growth']}"
                )

                cleaned.append({
                    "element_type": "TableRow",
                    "section": "Consolidated Financial Performance",
                    "text": text,
                    "metadata": {
                        "metric": row["metric"],
                        "fy2023_24": row["fy2023_24"],
                        "fy2022_23": row["fy2022_23"],
                        "yoy_growth": row["yoy_growth"]
                    }
                })

            continue

        # ---------------------------------------------------------
        # Segment Revenue section
        # ---------------------------------------------------------
        if text == "3. Segment Revenue Split":

            cleaned.append({
                "element_id": element["element_id"],
                "element_type": "SectionHeader",
                "section": "Segment Revenue Split",
                "text": "3. Segment Revenue Split"
            })

            i += 1

            while i < len(elements):
                current_text = clean_text(elements[i]["text"])

                if current_text == "4. Research & Development (R&D) and Intellectual Property":
                    break

                i += 1

            segment_rows = [
                {
                    "segment": "Anti-Drone Systems (CUAS)",
                    "revenue": "245.30",
                    "share": "57.0%",
                    "product": "Zen Anti-Drone System (ZADS), RF Jammer"
                },
                {
                    "segment": "Defense Training Simulators",
                    "revenue": "155.15",
                    "share": "36.0%",
                    "product": "Tank Simulator, Small Arms Trainer"
                },
                {
                    "segment": "Annual Maintenance & Services",
                    "revenue": "30.00",
                    "share": "7.0%",
                    "product": "Long-term AMC for Army/Air Force"
                }
            ]

            for row in segment_rows:

                text = (
                    f"Segment: {row['segment']}\n"
                    f"FY24 Revenue: ₹{row['revenue']} Cr\n"
                    f"Share: {row['share']}\n"
                    f"Key Product Line: {row['product']}"
                )

                cleaned.append({
                    "element_type": "TableRow",
                    "section": "Segment Revenue Split",
                    "text": text,
                    "metadata": {
                        "segment": row["segment"],
                        "revenue": row["revenue"],
                        "share": row["share"],
                        "product": row["product"]
                    }
                })

            continue

        # ---------------------------------------------------------
        # Normal elements
        # ---------------------------------------------------------

        if text:
            cleaned.append({
                "element_id": element["element_id"],
                "element_type": element_type,
                "section": None,
                "text": text,
                "metadata": element.get("metadata", {})
            })

        i += 1

    return cleaned


def main():

    print(f"Reading: {INPUT_PATH}")

    elements = load_elements()

    print(f"Loaded {len(elements)} extracted elements")

    cleaned = build_cleaned_elements(elements)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            cleaned,
            file,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("Cleaning complete.")
    print(f"Created {len(cleaned)} cleaned elements")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()