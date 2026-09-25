import json
from pathlib import Path


INPUT_PATH = Path(
    "data/processed/unstructured_elements.json"
)


with open(INPUT_PATH, "r", encoding="utf-8") as file:
    elements = json.load(file)


print(f"Total elements: {len(elements)}")

for element in elements:
    print("\n" + "=" * 80)
    print(
        f"ID: {element['element_id']} | "
        f"TYPE: {element['element_type']}"
    )
    print("=" * 80)

    print(element["text"])