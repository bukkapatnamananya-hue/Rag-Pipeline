import json
from pathlib import Path


INPUT_PATH = Path(
    "data/processed/cleaned_elements.json"
)


with open(INPUT_PATH, "r", encoding="utf-8") as file:
    elements = json.load(file)


print(f"Total cleaned elements: {len(elements)}")

for element in elements:

    print("\n" + "=" * 80)

    print(
        f"TYPE: {element['element_type']} | "
        f"SECTION: {element.get('section')}"
    )

    print("=" * 80)

    print(element["text"])