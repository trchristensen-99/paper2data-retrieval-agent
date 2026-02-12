from __future__ import annotations

CATEGORY_SUBCATEGORY: dict[str, list[str]] = {
    "biology": [
        "general_biology",
        "genomics",
        "cancer",
        "development",
        "evolutionary_biology",
        "neuroscience",
        "immunology",
        "microbiology",
    ],
    "computational": [
        "bioinformatics",
        "machine_learning",
        "data_resource",
    ],
    "environmental": [
        "climate",
        "ecology",
        "earth_systems",
    ],
    "clinical": [
        "translational",
        "epidemiology",
        "public_health",
    ],
    "general_science": [
        "uncategorized",
    ],
}


def all_categories_text() -> str:
    lines: list[str] = []
    for category, subcats in CATEGORY_SUBCATEGORY.items():
        lines.append(f"- {category}: {', '.join(subcats)}")
    return "\n".join(lines)


def category_map() -> dict[str, list[str]]:
    return {k: list(v) for k, v in CATEGORY_SUBCATEGORY.items()}


def is_valid_category_subcategory(category: str | None, subcategory: str | None) -> bool:
    c = (category or "").strip().lower()
    s = (subcategory or "").strip().lower()
    if c not in CATEGORY_SUBCATEGORY:
        return False
    return s in CATEGORY_SUBCATEGORY[c]


def normalize_category_subcategory(
    category: str | None,
    subcategory: str | None,
) -> tuple[str, str]:
    c = (category or "").strip().lower()
    s = (subcategory or "").strip().lower()
    if c in CATEGORY_SUBCATEGORY:
        if s in CATEGORY_SUBCATEGORY[c]:
            return c, s
        return c, CATEGORY_SUBCATEGORY[c][0]
    return "general_science", "uncategorized"
