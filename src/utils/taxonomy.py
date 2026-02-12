from __future__ import annotations

FIELD_SUBFIELD: dict[str, list[str]] = {
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
    "medicine_health": [
        "general_medicine_health",
        "clinical_medicine",
        "epidemiology_public_health",
        "biomedical_research",
    ],
    "chemistry": [
        "general_chemistry",
        "organic_chemistry",
        "inorganic_chemistry",
        "analytical_chemistry",
        "biochemistry",
    ],
    "physics": [
        "general_physics",
        "applied_physics",
        "astrophysics",
        "condensed_matter",
    ],
    "materials_science": [
        "general_materials_science",
        "biomaterials",
        "nanomaterials",
        "polymers",
    ],
    "earth_science": [
        "general_earth_science",
        "geology",
        "geophysics",
        "oceanography",
    ],
    "environmental_science": [
        "general_environmental_science",
        "ecology",
        "conservation",
        "pollution",
    ],
    "climate_science": [
        "general_climate_science",
        "climate_modeling",
        "atmospheric_science",
        "carbon_cycle",
    ],
    "agriculture_food_science": [
        "general_agriculture_food_science",
        "crop_science",
        "soil_science",
        "food_nutrition",
    ],
    "computer_science": [
        "general_computer_science",
        "systems_networks",
        "software_engineering",
        "theory_algorithms",
    ],
    "data_science_ai": [
        "general_data_science_ai",
        "machine_learning",
        "natural_language_processing",
        "computer_vision",
    ],
    "mathematics_statistics": [
        "general_mathematics_statistics",
        "pure_mathematics",
        "applied_mathematics",
        "statistics",
    ],
    "engineering": [
        "general_engineering",
        "electrical_engineering",
        "mechanical_engineering",
        "chemical_engineering",
        "biomedical_engineering",
    ],
    "neuroscience_psychology": [
        "general_neuroscience_psychology",
        "cognitive_science",
        "behavioral_science",
        "psychiatry_mental_health",
    ],
    "social_science": [
        "general_social_science",
        "sociology",
        "political_science",
        "anthropology",
    ],
    "economics_finance": [
        "general_economics_finance",
        "macroeconomics",
        "microeconomics",
        "finance",
    ],
    "linguistics_language": [
        "general_linguistics_language",
        "theoretical_linguistics",
        "applied_linguistics",
        "language_technology",
    ],
    "history_archaeology": [
        "general_history_archaeology",
        "history",
        "archaeology",
        "digital_history",
    ],
    "philosophy_ethics": [
        "general_philosophy_ethics",
        "ethics",
        "philosophy_of_science",
        "logic",
    ],
    "education": [
        "general_education",
        "learning_sciences",
        "higher_education",
        "education_policy",
    ],
    "law_policy": [
        "general_law_policy",
        "regulatory_policy",
        "technology_policy",
        "health_policy",
    ],
    "business_management": [
        "general_business_management",
        "operations",
        "strategy",
        "organizational_behavior",
    ],
    "arts_humanities": [
        "general_arts_humanities",
        "literature",
        "cultural_studies",
        "media_studies",
    ],
    "interdisciplinary": [
        "general_interdisciplinary",
        "data_resource",
        "methods_benchmark",
        "uncategorized",
    ],
}

LEGACY_FIELD_MAP: dict[str, str] = {
    "computational": "data_science_ai",
    "environmental": "environmental_science",
    "clinical": "medicine_health",
    "general_science": "interdisciplinary",
}

LEGACY_SUBFIELD_MAP: dict[tuple[str, str], str] = {
    ("computational", "bioinformatics"): "machine_learning",
    ("computational", "machine_learning"): "machine_learning",
    ("computational", "data_resource"): "data_resource",
    ("environmental", "climate"): "general_climate_science",
    ("environmental", "ecology"): "ecology",
    ("environmental", "earth_systems"): "general_earth_science",
    ("clinical", "translational"): "biomedical_research",
    ("clinical", "epidemiology"): "epidemiology_public_health",
    ("clinical", "public_health"): "epidemiology_public_health",
    ("general_science", "uncategorized"): "uncategorized",
}


def all_categories_text() -> str:
    lines: list[str] = []
    for category, subcats in FIELD_SUBFIELD.items():
        lines.append(f"- {category}: {', '.join(subcats)}")
    return "\n".join(lines)


def category_map() -> dict[str, list[str]]:
    return {k: list(v) for k, v in FIELD_SUBFIELD.items()}


def is_valid_category_subcategory(category: str | None, subcategory: str | None) -> bool:
    c = (category or "").strip().lower()
    s = (subcategory or "").strip().lower()
    if c not in FIELD_SUBFIELD:
        return False
    return s in FIELD_SUBFIELD[c]


def normalize_category_subcategory(
    category: str | None,
    subcategory: str | None,
) -> tuple[str, str]:
    c = (category or "").strip().lower()
    s = (subcategory or "").strip().lower()
    if c in LEGACY_FIELD_MAP:
        mapped = LEGACY_FIELD_MAP[c]
        mapped_sub = LEGACY_SUBFIELD_MAP.get((c, s), FIELD_SUBFIELD[mapped][0])
        return mapped, mapped_sub
    if c in FIELD_SUBFIELD:
        if s in FIELD_SUBFIELD[c]:
            return c, s
        return c, FIELD_SUBFIELD[c][0]
    return "interdisciplinary", "uncategorized"


# Backward-compatible aliases while codebase transitions naming from category/subcategory to field/subfield.
CATEGORY_SUBCATEGORY = FIELD_SUBFIELD
