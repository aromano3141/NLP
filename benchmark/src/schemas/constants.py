CORE_TASK_CATEGORIES = [
    "Classification",
    "Information Extraction",
    "Text Generation",
    "Dialogue System",
    "Reasoning and Logic",
    "Language Style",
    "Programming"
]

CONSTRAINT_DIMENSIONS = {
    "Content Constraint": [
        "Theme Constraint", "Exclusion Constraint", "Inclusion Constraint", 
        "Value Constraint", "Privacy Constraint", "Numerical Constraint"
    ],
    "Situation Constraint": [
        "Role-Playing Constraint", "Target Audience Constraint", "Prior Condition Constraint",
        "Natural Language Process Background Information Constraint", "Markdown Process Background Information Constraint",
        "Table Background Information Constraint", "Text Background Information Constraint"
    ],
    "Style Constraint": [
        "Tone and Style Constraint", "Emotion Constraint", "Linguistic Characteristics Constraint", 
        "Multilingual Constraint"
    ],
    "Format Constraint": [
        "Output Format Constraint", "Text Pattern Constraint", "Grammar Structure Constraint",
        "Citation Constraint", "Numbering and List Constraint", "Hierarchical Structure Constraint",
        "Template Constraint"
    ],
    "Length Constraint": [
        "Word Count Limit", "Paragraph Count Limit", "Sentence Count Limit"
    ]
}

LANGUAGES = ["English", "Chinese", "Arabic", "Hindi"]

# LANGUAGE_DISTRIBUTION = {
#     "English": 101,
#     "Chinese": 75,
#     "Arabic": 62,
#     "Hindi": 62
# }

LANGUAGE_DISTRIBUTION = {
    "English": 3,
    "Chinese": 2,
    "Arabic": 2,
    "Hindi": 2
}
