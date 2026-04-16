CLASS_NAMES = [
    "pothole",
    "road_crack",
    "garbage",
    "graffiti",
    "fallen_tree",
    "damaged_sign",
    "damaged_electric_pole",
]

LABEL_VI = {
    'pothole': 'ổ gà',
    'road_crack': 'đường nứt',
    'garbage': 'rác thải',
    'graffiti': 'vẽ bậy',
    'fallen_tree': 'cây đổ',
    'damaged_sign': 'biển báo hư hỏng',
    'damaged_electric_pole': 'cột điện hư hỏng',
}

# Mapping index -> label (cho YOLO)
ID_TO_LABEL = {i: label for i, label in enumerate(CLASS_NAMES)}
LABEL_TO_ID = {label: i for i, label in enumerate(CLASS_NAMES)}