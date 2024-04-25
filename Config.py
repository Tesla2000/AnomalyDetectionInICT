from pathlib import Path


class Config:
    root = Path(__file__).parent
    text_sections = root / "sections"
    datasets_path = root / "datasets"
    base_tex_path = root / "base.tex"
    tex_path = root / "main.tex"
    tex_parts_path = root / "tex_parts"
    image_path = root / "image_path"
    dataset_descriptions = root / "dataset_descriptions"
    image_path.mkdir(exist_ok=True)
    text_sections.mkdir(exist_ok=True)
    datasets_path.mkdir(exist_ok=True)
    dataset_descriptions.mkdir(exist_ok=True)
    random_state = 42


