from pathlib import Path


class Config:
    root = Path(__file__).parent
    datasets_path = root / 'datasets'
    base_tex_path = root / 'base.tex'
    tex_path = root / 'main.tex'
    tex_parts_path = root / 'tex_parts'
    image_path = root / 'image_path'
    image_path.mkdir(exist_ok=True)
