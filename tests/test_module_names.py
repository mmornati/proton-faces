import sys
from pathlib import Path

APP_SRC = Path(__file__).resolve().parent.parent / "app" / "src"


def test_no_app_module_shadows_stdlib():
    # app/src is sys.path[0] in the image (WORKDIR + `python main.py`), so a
    # module named like a stdlib one hides it: compression.py broke `import gzip`
    # on Python 3.14, which added the `compression` package.
    names = {p.stem for p in APP_SRC.glob("*.py")} | {p.name for p in APP_SRC.iterdir() if (p / "__init__.py").exists()}
    assert not names & set(sys.stdlib_module_names)
