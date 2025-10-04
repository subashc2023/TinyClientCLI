import asyncio
import importlib.util
import sys
from pathlib import Path


def _load_app_main():
    module_path = Path(__file__).parent / "src" / "main.py"
    src_dir = str(module_path.parent)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    spec = importlib.util.spec_from_file_location("tinyclient_main", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    if not hasattr(module, "main"):
        raise AttributeError("'main' callable not found in src/main.py")
    return getattr(module, "main")


def main() -> None:
    app_main = _load_app_main()
    asyncio.run(app_main())


if __name__ == "__main__":
    main()
