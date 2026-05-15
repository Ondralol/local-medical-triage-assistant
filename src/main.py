import sys
from pathlib import Path

from streamlit.web import cli as stcli


def main():
    app_path = Path(__file__).resolve().parent / "ui" / "streamlit_app.py"
    sys.argv = ["streamlit", "run", str(app_path)]
    return stcli.main()


if __name__ == "__main__":
    raise SystemExit(main())
