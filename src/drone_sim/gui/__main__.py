# src/drone_sim/gui/__main__.py
import os
os.environ["QT_API"] = "pyside6"  # MUST precede all matplotlib/PySide6 imports

import sys
from PySide6.QtWidgets import QApplication
from drone_sim.gui.main_window import MainWindow


def main() -> None:
    # Use QApplication.instance() guard — never assume no QApplication exists
    # (locked decision: some IDEs already own the event loop)
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
