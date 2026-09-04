# © 2025 Kazuki Higo
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/
import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget
from PyQt6.QtGui import QImage, QPainter, QPen, QColor
from PyQt6.QtCore import Qt, QPoint


class Canvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(800, 800)  # Canvas size
        self.canvas = QImage(self.size(), QImage.Format.Format_RGB32)
        self.canvas.fill(Qt.GlobalColor.black)  # Black background

        self.history = []  # Drawing history
        self.save_state()  # Save initial state

        self.drawing = False
        self.last_point = QPoint()
        self.current_color = QColor(255, 0, 0)  # Default color (red = valley)
        self.pen_width = 2

    def set_color(self, color):
        """Change drawing color."""
        self.current_color = color

    def save_state(self):
        """Save current canvas state to history."""
        self.history.append(self.canvas.copy())

    def undo(self):
        """Undo to previous state."""
        if len(self.history) > 1:
            self.history.pop()  # Remove latest state
            self.canvas = self.history[-1]  # Restore previous state
            self.update()  # Repaint

    def reset_canvas(self):
        """Reset canvas to black."""
        self.canvas.fill(Qt.GlobalColor.black)
        self.history = []
        self.save_state()
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = True
            self.last_point = event.position().toPoint()
            self.save_state()  # Save state before drawing

    def mouseMoveEvent(self, event):
        if self.drawing:
            painter = QPainter(self.canvas)
            pen = QPen(self.current_color, self.pen_width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawLine(self.last_point, event.position().toPoint())
            self.last_point = event.position().toPoint()
            painter.end()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = False

    def paintEvent(self, event):
        """Paint event."""
        canvas_painter = QPainter(self)
        canvas_painter.drawImage(self.rect(), self.canvas, self.canvas.rect())

    def save_image(self, file_path="sketch_result.png"):
        """Save the drawing result."""
        if isinstance(file_path, str):
            try:
                self.canvas.save(file_path, "PNG")
                print(f"Image saved: {file_path}")
            except Exception as e:
                print(f"Error saving image: {e}")
        else:
            print("Error: file_path must be a string.")


class SketchTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Terrain Sketch Tool")
        self.setGeometry(100, 100, 1000, 800)

        self.canvas = Canvas(self)

        # Button setup
        red_button = QPushButton("Valley (Red)")
        green_button = QPushButton("Ridge (Green)")
        blue_button = QPushButton("Cliff (Blue)")
        undo_button = QPushButton("Undo")
        reset_button = QPushButton("Reset")
        save_button = QPushButton("Save")

        # Button actions
        red_button.clicked.connect(lambda: self.canvas.set_color(QColor(255, 0, 0)))   # Red
        green_button.clicked.connect(lambda: self.canvas.set_color(QColor(0, 255, 0))) # Green
        blue_button.clicked.connect(lambda: self.canvas.set_color(QColor(0, 0, 255)))  # Blue
        undo_button.clicked.connect(self.canvas.undo)
        reset_button.clicked.connect(self.canvas.reset_canvas)
        save_button.clicked.connect(lambda: self.canvas.save_image("sketch_result.png"))

        # Layout
        button_layout = QVBoxLayout()
        button_layout.addWidget(red_button)
        button_layout.addWidget(green_button)
        button_layout.addWidget(blue_button)
        button_layout.addWidget(undo_button)
        button_layout.addWidget(reset_button)
        button_layout.addWidget(save_button)
        button_layout.addStretch()

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.canvas)
        main_layout.addLayout(button_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SketchTool()
    window.show()
    sys.exit(app.exec())

