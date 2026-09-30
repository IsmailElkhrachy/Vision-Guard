import time
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QImage, QPixmap
import numpy as np

app = QApplication.instance() or QApplication([])

# Simulate a 1280x720 frame
frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

latencies = []
for _ in range(100):
    t0 = time.perf_counter_ns()
    qimg = QImage(frame.data, 1280, 720, 1280 * 3, QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(qimg)
    t1 = time.perf_counter_ns()
    latencies.append((t1 - t0) / 1e6)

print(f"GUI rendering: {np.mean(latencies):.2f} ± {np.std(latencies):.2f} ms")