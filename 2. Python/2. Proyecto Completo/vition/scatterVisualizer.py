import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import queue


class ScatterVisualizer:
    """Visualizador de nube de puntos en el hilo principal con animación.

    Args:
        point_queue (queue.Queue): cola segura para recibir coordenadas (x, y).
        xlim (tuple): límites en X en cm.
        ylim (tuple): límites en Y en cm.
        refresh_rate (float): segundos entre cuadros.
    """

    def __init__(self, point_queue, xlim=(0, 22), ylim=(0, 10), refresh_rate=0.05):
        self.point_queue = point_queue
        self.xlim = xlim
        self.ylim = ylim
        self.refresh_rate = refresh_rate
        self.coords = []

    def start(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(*self.xlim)
        self.ax.set_ylim(*self.ylim)
        self.ax.set_xlabel("X (cm)")
        self.ax.set_ylabel("Y (cm)")
        self.ax.set_title("Nube de puntos – coordenadas visión")
        self.scatter = self.ax.scatter([], [], s=10, c='blue', alpha=0.6)

        self.ani = animation.FuncAnimation(
            self.fig,
            self._update,
            interval=int(self.refresh_rate * 1000),
            blit=False
        )

        plt.ion()
        plt.show(block=True)

    def _update(self, frame):
        updated = False
        while not self.point_queue.empty():
            try:
                point = self.point_queue.get_nowait()
                self.coords.append(point)
                updated = True
            except queue.Empty:
                break

        if updated and self.coords:
            xs, ys = zip(*self.coords)
            self.scatter.set_offsets(np.c_[xs, ys])

        return self.scatter,