"""Render the completed IWAE learning curve as a vector PDF."""

from __future__ import annotations

import csv
from pathlib import Path

from reportlab.lib.colors import Color, HexColor, black, white
from reportlab.pdfgen.canvas import Canvas

ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "outputs/iwae/lightning_logs/version_0/metrics.csv"
OUTPUT = ROOT / "report/figures/iwae_learning_curve.pdf"


def load_series() -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    with METRICS.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    train = [
        (int(row["epoch"]), float(row["train/nll_bound"])) for row in rows if row["train/nll_bound"]
    ]
    validation = [
        (int(row["epoch"]), float(row["val/nll_bound"])) for row in rows if row["val/nll_bound"]
    ]
    return train, validation


def render() -> None:
    train, validation = load_series()
    width, height = 500, 310
    left, right, bottom, top = 58, 20, 48, 42
    plot_width = width - left - right
    plot_height = height - bottom - top
    x_min, x_max = 0, 120
    y_min, y_max = 88, 132

    def x_position(epoch: float) -> float:
        return left + (epoch - x_min) / (x_max - x_min) * plot_width

    def y_position(value: float) -> float:
        return bottom + (value - y_min) / (y_max - y_min) * plot_height

    canvas = Canvas(str(OUTPUT), pagesize=(width, height))
    canvas.setFillColor(white)
    canvas.rect(0, 0, width, height, fill=1, stroke=0)

    canvas.setFont("Helvetica-Bold", 12)
    canvas.setFillColor(black)
    canvas.drawString(left, height - 24, "IWAE optimization over 121 MNIST passes")

    canvas.setStrokeColor(HexColor("#D8DEE9"))
    canvas.setLineWidth(0.5)
    for value in range(90, 131, 10):
        y = y_position(value)
        canvas.line(left, y, width - right, y)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(HexColor("#4C566A"))
        canvas.drawRightString(left - 7, y - 3, str(value))

    for epoch in (0, 20, 40, 60, 80, 100, 120):
        x = x_position(epoch)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(HexColor("#4C566A"))
        canvas.drawCentredString(x, bottom - 15, str(epoch))

    for milestone in (1, 4, 13, 40):
        x = x_position(milestone)
        canvas.setStrokeColor(Color(0.55, 0.57, 0.62, alpha=0.7))
        canvas.setDash(2, 2)
        canvas.line(x, bottom, x, height - top)
    canvas.setDash()

    canvas.setStrokeColor(black)
    canvas.setLineWidth(0.8)
    canvas.line(left, bottom, width - right, bottom)
    canvas.line(left, bottom, left, height - top)

    def draw_series(points: list[tuple[int, float]], color: str) -> None:
        path = canvas.beginPath()
        for index, (epoch, value) in enumerate(points):
            x, y = x_position(epoch), y_position(value)
            if index == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        canvas.setStrokeColor(HexColor(color))
        canvas.setLineWidth(1.7)
        canvas.drawPath(path)

    draw_series(train, "#5E81AC")
    draw_series(validation, "#BF616A")

    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(black)
    canvas.drawCentredString(left + plot_width / 2, 14, "Training pass (epoch)")
    canvas.saveState()
    canvas.translate(14, bottom + plot_height / 2)
    canvas.rotate(90)
    canvas.drawCentredString(0, 0, "Negative bound (nats; lower is better)")
    canvas.restoreState()

    legend_y = height - 25
    canvas.setStrokeColor(HexColor("#5E81AC"))
    canvas.setLineWidth(2)
    canvas.line(width - 164, legend_y, width - 148, legend_y)
    canvas.setFillColor(black)
    canvas.drawString(width - 143, legend_y - 3, "train")
    canvas.setStrokeColor(HexColor("#BF616A"))
    canvas.line(width - 105, legend_y, width - 89, legend_y)
    canvas.setFillColor(black)
    canvas.drawString(width - 84, legend_y - 3, "validation")

    canvas.showPage()
    canvas.save()


if __name__ == "__main__":
    render()
