"""Render the compact, source-backed figures used by the final report."""

from __future__ import annotations

import csv
from pathlib import Path

from reportlab.lib.colors import HexColor, black, white
from reportlab.pdfgen.canvas import Canvas

ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "report/figures"

NAVY = HexColor("#17365D")
TEAL = HexColor("#198C9C")
AMBER = HexColor("#D9922E")
SLATE = HexColor("#586579")
GRID = HexColor("#D9E2EA")
PALE_TEAL = HexColor("#EAF5F6")


def _series(path: Path, metric: str) -> list[tuple[int, float]]:
    with path.open(newline="") as handle:
        rows = csv.DictReader(handle)
        return [
            (int(row["epoch"]), float(row[metric])) for row in rows if row[metric]
        ]


def _label(canvas: Canvas, x: float, y: float, text: str, *, size: int = 8) -> None:
    canvas.setFillColor(SLATE)
    canvas.setFont("Helvetica", size)
    canvas.drawCentredString(x, y, text)


def render_architecture() -> None:
    output = FIGURES / "architecture.pdf"
    width, height = 650, 245
    canvas = Canvas(str(output), pagesize=(width, height))
    canvas.setFillColor(white)
    canvas.rect(0, 0, width, height, fill=1, stroke=0)
    canvas.setFillColor(NAVY)
    canvas.setFont("Helvetica-Bold", 15)
    canvas.drawString(18, 219, "One-stochastic-layer IWAE: exact layer-by-layer architecture")
    canvas.setFillColor(SLATE)
    canvas.setFont("Helvetica", 9)
    canvas.drawString(
        18, 203, "The same architecture is used for the reconstruction, DReG, and progressive DReG."
    )

    blocks = [
        (15, 89, 58, "Input x", "784 pixels", NAVY),
        (85, 89, 60, "Encoder 1", "200 tanh", TEAL),
        (157, 89, 60, "Encoder 2", "200 tanh", TEAL),
        (229, 89, 80, "Posterior heads", "mean 50 | log std 50", TEAL),
        (321, 89, 50, "Sample z", "50 dims", AMBER),
        (383, 89, 60, "Decoder 1", "200 tanh", TEAL),
        (455, 89, 60, "Decoder 2", "200 tanh", TEAL),
        (527, 89, 70, "Output logits", "784 parameters", NAVY),
    ]
    for x, y, block_width, title, subtitle, color in blocks:
        canvas.setFillColor(PALE_TEAL if color is TEAL else HexColor("#F4F7FA"))
        canvas.setStrokeColor(color)
        canvas.setLineWidth(1.3)
        canvas.roundRect(x, y, block_width, 72, 9, fill=1, stroke=1)
        canvas.setFillColor(color)
        canvas.setFont("Helvetica-Bold", 11)
        canvas.drawCentredString(x + block_width / 2, y + 43, title)
        canvas.setFillColor(SLATE)
        canvas.setFont("Helvetica", 8)
        canvas.drawCentredString(x + block_width / 2, y + 26, subtitle)
    for x in (73, 145, 217, 309, 371, 443, 515):
        canvas.setStrokeColor(SLATE)
        canvas.setLineWidth(1.4)
        canvas.line(x + 2, 125, x + 10, 125)
        canvas.line(x + 10, 125, x + 6, 129)
        canvas.line(x + 10, 125, x + 6, 121)
    canvas.setStrokeColor(TEAL)
    canvas.setLineWidth(0.9)
    canvas.line(84, 176, 309, 176)
    canvas.setFillColor(TEAL)
    canvas.setFont("Helvetica-Bold", 9)
    canvas.drawCentredString(196, 182, "Recognition model q_phi(z | x)")
    canvas.setStrokeColor(NAVY)
    canvas.line(382, 176, 597, 176)
    canvas.setFillColor(NAVY)
    canvas.drawCentredString(489, 182, "Generative model p_theta(x | z)")
    canvas.setFillColor(SLATE)
    canvas.setFont("Helvetica", 8)
    canvas.drawCentredString(
        width / 2,
        54,
        "Two 200-unit tanh layers in each network; a 50-dimensional diagonal Gaussian posterior; "
        "a factorized-Bernoulli decoder.",
    )
    canvas.showPage()
    canvas.save()


def _draw_train_validation_graph(
    *,
    output: Path,
    title: str,
    metrics: Path,
    boundaries: tuple[int, ...] = (),
) -> None:
    """Render one comparable train-versus-validation curve for a completed run."""
    width, height = 520, 300
    left, right, bottom, top = 56, 24, 48, 48
    canvas = Canvas(str(output), pagesize=(width, height))
    canvas.setFillColor(white)
    canvas.rect(0, 0, width, height, fill=1, stroke=0)
    canvas.setFillColor(NAVY)
    canvas.setFont("Helvetica-Bold", 13)
    canvas.drawString(left, height - 24, title)
    canvas.setFillColor(SLATE)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(left, height - 38, "Negative L_50 in nats; lower is better.")
    x0, x1, y0, y1 = 0, 120, 86, 132
    plot_width = width - left - right
    plot_height = height - bottom - top

    def xpos(value: float) -> float:
        return left + (value - x0) / (x1 - x0) * plot_width

    def ypos(value: float) -> float:
        return bottom + (value - y0) / (y1 - y0) * plot_height

    for value in (90, 100, 110, 120, 130):
        canvas.setStrokeColor(GRID)
        canvas.setLineWidth(0.5)
        canvas.line(left, ypos(value), width - right, ypos(value))
        _label(canvas, left - 15, ypos(value) - 3, str(value))
    for value in (0, 20, 40, 60, 80, 100, 120):
        _label(canvas, xpos(value), bottom - 15, str(value))
    canvas.setStrokeColor(black)
    canvas.setLineWidth(0.8)
    canvas.line(left, bottom, width - right, bottom)
    canvas.line(left, bottom, left, height - top)
    canvas.setFillColor(SLATE)
    canvas.setFont("Helvetica", 8)
    canvas.drawCentredString(left + plot_width / 2, 13, "Training pass")
    canvas.saveState()
    canvas.translate(14, bottom + plot_height / 2)
    canvas.rotate(90)
    canvas.drawCentredString(0, 0, "Negative bound (nats)")
    canvas.restoreState()
    series = {
        "training": (_series(metrics, "train/nll_bound"), NAVY),
        "validation": (_series(metrics, "val/nll_bound"), TEAL),
    }
    for index, (name, (points, color)) in enumerate(series.items()):
        curve = canvas.beginPath()
        for point_index, (epoch, value) in enumerate(points):
            if point_index == 0:
                curve.moveTo(xpos(epoch), ypos(value))
            else:
                curve.lineTo(xpos(epoch), ypos(value))
        canvas.setStrokeColor(color)
        canvas.setLineWidth(1.8)
        canvas.drawPath(curve)
        legend_x = left + index * 105
        canvas.setLineWidth(2.2)
        canvas.line(legend_x, height - 45, legend_x + 14, height - 45)
        canvas.setFillColor(SLATE)
        canvas.setFont("Helvetica", 8)
        canvas.drawString(legend_x + 19, height - 48, name)
    for boundary in boundaries:
        canvas.setStrokeColor(AMBER)
        canvas.setDash(2, 2)
        canvas.line(xpos(boundary), bottom, xpos(boundary), height - top)
    canvas.setDash()
    canvas.showPage()
    canvas.save()


def render_part_loss_graphs() -> None:
    _draw_train_validation_graph(
        output=FIGURES / "iwae_training_validation.pdf",
        title="IWAE reconstruction: training vs validation",
        metrics=ROOT / "outputs/iwae/lightning_logs/version_0/metrics.csv",
    )
    _draw_train_validation_graph(
        output=FIGURES / "dreg_training_validation.pdf",
        title="Fixed-K DReG: training vs validation",
        metrics=ROOT / "outputs/dreg/lightning_logs/version_0/metrics.csv",
    )
    _draw_train_validation_graph(
        output=FIGURES / "progressive_dreg_training_validation.pdf",
        title="Progressive DReG: training vs validation",
        metrics=ROOT / "outputs/fast-dreg/lightning_logs/version_0/metrics.csv",
        boundaries=(60, 100),
    )


def render_loss_comparison() -> None:
    sources = {
        "IWAE reconstruction": (ROOT / "outputs/iwae/lightning_logs/version_0/metrics.csv", NAVY),
        "DReG": (ROOT / "outputs/dreg/lightning_logs/version_0/metrics.csv", AMBER),
        "Progressive DReG": (ROOT / "outputs/fast-dreg/lightning_logs/version_0/metrics.csv", TEAL),
    }
    output = FIGURES / "loss_comparison.pdf"
    width, height = 520, 300
    left, right, bottom, top = 56, 24, 48, 48
    canvas = Canvas(str(output), pagesize=(width, height))
    canvas.setFillColor(white)
    canvas.rect(0, 0, width, height, fill=1, stroke=0)
    canvas.setFillColor(NAVY)
    canvas.setFont("Helvetica-Bold", 13)
    canvas.drawString(left, height - 24, "Validation loss across the matched 121-pass budget")
    canvas.setFillColor(SLATE)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(left, height - 38, "Shared metric: negative L_50 (lower is better).")
    x0, x1, y0, y1 = 0, 120, 86, 115
    plot_width = width - left - right
    plot_height = height - bottom - top

    def xpos(value: float) -> float:
        return left + (value - x0) / (x1 - x0) * plot_width

    def ypos(value: float) -> float:
        return bottom + (value - y0) / (y1 - y0) * plot_height

    for value in (90, 95, 100, 105, 110, 115):
        canvas.setStrokeColor(GRID)
        canvas.setLineWidth(0.5)
        canvas.line(left, ypos(value), width - right, ypos(value))
        _label(canvas, left - 15, ypos(value) - 3, str(value))
    for value in (0, 20, 40, 60, 80, 100, 120):
        _label(canvas, xpos(value), bottom - 15, str(value))
    canvas.setStrokeColor(black)
    canvas.setLineWidth(0.8)
    canvas.line(left, bottom, width - right, bottom)
    canvas.line(left, bottom, left, height - top)
    canvas.setFillColor(SLATE)
    canvas.setFont("Helvetica", 8)
    canvas.drawCentredString(left + plot_width / 2, 13, "Training pass")
    canvas.saveState()
    canvas.translate(14, bottom + plot_height / 2)
    canvas.rotate(90)
    canvas.drawCentredString(0, 0, "Validation negative bound (nats)")
    canvas.restoreState()
    for index, (name, (path, color)) in enumerate(sources.items()):
        points = _series(path, "val/nll_bound")
        curve = canvas.beginPath()
        for point_index, (epoch, value) in enumerate(points):
            if point_index == 0:
                curve.moveTo(xpos(epoch), ypos(value))
            else:
                curve.lineTo(xpos(epoch), ypos(value))
        canvas.setStrokeColor(color)
        canvas.setLineWidth(1.7)
        canvas.drawPath(curve)
        legend_x = left + index * 142
        canvas.setLineWidth(2.2)
        canvas.line(legend_x, height - 45, legend_x + 14, height - 45)
        canvas.setFillColor(SLATE)
        canvas.setFont("Helvetica", 8)
        canvas.drawString(legend_x + 19, height - 48, name)
    for boundary in (60, 100):
        canvas.setStrokeColor(TEAL)
        canvas.setDash(2, 2)
        canvas.line(xpos(boundary), bottom, xpos(boundary), height - top)
    canvas.setDash()
    canvas.showPage()
    canvas.save()


def render_sample_comparison() -> None:
    """Place identically seeded prior samples from all three models side by side."""
    output = FIGURES / "sample_comparison.pdf"
    width, height = 520, 230
    canvas = Canvas(str(output), pagesize=(width, height))
    canvas.setFillColor(white)
    canvas.rect(0, 0, width, height, fill=1, stroke=0)
    canvas.setFillColor(NAVY)
    canvas.setFont("Helvetica-Bold", 13)
    canvas.drawString(20, 204, "Prior samples from matched seed-236 checkpoints")
    canvas.setFillColor(SLATE)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(
        20,
        189,
        "Each panel uses 64 identically seeded latent draws; brightness is Bernoulli mean.",
    )
    samples = [
        ("Reconstruction (IWAE)", FIGURES / "iwae_samples_epoch120.png"),
        ("Fixed-K DReG", FIGURES / "dreg_samples_epoch114.png"),
        ("Progressive DReG", FIGURES / "progressive_dreg_samples_epoch116.png"),
    ]
    for x, (label, path) in zip((18, 187, 356), samples, strict=True):
        canvas.setFillColor(SLATE)
        canvas.setFont("Helvetica-Bold", 9)
        canvas.drawCentredString(x + 73, 169, label)
        canvas.drawImage(str(path), x, 16, width=146, height=146, mask="auto")
    canvas.showPage()
    canvas.save()


def render_training_profile() -> None:
    output = FIGURES / "progressive_schedule.pdf"
    canvas = Canvas(str(output), pagesize=(520, 245))
    canvas.setFillColor(white)
    canvas.rect(0, 0, 520, 245, fill=1, stroke=0)
    canvas.setFillColor(NAVY)
    canvas.setFont("Helvetica-Bold", 13)
    canvas.drawString(28, 219, "Progressive DReG training profile")
    canvas.setFillColor(SLATE)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(
        28, 204, "The accelerated extension retains DReG and changes only the optimisation profile."
    )
    stages = [
        (28, 110, 218, "passes 1-60", "K=5", "batch 100", "high LR"),
        (246, 110, 146, "passes 61-100", "K=10", "batch 100", "cosine decay"),
        (392, 110, 100, "passes 101-121", "K=50", "batch 100", "to 1e-4"),
    ]
    for x, y, width, title, particles, batch, lr in stages:
        canvas.setFillColor(PALE_TEAL)
        canvas.setStrokeColor(TEAL)
        canvas.roundRect(x, y, width, 66, 8, fill=1, stroke=1)
        canvas.setFillColor(NAVY)
        canvas.setFont("Helvetica-Bold", 9)
        canvas.drawCentredString(x + width / 2, y + 48, title)
        canvas.setFillColor(TEAL)
        canvas.setFont("Helvetica-Bold", 14)
        canvas.drawCentredString(x + width / 2, y + 28, particles)
        canvas.setFillColor(SLATE)
        canvas.setFont("Helvetica", 8)
        canvas.drawCentredString(x + width / 2, y + 14, f"{batch}; {lr}")
    canvas.setStrokeColor(AMBER)
    canvas.setLineWidth(2)
    canvas.line(42, 72, 480, 72)
    canvas.line(480, 72, 470, 77)
    canvas.line(480, 72, 470, 67)
    canvas.setFillColor(SLATE)
    canvas.setFont("Helvetica", 8)
    canvas.drawCentredString(
        260, 52, "Cheap early proposal learning -> intermediate refinement -> matched K=50 finish"
    )
    canvas.showPage()
    canvas.save()


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    render_architecture()
    render_part_loss_graphs()
    render_loss_comparison()
    render_sample_comparison()
    render_training_profile()


if __name__ == "__main__":
    main()
