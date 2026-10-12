"""
Discrete-Time Sequence Operations Visualizer  (Version 2)

Same function as version 1: given x[n], a starting index n0 and a shift k,
it computes and plots
    1) Time Delay        y[n] = x[n - k]
    2) Time Advance      y[n] = x[n + k]
    3) Time Reversal     y[n] = x[-n]
    4) Composite M1      shift then fold
    5) Composite M2      fold then shift
and reports the energy of the original sequence.

What changed: the code is restructured around a small immutable Signal class
whose shift/fold methods are chained to build each operation.
"""
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
from dataclasses import dataclass

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

DEFAULT_SEQ, DEFAULT_N0, DEFAULT_K = "1, 2, 3, 4, 5", "-2", "2"


# ---------------------------------------------------------------- Signal model
@dataclass(frozen=True)
class Signal:
    """A finite discrete-time signal: sample values `x` located at indices `n`."""
    n: np.ndarray
    x: np.ndarray

    @classmethod
    def from_values(cls, values, n0):
        values = np.asarray(values, dtype=float)
        return cls(np.arange(n0, n0 + len(values)), values)

    def shift(self, d):
        """Move every sample d steps to the right (index n -> n + d)."""
        return Signal(self.n + d, self.x)

    def fold(self):
        """Mirror about n = 0 (index n -> -n)."""
        return Signal(-self.n, self.x)

    def sorted(self):
        order = np.argsort(self.n)
        return Signal(self.n[order], self.x[order])

    @property
    def energy(self):
        return float(np.sum(np.abs(self.x) ** 2))


def run_operations(sig, k):
    """Return every result signal (all sorted by n) keyed by name."""
    ops = {
        "original": sig,
        "delay": sig.shift(k),                 # x[n - k]
        "advance": sig.shift(-k),              # x[n + k]
        "reversal": sig.fold(),                # x[-n]
        "composite1": sig.shift(-k).fold(),    # shift: x[n + k], then fold
        "composite2": sig.fold().shift(k),     # fold: x[-n], then shift
    }
    return {name: s.sorted() for name, s in ops.items()}


# ------------------------------------------------------------ Text formatting
def describe(sig, title, var="y"):
    pairs = ", ".join(f"{var}[{int(i)}]={v:g}" for i, v in zip(sig.n, sig.x))
    values = ", ".join(f"{v:g}" for v in sig.x)
    return (f"{title}:\n"
            f"  {pairs}\n"
            f"  Result: {var}[n] = {{{values}}}  for n = {int(sig.n[0])}..{int(sig.n[-1])}")


def build_report(res, k, n0, count):
    orig = res["original"]
    c1, c2 = res["composite1"], res["composite2"]
    match = np.array_equal(c1.n, c2.n) and np.allclose(c1.x, c2.x)

    blocks = [
        f"Original Sequence: x[n] over n = {n0}..{n0 + count - 1}\n"
        f"  x[n] = {{{', '.join(f'{v:g}' for v in orig.x)}}}",
        describe(res["delay"], f"1) Time Delay: y[n] = x[n - {k}]"),
        describe(res["advance"], f"2) Time Advance: y[n] = x[n + {k}]"),
        describe(res["reversal"], "3) Time Reversal: y[n] = x[-n]"),
        describe(c1, f"4) Composite Method 1 (Shift then Fold): y[n] = x[-n + {k}]"),
        describe(c2, f"5) Composite Method 2 (Fold then Shift): y[n] = x[-n + {k}]"),
        f"Note: Method 1 and Method 2 give the SAME result "
        f"({'confirmed match' if match else 'MISMATCH - check inputs'}), "
        f"as expected since both compute y[n] = x[-n + {k}].",
        f"Energy of original sequence: E = sum |x[n]|^2 = {orig.energy:g}",
    ]
    return "\n\n".join(blocks)


# ------------------------------------------------------------------------ GUI
class SequenceApp:
    def __init__(self, root):
        self.root = root
        root.title("Discrete-Time Sequence Operations Visualizer")
        root.geometry("1250x850")

        self.vars = {
            "seq": tk.StringVar(value=DEFAULT_SEQ),
            "n0": tk.StringVar(value=DEFAULT_N0),
            "k": tk.StringVar(value=DEFAULT_K),
        }
        self._make_inputs()
        self._make_results_box()
        self._make_plots()
        self.compute_and_plot()  # reproduce the slides' worked example on launch

    # ---- layout
    def _make_inputs(self):
        box = ttk.LabelFrame(self.root, text="Input")
        box.pack(fill=tk.X, padx=10, pady=8)

        fields = [
            ("Sequence x[n] values (comma-separated):", "seq", 40),
            ("Starting index n0:", "n0", 8),
            ("Shift amount k:", "k", 8),
        ]
        col = 0
        for text, key, width in fields:
            ttk.Label(box, text=text).grid(row=0, column=col, padx=5, pady=5, sticky="w")
            ttk.Entry(box, width=width, textvariable=self.vars[key]).grid(
                row=0, column=col + 1, padx=5, pady=5, sticky="w")
            col += 2

        ttk.Button(box, text="Compute & Plot", command=self.compute_and_plot).grid(
            row=0, column=col, padx=10, pady=5)
        ttk.Button(box, text="Reset Example", command=self.reset_defaults).grid(
            row=0, column=col + 1, padx=5, pady=5)

    def _make_results_box(self):
        box = ttk.LabelFrame(self.root, text="Numeric Results")
        box.pack(fill=tk.X, padx=10, pady=4)
        self.results_box = ScrolledText(box, height=11, wrap=tk.WORD, font=("Consolas", 10))
        self.results_box.pack(fill=tk.X, padx=5, pady=5)

    def _make_plots(self):
        box = ttk.LabelFrame(self.root, text="Graphs")
        box.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

        self.fig = Figure(figsize=(12, 6), dpi=100)
        self.axes = self.fig.subplots(2, 3).ravel()
        self.canvas = FigureCanvasTkAgg(self.fig, master=box)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        bar = ttk.Frame(box)
        bar.pack(fill=tk.X)
        NavigationToolbar2Tk(self.canvas, bar).update()

    # ---- actions
    def reset_defaults(self):
        self.vars["seq"].set(DEFAULT_SEQ)
        self.vars["n0"].set(DEFAULT_N0)
        self.vars["k"].set(DEFAULT_K)
        self.compute_and_plot()

    def read_inputs(self):
        raw = self.vars["seq"].get().strip()
        if not raw:
            raise ValueError("Please enter at least one sequence value.")
        try:
            values = [float(t) for t in raw.split(",") if t.strip()]
        except ValueError:
            raise ValueError("Sequence values must be numbers separated by commas, "
                             "e.g. 1, 2, 3, 4, 5")
        if not values:
            raise ValueError("Please enter at least one sequence value.")

        try:
            n0 = int(self.vars["n0"].get().strip())
        except ValueError:
            raise ValueError("Starting index n0 must be an integer.")
        try:
            k = int(self.vars["k"].get().strip())
        except ValueError:
            raise ValueError("Shift amount k must be an integer.")
        return values, n0, k

    def compute_and_plot(self):
        try:
            values, n0, k = self.read_inputs()
        except ValueError as err:
            messagebox.showerror("Invalid input", str(err))
            return

        res = run_operations(Signal.from_values(values, n0), k)

        self.results_box.delete("1.0", tk.END)
        self.results_box.insert(tk.END, build_report(res, k, n0, len(values)))
        self.draw(res, k)

    def draw(self, res, k):
        panels = [
            ("original", "Original: x[n]"),
            ("delay", f"1) Time Delay: y[n] = x[n - {k}]"),
            ("advance", f"2) Time Advance: y[n] = x[n + {k}]"),
            ("reversal", "3) Time Reversal: y[n] = x[-n]"),
            ("composite1", f"4) Composite M1 (Shift->Fold): y[n]=x[-n+{k}]"),
            ("composite2", f"5) Composite M2 (Fold->Shift): y[n]=x[-n+{k}]"),
        ]
        for ax, (key, title) in zip(self.axes, panels):
            ax.clear()
            ax.stem(res[key].n, res[key].x, basefmt=" ")
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("n", fontsize=8)
            ax.set_ylabel("value", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.4)
            ax.axhline(0, color="black", lw=0.6)
            ax.axvline(0, color="black", lw=0.6)

        self.fig.tight_layout(pad=2.0)
        self.canvas.draw()


def main():
    root = tk.Tk()
    SequenceApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
