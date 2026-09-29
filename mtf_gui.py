"""Fluorescence USAF 1951 contrast-transfer analysis GUI.

The application loads fluorescence target images, extracts intensity profiles
from user-defined USAF regions, calculates Michelson contrast transfer (CTF),
and exports publication-ready figures and measurements.

Packed uint32 RGB32 TIFF images encoded as 0x00RRGGBB are supported.
"""
from __future__ import annotations

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from dataclasses import dataclass

import numpy as np
import pandas as pd
from PIL import Image
import tifffile
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import RectangleSelector
from matplotlib.patches import Rectangle
from matplotlib import gridspec


APP_TITLE = "MUSE GUI_RRK LAB | Fluorescence USAF MTF Analyzer"
ORIENTATION_X = "Vertical bars (X response)"
ORIENTATION_Y = "Horizontal bars (Y response)"


@dataclass
class ROIResult:
    roi_id: int
    x0: int
    y0: int
    x1: int
    y1: int
    orientation: str
    group: int
    element: int
    channel: str
    rotation: int
    frequency_lpmm: float
    period_um: float
    line_width_um: float
    measured_frequency_lpmm: float | None
    ctf: float
    imax: float
    imin: float
    n_peaks: int
    n_valleys: int
    profile_x: np.ndarray
    profile_raw: np.ndarray
    profile_smooth: np.ndarray
    peaks: np.ndarray
    valleys: np.ndarray


def usa_frequency(group: int, element: int) -> float:
    return 2.0 ** (group + (element - 1) / 6.0)


def decode_image(path: str):
    ext = os.path.splitext(path)[1].lower()
    arr = tifffile.imread(path) if ext in (".tif", ".tiff") else np.asarray(Image.open(path))

    if arr.ndim == 2 and arr.dtype == np.uint32 and np.max(arr) > 255:
        r = ((arr >> 16) & 255).astype(np.uint8)
        g = ((arr >> 8) & 255).astype(np.uint8)
        b = (arr & 255).astype(np.uint8)
        rgb = np.dstack([r, g, b])
        gray = 0.2126 * r + 0.7152 * g + 0.0722 * b
        channels = {"Gray": gray, "Red": r, "Green": g, "Blue": b}
        info = f"Packed RGB32 decoded | {arr.shape[1]} × {arr.shape[0]} | uint32 → RGB8"
        return channels, info

    if arr.ndim == 3 and arr.shape[2] >= 3:
        rgb = arr[..., :3]
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        gray = 0.2126 * r.astype(float) + 0.7152 * g.astype(float) + 0.0722 * b.astype(float)
        channels = {"Gray": gray, "Red": r, "Green": g, "Blue": b}
        info = f"RGB image | {arr.shape[1]} × {arr.shape[0]} | {arr.dtype}"
        return channels, info

    if arr.ndim == 2:
        channels = {"Gray": arr, "Red": arr, "Green": arr, "Blue": arr}
        info = f"Grayscale image | {arr.shape[1]} × {arr.shape[0]} | {arr.dtype}"
        return channels, info

    raise ValueError(f"Unsupported image shape: {arr.shape}")


def rotate_array(img: np.ndarray, deg: int) -> np.ndarray:
    k = {0: 0, 90: 1, 180: 2, 270: 3}.get(int(deg), 0)
    return np.rot90(img, k=k)


def robust_limits(img: np.ndarray):
    x = np.asarray(img, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0, 1
    lo, hi = np.percentile(x, [0.5, 99.5])
    if hi <= lo:
        lo, hi = float(np.min(x)), float(np.max(x) + 1.0)
    return float(lo), float(hi)


def _fallback_extrema(y: np.ndarray, kind: str):
    if y.size < 3:
        return np.array([], dtype=int)
    dy = np.diff(y)
    s = np.sign(dy)
    s[s == 0] = 1
    d2 = np.diff(s)
    if kind == "max":
        return (np.where(d2 < 0)[0] + 1).astype(int)
    return (np.where(d2 > 0)[0] + 1).astype(int)


def analyze_profile(crop: np.ndarray, orientation: str, n_average: int,
                    smooth_sigma: float, frequency_lpmm: float,
                    pixel_size_um: float | None):
    crop = np.asarray(crop, dtype=float)
    if crop.size == 0:
        raise ValueError("Empty ROI.")

    profile = np.mean(crop, axis=0) if orientation.startswith("Vertical") else np.mean(crop, axis=1)
    if profile.size < 8:
        raise ValueError("ROI is too small. Select a larger ROI around one complete USAF element.")

    sigma = max(0.0, float(smooth_sigma))
    smooth = gaussian_filter1d(profile, sigma) if sigma > 0 else profile.copy()
    dyn = float(np.max(smooth) - np.min(smooth))
    if dyn <= 0:
        raise ValueError("The ROI has no measurable modulation.")

    if pixel_size_um and pixel_size_um > 0:
        period_um = 1000.0 / frequency_lpmm
        period_px = max(period_um / pixel_size_um, 1.0)
        min_distance = max(2, int(round(0.4 * period_px)))
    else:
        min_distance = max(2, profile.size // 20)

    prom = max(1e-12, 0.03 * dyn)
    peaks, pprops = find_peaks(smooth, prominence=prom, distance=min_distance)
    valleys, vprops = find_peaks(-smooth, prominence=prom, distance=min_distance)

    if len(peaks) < 2:
        peaks = _fallback_extrema(smooth, "max")
        pprops = {"prominences": smooth[peaks] - np.min(smooth) if len(peaks) else np.array([])}
    if len(valleys) < 1:
        valleys = _fallback_extrema(smooth, "min")
        vprops = {"prominences": np.max(smooth) - smooth[valleys] if len(valleys) else np.array([])}

    if len(peaks) == 0 or len(valleys) == 0:
        raise ValueError("Could not identify both peaks and valleys. Try a tighter ROI, lower smoothing, or the raw higher-resolution image.")

    n_average = max(1, int(n_average))
    peak_prom = np.asarray(pprops.get("prominences", np.ones(len(peaks))))
    valley_prom = np.asarray(vprops.get("prominences", np.ones(len(valleys))))

    use_peaks = peaks[np.argsort(peak_prom)[-min(n_average, len(peaks)):]]
    use_valleys = valleys[np.argsort(valley_prom)[-min(max(1, n_average - 1), len(valleys)):]]
    use_peaks = np.sort(use_peaks)
    use_valleys = np.sort(use_valleys)

    imax = float(np.mean(smooth[use_peaks]))
    imin = float(np.mean(smooth[use_valleys]))
    denom = imax + imin
    ctf = float((imax - imin) / denom) if denom != 0 else np.nan
    ctf = float(np.clip(ctf, 0.0, 1.0)) if np.isfinite(ctf) else np.nan

    measured_frequency = None
    if pixel_size_um and pixel_size_um > 0 and len(use_peaks) >= 2:
        spacings = np.diff(use_peaks.astype(float))
        if np.mean(spacings) > 0:
            measured_period_um = np.mean(spacings) * pixel_size_um
            measured_frequency = 1000.0 / measured_period_um

    x = np.arange(profile.size, dtype=float)
    if pixel_size_um and pixel_size_um > 0:
        x = x * pixel_size_um

    return dict(
        profile_x=x,
        profile_raw=profile,
        profile_smooth=smooth,
        peaks=use_peaks,
        valleys=use_valleys,
        imax=imax,
        imin=imin,
        ctf=ctf,
        measured_frequency_lpmm=measured_frequency,
    )


class MTFAnalyzerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1650x960")
        self.root.minsize(1320, 820)

        self.image_path = None
        self.original_channels = None
        self.rotated_channels = None
        self.results: list[ROIResult] = []
        self.selected_roi_id: int | None = None
        self.next_roi_id = 1
        self.selector = None
        self._tree_event_lock = False

        self._build_ui()

    def _build_ui(self):
        try:
            ttk.Style().theme_use("clam")
        except Exception:
            pass

        pw = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        pw.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(pw, padding=10)
        right = ttk.Frame(pw, padding=6)
        pw.add(left, weight=0)
        pw.add(right, weight=1)

        ttk.Label(left, text="Fluorescence USAF MTF Analysis", font=("Segoe UI", 14, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

        ttk.Button(left, text="Load raw image", command=self.load_image).grid(row=1, column=0, columnspan=2, sticky="ew", pady=3)
        self.file_label = ttk.Label(left, text="No image loaded", wraplength=330)
        self.file_label.grid(row=2, column=0, columnspan=2, sticky="w")
        self.info_label = ttk.Label(left, text="", wraplength=330)
        self.info_label.grid(row=3, column=0, columnspan=2, sticky="w", pady=(2, 10))

        ttk.Label(left, text="Target").grid(row=4, column=0, sticky="w")
        self.target_var = tk.StringVar(value="USAF 1951 fluorescent")
        ttk.Combobox(left, textvariable=self.target_var, state="readonly", values=["USAF 1951 fluorescent"]).grid(row=4, column=1, sticky="ew")

        ttk.Label(left, text="Analysis channel").grid(row=5, column=0, sticky="w")
        self.channel_var = tk.StringVar(value="Gray")
        cb = ttk.Combobox(left, textvariable=self.channel_var, state="readonly", values=["Gray", "Red", "Green", "Blue"])
        cb.grid(row=5, column=1, sticky="ew")
        cb.bind("<<ComboboxSelected>>", lambda e: self.refresh_image())

        ttk.Label(left, text="Rotation").grid(row=6, column=0, sticky="w")
        self.rotation_var = tk.IntVar(value=0)
        rot = ttk.Combobox(left, textvariable=self.rotation_var, state="readonly", values=[0, 90, 180, 270])
        rot.grid(row=6, column=1, sticky="ew")
        rot.bind("<<ComboboxSelected>>", lambda e: self.apply_rotation())

        ttk.Label(left, text="Bar orientation").grid(row=7, column=0, sticky="w")
        self.orientation_var = tk.StringVar(value=ORIENTATION_X)
        ttk.Combobox(left, textvariable=self.orientation_var, state="readonly",
                     values=[ORIENTATION_X, ORIENTATION_Y]).grid(row=7, column=1, sticky="ew")

        ttk.Label(left, text="USAF group").grid(row=8, column=0, sticky="w")
        self.group_var = tk.IntVar(value=7)
        ttk.Spinbox(left, from_=-2, to=9, textvariable=self.group_var, width=10, command=self.update_usaf_label).grid(row=8, column=1, sticky="w")

        ttk.Label(left, text="USAF element").grid(row=9, column=0, sticky="w")
        self.element_var = tk.IntVar(value=4)
        ttk.Spinbox(left, from_=1, to=6, textvariable=self.element_var, width=10, command=self.update_usaf_label).grid(row=9, column=1, sticky="w")

        ttk.Label(left, text="Pixel size (µm/pixel)").grid(row=10, column=0, sticky="w")
        self.pixel_var = tk.StringVar(value="")
        ttk.Entry(left, textvariable=self.pixel_var).grid(row=10, column=1, sticky="ew")

        ttk.Label(left, text="Peaks to average").grid(row=11, column=0, sticky="w")
        self.npeaks_var = tk.IntVar(value=3)
        ttk.Spinbox(left, from_=1, to=10, textvariable=self.npeaks_var, width=10).grid(row=11, column=1, sticky="w")

        ttk.Label(left, text="Profile smoothing σ (px)").grid(row=12, column=0, sticky="w")
        self.sigma_var = tk.DoubleVar(value=0.5)
        ttk.Entry(left, textvariable=self.sigma_var).grid(row=12, column=1, sticky="ew")

        self.usaf_label = ttk.Label(left, text="")
        self.usaf_label.grid(row=13, column=0, columnspan=2, sticky="w", pady=(6, 8))
        ttk.Button(left, text="Update USAF values", command=self.update_usaf_label).grid(row=14, column=0, columnspan=2, sticky="ew", pady=2)

        ttk.Separator(left).grid(row=15, column=0, columnspan=2, sticky="ew", pady=8)
        ttk.Button(left, text="Add ROI", command=self.enable_roi).grid(row=16, column=0, sticky="ew", padx=(0, 3), pady=2)
        ttk.Button(left, text="Delete selected ROI", command=self.delete_selected).grid(row=16, column=1, sticky="ew", pady=2)
        ttk.Button(left, text="Deselect ROI", command=self.deselect_roi).grid(row=17, column=0, sticky="ew", padx=(0, 3), pady=2)
        ttk.Button(left, text="Clear all ROIs", command=self.clear_all).grid(row=17, column=1, sticky="ew", pady=2)

        ttk.Button(left, text="Export measurements CSV", command=self.export_csv).grid(row=18, column=0, columnspan=2, sticky="ew", pady=(10, 2))
        ttk.Button(left, text="Save scientific figure", command=self.save_publication_figure).grid(row=19, column=0, columnspan=2, sticky="ew", pady=2)
        ttk.Button(left, text="Save current plots", command=self.save_current_plots).grid(row=20, column=0, columnspan=2, sticky="ew", pady=2)

        note = (
            "Recommended: use the native raw image, Peaks = 3 for one USAF triad,\n"
            "and σ = 0.3–0.7 px. The scientific figure includes the selected\n"
            "ROI region and all ROI regions in the overview panel."
        )
        ttk.Label(left, text=note, wraplength=330).grid(row=21, column=0, columnspan=2, sticky="w", pady=(10, 0))
        left.columnconfigure(1, weight=1)
        self.update_usaf_label()

        self.tabs = ttk.Notebook(right)
        self.tabs.pack(fill=tk.BOTH, expand=True)
        self.tab_image = ttk.Frame(self.tabs)
        self.tab_profile = ttk.Frame(self.tabs)
        self.tab_transfer = ttk.Frame(self.tabs)
        self.tab_table = ttk.Frame(self.tabs)
        self.tabs.add(self.tab_image, text="Image / ROI")
        self.tabs.add(self.tab_profile, text="Peak profile")
        self.tabs.add(self.tab_transfer, text="Transfer curves")
        self.tabs.add(self.tab_table, text="Measurements")

        self.fig_image = Figure(figsize=(8, 6), dpi=100)
        self.ax_image = self.fig_image.add_subplot(111)
        self.ax_image.set_axis_off()
        self.canvas_image = FigureCanvasTkAgg(self.fig_image, master=self.tab_image)
        self.canvas_image.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas_image, self.tab_image).update()
        self.canvas_image.mpl_connect("pick_event", self.on_pick)

        self.fig_profile = Figure(figsize=(8, 5), dpi=100)
        self.ax_profile = self.fig_profile.add_subplot(111)
        self.canvas_profile = FigureCanvasTkAgg(self.fig_profile, master=self.tab_profile)
        self.canvas_profile.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas_profile, self.tab_profile).update()

        self.fig_transfer = Figure(figsize=(10, 7), dpi=100)
        self.ax_ctf = self.fig_transfer.add_subplot(211)
        self.ax_norm = self.fig_transfer.add_subplot(212)
        self.canvas_transfer = FigureCanvasTkAgg(self.fig_transfer, master=self.tab_transfer)
        self.canvas_transfer.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas_transfer, self.tab_transfer).update()

        cols = ("ID", "Orientation", "G", "E", "lp/mm", "Line µm", "CTF", "Norm.", "Channel", "Rotation")
        self.tree = ttk.Treeview(self.tab_table, columns=cols, show="headings", selectmode="browse")
        widths = {"ID": 50, "Orientation": 160, "G": 40, "E": 40, "lp/mm": 90, "Line µm": 85,
                  "CTF": 70, "Norm.": 70, "Channel": 70, "Rotation": 65}
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=widths[c], anchor="center")
        self.tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        sy = ttk.Scrollbar(self.tab_table, orient="vertical", command=self.tree.yview)
        sy.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=sy.set)
        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)

        self.clear_profile_plot()
        self.plot_transfer_curves()

    def current_image(self):
        if self.rotated_channels is None:
            return None
        return np.asarray(self.rotated_channels[self.channel_var.get()])

    def update_usaf_label(self):
        try:
            g = int(self.group_var.get())
            e = int(self.element_var.get())
            f = usa_frequency(g, e)
            self.usaf_label.configure(text=f"G{g} E{e}: {f:.2f} lp/mm | period {1000.0/f:.3f} µm | line width {500.0/f:.3f} µm")
        except Exception:
            self.usaf_label.configure(text="Invalid USAF values")

    def _pixel_size(self):
        txt = self.pixel_var.get().strip()
        if not txt:
            return None
        try:
            val = float(txt)
            return val if val > 0 else None
        except ValueError:
            return None

    def load_image(self):
        path = filedialog.askopenfilename(
            title="Select fluorescence USAF image",
            filetypes=[("Images", "*.tif *.tiff *.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")])
        if not path:
            return
        try:
            channels, info = decode_image(path)
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))
            return
        self.image_path = path
        self.original_channels = channels
        self.file_label.configure(text=os.path.basename(path))
        self.info_label.configure(text=info)
        self.results = []
        self.selected_roi_id = None
        self.next_roi_id = 1
        self.apply_rotation(reset_results=False)

    def apply_rotation(self, reset_results=True):
        if self.original_channels is None:
            return
        deg = int(self.rotation_var.get())
        self.rotated_channels = {k: rotate_array(np.asarray(v), deg) for k, v in self.original_channels.items()}
        if reset_results:
            self.results = []
            self.selected_roi_id = None
            self.next_roi_id = 1
        self.refresh_results()
        self.refresh_image()
        self.clear_profile_plot()
        self.plot_transfer_curves()

    def refresh_image(self):
        self.ax_image.clear()
        self.ax_image.set_axis_off()
        img = self.current_image()
        if img is None:
            self.canvas_image.draw_idle()
            return
        lo, hi = robust_limits(img)
        self.ax_image.imshow(img, cmap="gray", vmin=lo, vmax=hi)
        self.ax_image.set_title(f"{self.channel_var.get()} channel | rotation {int(self.rotation_var.get())}°")
        for r in self.results:
            selected = (r.roi_id == self.selected_roi_id)
            color = "cyan" if selected else "yellow"
            lw = 2.8 if selected else 1.6
            rect = Rectangle((r.x0, r.y0), r.x1-r.x0, r.y1-r.y0, fill=False, edgecolor=color, linewidth=lw)
            rect.set_picker(True)
            rect._roi_id = r.roi_id
            self.ax_image.add_patch(rect)
            txt = self.ax_image.text(r.x0, max(0, r.y0-4), f"ROI {r.roi_id}", fontsize=8,
                                     bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))
            txt.set_picker(True)
            txt._roi_id = r.roi_id
        self.fig_image.tight_layout()
        self.canvas_image.draw_idle()

    def clear_profile_plot(self):
        self.ax_profile.clear()
        self.ax_profile.text(0.5, 0.5, "Select an ROI to view the averaged line profile.", transform=self.ax_profile.transAxes,
                             ha="center", va="center")
        self.ax_profile.set_axis_off()
        self.canvas_profile.draw_idle()

    def enable_roi(self):
        if self.current_image() is None:
            messagebox.showinfo("Load image", "Load an image first.")
            return
        if self.selector is not None:
            self.selector.set_active(False)
        self.selector = RectangleSelector(self.ax_image, self.on_roi_selected,
                                          useblit=True, button=[1], minspanx=5, minspany=5,
                                          spancoords="pixels", interactive=False)
        self.selector.set_active(True)
        self.tabs.select(self.tab_image)

    def on_roi_selected(self, eclick, erelease):
        img = self.current_image()
        if img is None or eclick.xdata is None or erelease.xdata is None:
            return
        x0, x1 = sorted([int(round(eclick.xdata)), int(round(erelease.xdata))])
        y0, y1 = sorted([int(round(eclick.ydata)), int(round(erelease.ydata))])
        x0, x1 = max(0, x0), min(img.shape[1], x1)
        y0, y1 = max(0, y0), min(img.shape[0], y1)
        if x1 - x0 < 5 or y1 - y0 < 5:
            return

        try:
            group = int(self.group_var.get())
            element = int(self.element_var.get())
            f = usa_frequency(group, element)
            metrics = analyze_profile(img[y0:y1, x0:x1], self.orientation_var.get(), int(self.npeaks_var.get()),
                                      float(self.sigma_var.get()), f, self._pixel_size())
        except Exception as exc:
            messagebox.showerror("ROI analysis", str(exc))
            return

        rr = ROIResult(
            roi_id=self.next_roi_id,
            x0=x0, y0=y0, x1=x1, y1=y1,
            orientation=self.orientation_var.get(),
            group=group, element=element,
            channel=self.channel_var.get(),
            rotation=int(self.rotation_var.get()),
            frequency_lpmm=f,
            period_um=1000.0 / f,
            line_width_um=500.0 / f,
            measured_frequency_lpmm=metrics["measured_frequency_lpmm"],
            ctf=metrics["ctf"],
            imax=metrics["imax"],
            imin=metrics["imin"],
            n_peaks=len(metrics["peaks"]),
            n_valleys=len(metrics["valleys"]),
            profile_x=metrics["profile_x"],
            profile_raw=metrics["profile_raw"],
            profile_smooth=metrics["profile_smooth"],
            peaks=metrics["peaks"],
            valleys=metrics["valleys"],
        )
        self.results.append(rr)
        self.selected_roi_id = rr.roi_id
        self.next_roi_id += 1
        if self.selector is not None:
            self.selector.set_active(False)
        self.refresh_results()
        self.refresh_image()
        self.plot_profile(rr)
        self.plot_transfer_curves()
        self.tabs.select(self.tab_profile)

    def normalized_map(self):
        rel = {}
        for ori in [ORIENTATION_X, ORIENTATION_Y]:
            rows = [r for r in self.results if r.orientation == ori and np.isfinite(r.ctf)]
            if not rows:
                continue
            rows = sorted(rows, key=lambda r: r.frequency_lpmm)
            ref = rows[0].ctf if rows[0].ctf > 0 else max(rr.ctf for rr in rows)
            for rr in rows:
                rel[rr.roi_id] = rr.ctf / ref if ref > 0 else np.nan
        return rel

    def refresh_results(self):
        """Refresh measurement table without triggering recursive selection callbacks."""
        self._tree_event_lock = True
        try:
            for item in self.tree.get_children():
                self.tree.delete(item)
            rel = self.normalized_map()
            for r in self.results:
                orient = "Vertical (X)" if r.orientation.startswith("Vertical") else "Horizontal (Y)"
                self.tree.insert("", "end", iid=str(r.roi_id), values=(
                    r.roi_id, orient, r.group, r.element, f"{r.frequency_lpmm:.2f}", f"{r.line_width_um:.3f}",
                    f"{r.ctf:.4f}", f"{rel.get(r.roi_id, np.nan):.4f}", r.channel, f"{r.rotation}°"
                ))
            if self.selected_roi_id is not None and str(self.selected_roi_id) in self.tree.get_children():
                self.tree.selection_set(str(self.selected_roi_id))
                self.tree.focus(str(self.selected_roi_id))
        finally:
            # Delay unlock until Tk has processed the selection redraw.
            self.root.after_idle(self._unlock_tree_event)

    def _unlock_tree_event(self):
        self._tree_event_lock = False

    def get_roi(self, rid: int | None):
        if rid is None:
            return None
        for r in self.results:
            if r.roi_id == rid:
                return r
        return None

    def plot_profile(self, r: ROIResult):
        self.ax_profile.clear()
        unit = "µm" if self._pixel_size() else "pixels"
        self.ax_profile.plot(r.profile_x, r.profile_raw, alpha=0.4, linewidth=1.2, label="Raw")
        self.ax_profile.plot(r.profile_x, r.profile_smooth, linewidth=2.0, label="Smoothed")
        self.ax_profile.scatter(r.profile_x[r.peaks], r.profile_smooth[r.peaks], s=35, label="Peaks", zorder=4)
        self.ax_profile.scatter(r.profile_x[r.valleys], r.profile_smooth[r.valleys], s=35, label="Valleys", zorder=4)
        self.ax_profile.set_xlabel(f"Distance ({unit})")
        self.ax_profile.set_ylabel("Intensity (a.u.)")
        self.ax_profile.set_title(f"ROI {r.roi_id} | G{r.group} E{r.element} | {r.frequency_lpmm:.2f} lp/mm | CTF = {r.ctf:.3f}")
        self.ax_profile.grid(True, alpha=0.25)
        self.ax_profile.legend(frameon=False)
        self.fig_profile.tight_layout()
        self.canvas_profile.draw_idle()

    def plot_transfer_curves(self):
        self.ax_ctf.clear()
        self.ax_norm.clear()
        rel = self.normalized_map()
        any_rows = False
        selected = self.get_roi(self.selected_roi_id)

        for ori, label in [(ORIENTATION_X, "Vertical — X response"),
                           (ORIENTATION_Y, "Horizontal — Y response")]:
            rows = [r for r in self.results if r.orientation == ori and np.isfinite(r.ctf)]
            if not rows:
                continue
            any_rows = True
            rows = sorted(rows, key=lambda r: r.frequency_lpmm)
            freq = np.array([r.frequency_lpmm for r in rows])
            ctf = np.array([r.ctf for r in rows])
            norm = np.array([rel.get(r.roi_id, np.nan) for r in rows])
            self.ax_ctf.plot(freq, ctf, marker="o", linewidth=2, label=label)
            self.ax_norm.plot(freq, norm, marker="o", linewidth=2, label=label)

        if selected is not None and np.isfinite(selected.ctf):
            self.ax_ctf.scatter([selected.frequency_lpmm], [selected.ctf], s=110, marker="*", zorder=6, label="Selected ROI")
            rn = rel.get(selected.roi_id, np.nan)
            if np.isfinite(rn):
                self.ax_norm.scatter([selected.frequency_lpmm], [rn], s=110, marker="*", zorder=6, label="Selected ROI")

        self.ax_ctf.set_title("Absolute Michelson CTF")
        self.ax_ctf.set_xlabel("Spatial frequency (lp/mm)")
        self.ax_ctf.set_ylabel("CTF")
        self.ax_ctf.set_ylim(0, 1.05)
        self.ax_ctf.grid(True, alpha=0.3)
        if any_rows:
            self.ax_ctf.legend(frameon=False)

        self.ax_norm.set_title("Normalized transfer curve")
        self.ax_norm.set_xlabel("Spatial frequency (lp/mm)")
        self.ax_norm.set_ylabel("Normalized CTF")
        self.ax_norm.set_ylim(0, 1.1)
        self.ax_norm.grid(True, alpha=0.3)
        if any_rows:
            self.ax_norm.legend(frameon=False)
        else:
            self.ax_ctf.text(0.5, 0.5, "Add multiple ROIs to build transfer curves.", transform=self.ax_ctf.transAxes,
                             ha="center", va="center")
            self.ax_norm.text(0.5, 0.5, "Normalized curve will appear here.", transform=self.ax_norm.transAxes,
                              ha="center", va="center")
        self.fig_transfer.tight_layout()
        self.canvas_transfer.draw_idle()

    def on_pick(self, event):
        rid = getattr(event.artist, "_roi_id", None)
        if rid is None:
            return
        if self.selected_roi_id == rid:
            self.deselect_roi()
        else:
            self.select_roi(rid)

    def select_roi(self, rid: int, from_table: bool = False):
        r = self.get_roi(rid)
        if r is None:
            return
        self.selected_roi_id = rid
        self.refresh_image()
        self.plot_profile(r)
        self.plot_transfer_curves()
        if not from_table:
            self._tree_event_lock = True
            try:
                if str(rid) in self.tree.get_children():
                    self.tree.selection_set(str(rid))
                    self.tree.focus(str(rid))
                    self.tree.see(str(rid))
            finally:
                self.root.after_idle(self._unlock_tree_event)
        self.tabs.select(self.tab_profile)

    def deselect_roi(self):
        self.selected_roi_id = None
        self._tree_event_lock = True
        try:
            sel = self.tree.selection()
            if sel:
                self.tree.selection_remove(*sel)
        finally:
            self.root.after_idle(self._unlock_tree_event)
        self.refresh_image()
        self.clear_profile_plot()
        self.plot_transfer_curves()

    def on_tree_select(self, _event=None):
        if self._tree_event_lock:
            return
        sel = self.tree.selection()
        if not sel:
            return
        try:
            rid = int(sel[0])
        except (ValueError, TypeError):
            return
        self.select_roi(rid, from_table=True)

    def delete_selected(self):
        if self.selected_roi_id is None:
            sel = self.tree.selection()
            if not sel:
                return
            rid = int(sel[0])
        else:
            rid = self.selected_roi_id
        self.results = [r for r in self.results if r.roi_id != rid]
        self.selected_roi_id = None
        self.refresh_results()
        self.refresh_image()
        self.clear_profile_plot()
        self.plot_transfer_curves()

    def clear_all(self):
        self.results = []
        self.selected_roi_id = None
        self.next_roi_id = 1
        self.refresh_results()
        self.refresh_image()
        self.clear_profile_plot()
        self.plot_transfer_curves()

    def export_csv(self):
        if not self.results:
            messagebox.showinfo("No results", "Add at least one ROI first.")
            return
        out = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")],
                                           initialfile="USAF_MTF_measurements.csv")
        if not out:
            return
        rel = self.normalized_map()
        rows = []
        for r in self.results:
            rows.append({
                "ROI": r.roi_id,
                "Image": os.path.basename(self.image_path) if self.image_path else "",
                "Channel": r.channel,
                "Rotation_deg": r.rotation,
                "Orientation": r.orientation,
                "USAF_Group": r.group,
                "USAF_Element": r.element,
                "SpatialFrequency_lp_per_mm": r.frequency_lpmm,
                "LinePairPeriod_um": r.period_um,
                "SingleLineWidth_um": r.line_width_um,
                "Imax_mean": r.imax,
                "Imin_mean": r.imin,
                "Michelson_CTF": r.ctf,
                "Normalized_CTF": rel.get(r.roi_id, np.nan),
                "PeaksUsed": r.n_peaks,
                "ValleysUsed": r.n_valleys,
                "MeasuredFrequency_lp_per_mm": r.measured_frequency_lpmm,
                "ROI_x0": r.x0, "ROI_y0": r.y0, "ROI_x1": r.x1, "ROI_y1": r.y1,
            })
        pd.DataFrame(rows).to_csv(out, index=False)
        messagebox.showinfo("Saved", f"Saved:\n{out}")

    def save_current_plots(self):
        if not self.results:
            messagebox.showinfo("No results", "Add at least one ROI first.")
            return
        folder = filedialog.askdirectory(title="Choose output folder")
        if not folder:
            return
        self.fig_profile.savefig(os.path.join(folder, "selected_roi_profile.png"), dpi=500, bbox_inches="tight")
        self.fig_transfer.savefig(os.path.join(folder, "transfer_curves.png"), dpi=500, bbox_inches="tight")
        self.fig_image.savefig(os.path.join(folder, "roi_overview.png"), dpi=500, bbox_inches="tight")
        messagebox.showinfo("Saved", f"Saved plots to:\n{folder}")

    def save_publication_figure(self):
        """Export a compact, publication-oriented multi-panel figure.

        The layout is intentionally fixed to 183 × 108 mm so figures from
        different images have the same final height and width automatically.
        Multiple ROIs at the same USAF spatial frequency are summarized as
        mean ± SD in the transfer plots; the selected ROI is overlaid as a star.
        """
        if not self.results:
            messagebox.showinfo("No results", "Add at least one ROI first.")
            return

        folder = filedialog.askdirectory(title="Choose folder for scientific figure")
        if not folder:
            return

        img = self.current_image()
        if img is None:
            return

        selected = self.get_roi(self.selected_roi_id) or self.results[0]
        lo, hi = robust_limits(img)

        # Fixed physical size keeps figures consistent across datasets.
        fig_w_mm = 183.0
        fig_h_mm = 108.0
        fig = Figure(figsize=(fig_w_mm/25.4, fig_h_mm/25.4), dpi=100, facecolor="white")
        gs = gridspec.GridSpec(
            2, 3, figure=fig,
            width_ratios=[1.06, 0.72, 1.42],
            height_ratios=[1.02, 0.88],
            left=0.055, right=0.988, top=0.975, bottom=0.115,
            wspace=0.34, hspace=0.43
        )

        axA = fig.add_subplot(gs[0, 0])
        axB = fig.add_subplot(gs[0, 1])
        axC = fig.add_subplot(gs[0, 2])
        axD = fig.add_subplot(gs[1, 0:2])
        axE = fig.add_subplot(gs[1, 2])

        # Figure palette.
        c_x = "#0072B2"
        c_y = "#E69F00"
        c_sel = "#C51B7D"
        c_raw = "#B7B7B7"
        c_dark = "#222222"
        c_roi = "#D9D9D9"

        font = "Arial"
        fs = 6.2
        panel_fs = 8.2

        def panel_label(ax, letter):
            ax.text(-0.10, 1.035, letter, transform=ax.transAxes,
                    fontsize=panel_fs, fontweight="bold", fontfamily=font,
                    va="bottom", ha="left", color="black", clip_on=False)

        def clean_axes(ax):
            ax.tick_params(direction="out", length=2.5, width=0.55,                           labelsize=fs, pad=2)
            for lab in ax.get_xticklabels() + ax.get_yticklabels():
                lab.set_fontfamily(font)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(0.55)
            ax.spines["bottom"].set_linewidth(0.55)
            ax.set_facecolor("white")

        def add_scale_bar(ax, width_px, height_px, pixel_size_um):
            if not pixel_size_um or pixel_size_um <= 0 or width_px <= 0:
                return
            physical_width = width_px * pixel_size_um
            choices = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000], dtype=float)
            candidates = choices[choices <= physical_width * 0.28]
            if candidates.size == 0:
                return
            bar_um = float(candidates[-1])
            bar_px = bar_um / pixel_size_um
            x0 = 0.08 * width_px
            y0 = 0.90 * height_px
            ax.plot([x0, x0 + bar_px], [y0, y0], color="white",
                    linewidth=2.0, solid_capstyle="butt")
            ax.text(x0 + bar_px/2, y0 - 0.045*height_px,
                    f"{bar_um:g} µm", color="white", fontsize=5.2,
                    fontfamily=font, ha="center", va="top")

        def grouped_series(orientation):
            """Return frequency, mean CTF, SD, n for each frequency."""
            rows = [r for r in self.results
                    if r.orientation == orientation and np.isfinite(r.ctf)]
            if not rows:
                return np.array([]), np.array([]), np.array([]), np.array([])
            data = {}
            for r in rows:
                key = round(float(r.frequency_lpmm), 6)
                data.setdefault(key, []).append(float(r.ctf))
            freq, mean, sd, n = [], [], [], []
            for key in sorted(data):
                vals = np.asarray(data[key], dtype=float)
                freq.append(key)
                mean.append(float(np.mean(vals)))
                sd.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
                n.append(len(vals))
            return np.asarray(freq), np.asarray(mean), np.asarray(sd), np.asarray(n)

        series = [
            (ORIENTATION_X, "X response", c_x),
            (ORIENTATION_Y, "Y response", c_y),
        ]

        # ---------------- a | full image ----------------
        axA.imshow(img, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        for r in self.results:
            is_sel = (r.roi_id == selected.roi_id)
            axA.add_patch(Rectangle(
                (r.x0, r.y0), r.x1-r.x0, r.y1-r.y0,
                fill=False,
                edgecolor=c_sel if is_sel else c_roi,
                linewidth=1.35 if is_sel else 0.55,
                alpha=1.0 if is_sel else 0.85
            ))
        axA.text(selected.x0, max(0, selected.y0-3), f"ROI {selected.roi_id}",
                 color=c_sel, fontsize=5.2, fontfamily=font, fontweight="bold",
                 bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.9))
        axA.set_axis_off()
        panel_label(axA, "a")

        # ---------------- b | selected ROI + modest context ----------------
        rw = max(1, selected.x1-selected.x0)
        rh = max(1, selected.y1-selected.y0)
        pad_x = max(5, int(round(0.45*rw)))
        pad_y = max(5, int(round(0.45*rh)))
        xlo = max(0, selected.x0-pad_x)
        xhi = min(img.shape[1], selected.x1+pad_x)
        ylo = max(0, selected.y0-pad_y)
        yhi = min(img.shape[0], selected.y1+pad_y)
        context = img[ylo:yhi, xlo:xhi]
        axB.imshow(context, cmap="gray", vmin=lo, vmax=hi,
                   interpolation="nearest", aspect="equal")
        axB.add_patch(Rectangle(
            (selected.x0-xlo, selected.y0-ylo), rw, rh,
            fill=False, edgecolor=c_sel, linewidth=1.25
        ))
        # indicate profile direction without clutter
        if selected.orientation.startswith("Vertical"):
            yarr = selected.y0-ylo + 0.15*rh
            axB.annotate("", xy=(selected.x1-xlo, yarr),
                         xytext=(selected.x0-xlo, yarr),
                         arrowprops=dict(arrowstyle="->", color=c_sel, lw=0.8))
        else:
            xarr = selected.x0-xlo + 0.15*rw
            axB.annotate("", xy=(xarr, selected.y1-ylo),
                         xytext=(xarr, selected.y0-ylo),
                         arrowprops=dict(arrowstyle="->", color=c_sel, lw=0.8))
        axB.text(0.03, 0.97,
                 f"G{selected.group} E{selected.element}\n{selected.frequency_lpmm:.1f} lp mm$^{{-1}}$",
                 transform=axB.transAxes, fontsize=5.0, fontfamily=font,
                 ha="left", va="top", color="white",
                 bbox=dict(facecolor="black", edgecolor="none", alpha=0.52, pad=1.0))
        add_scale_bar(axB, context.shape[1], context.shape[0], self._pixel_size())
        axB.set_axis_off()
        panel_label(axB, "b")

        # ---------------- c | selected profile ----------------
        axC.plot(selected.profile_x, selected.profile_raw,
                 color=c_raw, linewidth=0.75, label="Raw")
        axC.plot(selected.profile_x, selected.profile_smooth,
                 color=c_dark, linewidth=1.15, label="Smoothed")
        axC.scatter(selected.profile_x[selected.peaks], selected.profile_smooth[selected.peaks],
                    s=10, color=c_x, zorder=4, label="Peaks")
        axC.scatter(selected.profile_x[selected.valleys], selected.profile_smooth[selected.valleys],
                    s=10, color=c_y, zorder=4, label="Valleys")
        axC.set_xlabel(f"Distance ({'µm' if self._pixel_size() else 'pixels'})",
                       fontsize=fs, fontfamily=font)
        axC.set_ylabel("Intensity (a.u.)", fontsize=fs, fontfamily=font)
        axC.text(0.98, 0.96,
                 f"CTF = {selected.ctf:.3f}",
                 transform=axC.transAxes, ha="right", va="top",
                 fontsize=5.4, fontfamily=font)
        axC.legend(frameon=False, fontsize=4.9, ncol=2, loc="lower left",
                   handlelength=1.3, columnspacing=0.7, borderaxespad=0.15)
        clean_axes(axC)
        panel_label(axC, "c")

        # common frequency limits
        valid_freqs = [r.frequency_lpmm for r in self.results if np.isfinite(r.ctf)]
        if valid_freqs:
            fmin, fmax = min(valid_freqs), max(valid_freqs)
            span = max(fmax-fmin, 1.0)
            xlim = (max(0, fmin-0.06*span), fmax+0.06*span)
        else:
            xlim = None

        # ---------------- d | absolute CTF, grouped mean ± SD ----------------
        all_ctf_for_ylim = []
        for ori, label, color in series:
            freq, mean, sd, n = grouped_series(ori)
            if freq.size:
                all_ctf_for_ylim.extend((mean+sd).tolist())
                if np.any(sd > 0):
                    axD.errorbar(freq, mean, yerr=sd, color=color, marker="o",
                                 markersize=3.0, linewidth=0.95, elinewidth=0.7,
                                 capsize=2.0, label=label)
                else:
                    axD.plot(freq, mean, color=color, marker="o", markersize=3.0,
                             linewidth=0.95, label=label)
        axD.scatter([selected.frequency_lpmm], [selected.ctf],
                    color=c_sel, marker="*", s=30, zorder=6, label="Selected ROI")
        axD.set_xlabel("Spatial frequency (lp mm$^{-1}$)", fontsize=fs, fontfamily=font)
        axD.set_ylabel("Michelson CTF", fontsize=fs, fontfamily=font)
        # keep zero baseline, but do not waste 80–90% of the panel when contrast is low
        ymax = max(all_ctf_for_ylim + [selected.ctf, 0.0]) if all_ctf_for_ylim else max(selected.ctf, 0.0)
        step = 0.05 if ymax <= 0.4 else 0.1
        ylim_top = min(1.0, max(0.20, np.ceil((1.18*ymax)/step)*step))
        axD.set_ylim(0, ylim_top)
        if xlim:
            axD.set_xlim(*xlim)
        axD.legend(frameon=False, fontsize=5.0, loc="upper right",
                   handlelength=1.35, borderaxespad=0.1, ncol=1)
        clean_axes(axD)
        panel_label(axD, "d")

        # Normalized transfer curve.
        selected_norm = np.nan
        for ori, label, color in series:
            freq, mean, sd, n = grouped_series(ori)
            if not freq.size:
                continue
            ref = mean[0] if mean[0] > 0 else np.nanmax(mean)
            norm = mean/ref if ref > 0 else np.full_like(mean, np.nan)
            norm_sd = sd/ref if ref > 0 else np.zeros_like(sd)
            if np.any(norm_sd > 0):
                axE.errorbar(freq, norm, yerr=norm_sd, color=color, marker="o",
                             markersize=3.0, linewidth=0.95, elinewidth=0.7,
                             capsize=2.0, label=label)
            else:
                axE.plot(freq, norm, color=color, marker="o", markersize=3.0,
                         linewidth=0.95, label=label)
            if selected.orientation == ori:
                selected_norm = selected.ctf/ref if ref > 0 else np.nan
        if np.isfinite(selected_norm):
            axE.scatter([selected.frequency_lpmm], [selected_norm],
                        color=c_sel, marker="*", s=30, zorder=6, label="Selected ROI")
        axE.set_xlabel("Spatial frequency (lp mm$^{-1}$)", fontsize=fs, fontfamily=font)
        axE.set_ylabel("Normalized CTF", fontsize=fs, fontfamily=font)
        axE.set_ylim(0, 1.08)
        if xlim:
            axE.set_xlim(*xlim)
        # legend omitted here to reduce repetition; color identity is established in panel d
        clean_axes(axE)
        panel_label(axE, "e")

        # Preserve the requested physical figure size.
        base = os.path.join(folder, "USAF_Fig")
        fig.savefig(base + ".png", dpi=600, facecolor="white")
        fig.savefig(base + ".tif", dpi=600, facecolor="white")
        fig.savefig(base + ".pdf", facecolor="white")
        fig.savefig(base + ".svg", facecolor="white")

        orientation_text = "X response" if selected.orientation.startswith("Vertical") else "Y response"
        caption = (
            "Fluorescent USAF 1951 contrast-transfer analysis. "
            f"a, Fluorescence image with analysis regions; selected ROI {selected.roi_id} is highlighted. "
            f"b, Enlarged view of G{selected.group} E{selected.element} "
            f"({selected.frequency_lpmm:.2f} lp mm^-1; {selected.line_width_um:.3f} µm single-line width). "
            "c, Averaged intensity profile with detected peaks and valleys used for Michelson contrast. "
            "d, Absolute Michelson CTF versus spatial frequency; points are mean ± s.d. when replicate ROIs are available. "
            "e, CTF normalized to the lowest measured spatial frequency for each bar orientation. "
            f"The highlighted ROI corresponds to the {orientation_text}."
        )
        with open(base + "_caption.txt", "w", encoding="utf-8") as fh:
            fh.write(caption + "\n")

        messagebox.showinfo(
            "Saved",
            "Scientific figure saved with fixed 183 × 108 mm dimensions\n"
            "(PNG, TIFF, PDF, SVG) plus caption."
        )

def main():
    root = tk.Tk()
    try:
        root.tk.call("tk", "scaling", 1.15)
    except tk.TclError:
        pass
    MTFAnalyzerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()