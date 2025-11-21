import argparse
import random
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
from dataclasses import dataclass

from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageTk


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def fill_color_for_mode(mode: str):
    if mode == "RGBA":
        return (0, 0, 0, 0)
    if mode in {"L", "LA"}:
        return 0 if mode == "L" else (0, 0)
    return (0, 0, 0)


def random_resized_crop(img: Image.Image, min_scale: float = 0.7) -> Image.Image:
    w, h = img.size
    min_scale = min(max(min_scale, 0.1), 1.0)
    scale = random.uniform(min_scale, 1.0)
    crop_w, crop_h = int(w * scale), int(h * scale)
    if crop_w == w or crop_h == h:
        return img
    left = random.randint(0, w - crop_w)
    top = random.randint(0, h - crop_h)
    cropped = img.crop((left, top, left + crop_w, top + crop_h))
    return cropped.resize((w, h), resample=Image.BICUBIC)


def add_noise(img: Image.Image) -> Image.Image:
    noise_strength = random.uniform(2.0, 8.0)
    noise = Image.effect_noise(img.size, noise_strength).convert("RGB")
    return Image.blend(img, noise, alpha=random.uniform(0.04, 0.12))


@dataclass
class AugmentConfig:
    hflip_prob: float
    vflip_prob: float
    rotate_prob: float
    max_rotate_deg: float
    crop_prob: float
    crop_min_scale: float
    brightness_prob: float
    brightness_min: float
    brightness_max: float
    contrast_prob: float
    contrast_min: float
    contrast_max: float
    color_prob: float
    color_min: float
    color_max: float
    sharpness_prob: float
    sharpness_min: float
    sharpness_max: float
    blur_prob: float
    blur_max_radius: float
    autocontrast_prob: float
    solarize_prob: float
    solarize_min_thresh: int
    solarize_max_thresh: int
    noise_prob: float
    noise_strength_min: float
    noise_strength_max: float
    noise_alpha_min: float
    noise_alpha_max: float


def apply_random_transforms(img: Image.Image, cfg: AugmentConfig) -> Image.Image:
    result = img.copy()
    fill = fill_color_for_mode(result.mode)

    if random.random() < cfg.hflip_prob:
        result = ImageOps.mirror(result)
    if random.random() < cfg.vflip_prob:
        result = ImageOps.flip(result)
    if random.random() < cfg.rotate_prob and cfg.max_rotate_deg > 0:
        angle = random.uniform(-cfg.max_rotate_deg, cfg.max_rotate_deg)
        result = result.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=fill)
        result = ImageOps.fit(result, img.size, method=Image.BICUBIC)
    if random.random() < cfg.crop_prob:
        result = random_resized_crop(result, cfg.crop_min_scale)
    if random.random() < cfg.brightness_prob:
        result = ImageEnhance.Brightness(result).enhance(random.uniform(cfg.brightness_min, cfg.brightness_max))
    if random.random() < cfg.contrast_prob:
        result = ImageEnhance.Contrast(result).enhance(random.uniform(cfg.contrast_min, cfg.contrast_max))
    if random.random() < cfg.color_prob:
        result = ImageEnhance.Color(result).enhance(random.uniform(cfg.color_min, cfg.color_max))
    if random.random() < cfg.sharpness_prob:
        result = ImageEnhance.Sharpness(result).enhance(random.uniform(cfg.sharpness_min, cfg.sharpness_max))
    if random.random() < cfg.blur_prob and cfg.blur_max_radius > 0:
        radius = random.uniform(0.0, cfg.blur_max_radius)
        result = result.filter(ImageFilter.GaussianBlur(radius=radius))
    if random.random() < cfg.autocontrast_prob:
        result = ImageOps.autocontrast(result, cutoff=random.uniform(0, 5))
    if random.random() < cfg.solarize_prob:
        result = ImageOps.solarize(result, threshold=random.randint(cfg.solarize_min_thresh, cfg.solarize_max_thresh))
    if random.random() < cfg.noise_prob:
        strength = random.uniform(cfg.noise_strength_min, cfg.noise_strength_max)
        alpha = random.uniform(cfg.noise_alpha_min, cfg.noise_alpha_max)
        noise = Image.effect_noise(result.size, strength).convert("RGB")
        result = Image.blend(result, noise, alpha=alpha)
    return result


DEFAULT_CFG = AugmentConfig(
    hflip_prob=0.5,
    vflip_prob=0.25,
    rotate_prob=0.6,
    max_rotate_deg=25.0,
    crop_prob=0.45,
    crop_min_scale=0.7,
    brightness_prob=0.55,
    brightness_min=0.7,
    brightness_max=1.3,
    contrast_prob=0.55,
    contrast_min=0.7,
    contrast_max=1.35,
    color_prob=0.45,
    color_min=0.7,
    color_max=1.3,
    sharpness_prob=0.35,
    sharpness_min=0.5,
    sharpness_max=1.6,
    blur_prob=0.3,
    blur_max_radius=1.2,
    autocontrast_prob=0.3,
    solarize_prob=0.35,
    solarize_min_thresh=96,
    solarize_max_thresh=192,
    noise_prob=0.3,
    noise_strength_min=2.0,
    noise_strength_max=8.0,
    noise_alpha_min=0.04,
    noise_alpha_max=0.12,
)


def augment_dataset(input_dir: Path, output_dir: Path, num_per_image: int, cfg: AugmentConfig,
                    progress_cb=None, preview_cb=None, stop_event: threading.Event = None,
                    pause_event: threading.Event = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = [p for p in input_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS]
    total = len(image_paths) * num_per_image if image_paths else 0
    done = 0
    for path in sorted(image_paths):
        with Image.open(path) as img:
            base = img.convert("RGB")
            for i in range(num_per_image):
                if stop_event and stop_event.is_set():
                    return
                while pause_event and pause_event.is_set():
                    if stop_event and stop_event.is_set():
                        return
                    time.sleep(0.05)
                augmented = apply_random_transforms(base, cfg)
                out_name = f"{path.stem}_aug{i + 1}{path.suffix}"
                augmented.save(output_dir / out_name, quality=95)
                done += 1
                if preview_cb:
                    preview_cb(augmented)
                if progress_cb:
                    progress_cb(done, total, path.name)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate augmented images.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=Path("data"), help="Directory with input images.")
    parser.add_argument("--output-dir", type=Path, default=Path("augmented"), help="Where to write augmented images.")
    parser.add_argument("--num-per-image", type=int, default=5, help="How many augmented samples per input image.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for reproducibility.")
    parser.add_argument("--gui", action="store_true", help="Launch GUI instead of CLI.")
    parser.add_argument("--hflip-prob", type=float, default=0.5, help="Horizontal flip probability.")
    parser.add_argument("--vflip-prob", type=float, default=0.25, help="Vertical flip probability.")
    parser.add_argument("--rotate-prob", type=float, default=0.6, help="Rotation probability.")
    parser.add_argument("--max-rotate-deg", type=float, default=25.0, help="Maximum absolute rotation in degrees.")
    parser.add_argument("--crop-prob", type=float, default=0.45, help="Random resized crop probability.")
    parser.add_argument("--crop-min-scale", type=float, default=0.7, help="Minimum scale for random crop.")
    parser.add_argument("--brightness-prob", type=float, default=0.55, help="Brightness change probability.")
    parser.add_argument("--brightness-range", type=float, nargs=2, default=(0.7, 1.3), metavar=("MIN", "MAX"),
                        help="Brightness factor range.")
    parser.add_argument("--contrast-prob", type=float, default=0.55, help="Contrast change probability.")
    parser.add_argument("--contrast-range", type=float, nargs=2, default=(0.7, 1.35), metavar=("MIN", "MAX"),
                        help="Contrast factor range.")
    parser.add_argument("--color-prob", type=float, default=0.45, help="Color saturation change probability.")
    parser.add_argument("--color-range", type=float, nargs=2, default=(0.7, 1.3), metavar=("MIN", "MAX"),
                        help="Color factor range.")
    parser.add_argument("--sharpness-prob", type=float, default=0.35, help="Sharpness change probability.")
    parser.add_argument("--sharpness-range", type=float, nargs=2, default=(0.5, 1.6), metavar=("MIN", "MAX"),
                        help="Sharpness factor range.")
    parser.add_argument("--blur-prob", type=float, default=0.3, help="Gaussian blur probability.")
    parser.add_argument("--blur-max-radius", type=float, default=1.2, help="Maximum blur radius.")
    parser.add_argument("--autocontrast-prob", type=float, default=0.3, help="Autocontrast probability.")
    parser.add_argument("--solarize-prob", type=float, default=0.35, help="Solarize probability.")
    parser.add_argument("--solarize-thresholds", type=int, nargs=2, default=(96, 192), metavar=("MIN", "MAX"),
                        help="Solarize threshold range.")
    parser.add_argument("--noise-prob", type=float, default=0.3, help="Noise addition probability.")
    parser.add_argument("--noise-strength", type=float, nargs=2, default=(2.0, 8.0), metavar=("MIN", "MAX"),
                        help="Noise strength range for effect_noise.")
    parser.add_argument("--noise-alpha", type=float, nargs=2, default=(0.04, 0.12), metavar=("MIN", "MAX"),
                        help="Blend alpha range for noise.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.gui:
        run_gui()
        return
    if args.seed is not None:
        random.seed(args.seed)
    if not args.input_dir.exists():
        raise SystemExit(f"Input directory {args.input_dir} does not exist.")
    cfg = build_config_from_args(args)
    augment_dataset(args.input_dir, args.output_dir, args.num_per_image, cfg)
    print(f"Augmented images saved to {args.output_dir.resolve()}")


def build_config_from_args(args) -> AugmentConfig:
    brightness_min, brightness_max = args.brightness_range
    contrast_min, contrast_max = args.contrast_range
    color_min, color_max = args.color_range
    sharpness_min, sharpness_max = args.sharpness_range
    solarize_min, solarize_max = args.solarize_thresholds
    noise_strength_min, noise_strength_max = args.noise_strength
    noise_alpha_min, noise_alpha_max = args.noise_alpha
    return AugmentConfig(
        hflip_prob=args.hflip_prob,
        vflip_prob=args.vflip_prob,
        rotate_prob=args.rotate_prob,
        max_rotate_deg=args.max_rotate_deg,
        crop_prob=args.crop_prob,
        crop_min_scale=args.crop_min_scale,
        brightness_prob=args.brightness_prob,
        brightness_min=brightness_min,
        brightness_max=brightness_max,
        contrast_prob=args.contrast_prob,
        contrast_min=contrast_min,
        contrast_max=contrast_max,
        color_prob=args.color_prob,
        color_min=color_min,
        color_max=color_max,
        sharpness_prob=args.sharpness_prob,
        sharpness_min=sharpness_min,
        sharpness_max=sharpness_max,
        blur_prob=args.blur_prob,
        blur_max_radius=args.blur_max_radius,
        autocontrast_prob=args.autocontrast_prob,
        solarize_prob=args.solarize_prob,
        solarize_min_thresh=solarize_min,
        solarize_max_thresh=solarize_max,
        noise_prob=args.noise_prob,
        noise_strength_min=noise_strength_min,
        noise_strength_max=noise_strength_max,
        noise_alpha_min=noise_alpha_min,
        noise_alpha_max=noise_alpha_max,
    )


def run_gui():
    root = tk.Tk()
    root.title("DJSVS AI - Image Augmentation")
    root.geometry("820x720")
    style = ttk.Style()
    style.theme_use("clam")
    palettes = {
        "Light": {"bg": "#ffffff", "card": "#f8fafc", "border": "#d1d5db", "text": "#0f172a", "muted": "#6b7280",
                  "accent": "#22c55e"},
        "Aurora": {"bg": "#0f172a", "card": "#162238", "border": "#243b53", "text": "#e5e7eb", "muted": "#9ca3af",
                   "accent": "#0ea5e9"},
        "Sunrise": {"bg": "#1f130a", "card": "#2c1a0d", "border": "#4a240d", "text": "#fef2e8", "muted": "#d1b29a",
                    "accent": "#f97316"},
        "Emerald": {"bg": "#0b1913", "card": "#0f261c", "border": "#1f3b2b", "text": "#e6f4ec", "muted": "#9fb7a7",
                    "accent": "#22c55e"},
    }

    status_bar = ttk.Label(root)
    canvas = None
    header_logo_holder = {"photo": None}

    def apply_palette(name: str):
        palette = palettes.get(name, palettes["Aurora"])
        bg = palette["bg"]
        card_bg = palette["card"]
        card_border = palette["border"]
        text_primary = palette["text"]
        muted = palette["muted"]
        accent = palette["accent"]

        root.configure(bg=bg)
        style.configure("TFrame", background=bg)
        style.configure("Header.TLabel", background=bg, foreground=text_primary, font=("Segoe UI", 18, "bold"))
        style.configure("Subheader.TLabel", background=bg, foreground=muted, font=("Segoe UI", 11))
        style.configure("Card.TLabelframe", background=card_bg, foreground=text_primary, bordercolor=card_border,
                        relief="solid", borderwidth=1, padding=12)
        style.configure("Card.TLabelframe.Label", background=card_bg, foreground=text_primary,
                        font=("Segoe UI", 11, "bold"))
        style.configure("Card.TLabel", background=card_bg, foreground=text_primary, font=("Segoe UI", 10))
        style.configure("Card.TEntry", fieldbackground="#ffffff", background="#ffffff", foreground=text_primary)
        style.configure("Card.TButton", background=accent, foreground="white", font=("Segoe UI", 11, "bold"))
        style.map("Card.TButton",
                  background=[("active", accent), ("disabled", "#1f2937")],
                  foreground=[("disabled", muted)])
        style.configure("Status.TLabel", background=bg, foreground=muted, font=("Segoe UI", 10))
        style.configure("TCombobox", fieldbackground=card_bg, background=card_bg, foreground=text_primary)
        status_bar.configure(background=bg, foreground=muted)
        if canvas is not None:
            canvas.configure(background=bg)

    container = ttk.Frame(root, style="TFrame")
    container.pack(fill="both", expand=True)
    canvas = tk.Canvas(container, highlightthickness=0)
    vscroll = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=vscroll.set)
    vscroll.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    content = ttk.Frame(canvas, style="TFrame")
    window_id = canvas.create_window((0, 0), window=content, anchor="nw")

    def _on_config(event):
        canvas.configure(scrollregion=canvas.bbox("all"))
        canvas.itemconfigure(window_id, width=canvas.winfo_width())
        update_background_image()

    content.bind("<Configure>", _on_config)

    def _on_mousewheel(event):
        delta = event.delta
        if delta == 0 and hasattr(event, "num"):
            if event.num == 4:
                delta = 120
            elif event.num == 5:
                delta = -120
        canvas.yview_scroll(int(-1 * (delta / 120)), "units")

    canvas.bind_all("<MouseWheel>", _on_mousewheel)
    canvas.bind_all("<Button-4>", _on_mousewheel)
    canvas.bind_all("<Button-5>", _on_mousewheel)

    apply_palette("Light")

    input_var = tk.StringVar(value=str(Path("data").resolve()))
    output_var = tk.StringVar(value=str(Path("augmented").resolve()))
    seed_var = tk.StringVar(value="")
    num_var = tk.DoubleVar(value=5)
    sandbox_path_var = tk.StringVar(value="")
    sandbox_seed_var = tk.StringVar(value="")

    cfg_vars = {
        "hflip_prob": tk.DoubleVar(value=DEFAULT_CFG.hflip_prob),
        "vflip_prob": tk.DoubleVar(value=DEFAULT_CFG.vflip_prob),
        "rotate_prob": tk.DoubleVar(value=DEFAULT_CFG.rotate_prob),
        "max_rotate_deg": tk.DoubleVar(value=DEFAULT_CFG.max_rotate_deg),
        "crop_prob": tk.DoubleVar(value=DEFAULT_CFG.crop_prob),
        "crop_min_scale": tk.DoubleVar(value=DEFAULT_CFG.crop_min_scale),
        "brightness_prob": tk.DoubleVar(value=DEFAULT_CFG.brightness_prob),
        "brightness_min": tk.DoubleVar(value=DEFAULT_CFG.brightness_min),
        "brightness_max": tk.DoubleVar(value=DEFAULT_CFG.brightness_max),
        "contrast_prob": tk.DoubleVar(value=DEFAULT_CFG.contrast_prob),
        "contrast_min": tk.DoubleVar(value=DEFAULT_CFG.contrast_min),
        "contrast_max": tk.DoubleVar(value=DEFAULT_CFG.contrast_max),
        "color_prob": tk.DoubleVar(value=DEFAULT_CFG.color_prob),
        "color_min": tk.DoubleVar(value=DEFAULT_CFG.color_min),
        "color_max": tk.DoubleVar(value=DEFAULT_CFG.color_max),
        "sharpness_prob": tk.DoubleVar(value=DEFAULT_CFG.sharpness_prob),
        "sharpness_min": tk.DoubleVar(value=DEFAULT_CFG.sharpness_min),
        "sharpness_max": tk.DoubleVar(value=DEFAULT_CFG.sharpness_max),
        "blur_prob": tk.DoubleVar(value=DEFAULT_CFG.blur_prob),
        "blur_max_radius": tk.DoubleVar(value=DEFAULT_CFG.blur_max_radius),
        "autocontrast_prob": tk.DoubleVar(value=DEFAULT_CFG.autocontrast_prob),
        "solarize_prob": tk.DoubleVar(value=DEFAULT_CFG.solarize_prob),
        "solarize_min_thresh": tk.DoubleVar(value=DEFAULT_CFG.solarize_min_thresh),
        "solarize_max_thresh": tk.DoubleVar(value=DEFAULT_CFG.solarize_max_thresh),
        "noise_prob": tk.DoubleVar(value=DEFAULT_CFG.noise_prob),
        "noise_strength_min": tk.DoubleVar(value=DEFAULT_CFG.noise_strength_min),
        "noise_strength_max": tk.DoubleVar(value=DEFAULT_CFG.noise_strength_max),
        "noise_alpha_min": tk.DoubleVar(value=DEFAULT_CFG.noise_alpha_min),
        "noise_alpha_max": tk.DoubleVar(value=DEFAULT_CFG.noise_alpha_max),
    }

    def browse_dir(var: tk.StringVar, title: str):
        path = filedialog.askdirectory(title=title)
        if path:
            var.set(path)

    sandbox_image = {"img": None}

    def browse_file(var: tk.StringVar, holder: dict):
        path = filedialog.askopenfilename(title="Select image",
                                          filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")])
        if path:
            var.set(path)
            try:
                img = Image.open(path).convert("RGB")
                holder["img"] = img
                set_image_on_label(sandbox_orig_label, sandbox_orig_photo, img)
            except Exception as exc:
                messagebox.showerror("File error", f"Cannot open image: {exc}")

    def update_background_image():
        pass

    # Header
    header = ttk.Frame(content, style="TFrame")
    header.pack(fill="x", padx=14, pady=(14, 6))
    top_header = ttk.Frame(header, style="TFrame")
    top_header.pack(fill="x")
    ttk.Label(top_header, text="DJSVS AI - Image Augmentation", style="Header.TLabel").pack(side="left", anchor="w")
    # Header logo on the right if present (prefers a.png, fallback logo.png)
    for logo_candidate in ("a.png", "logo.png"):
        logo_path = Path(logo_candidate)
        if logo_path.exists():
            try:
                logo_img = Image.open(logo_path).convert("RGBA")
                logo_img.thumbnail((96, 96))
                logo_photo = ImageTk.PhotoImage(logo_img)
                header_logo_holder["photo"] = logo_photo
                ttk.Label(top_header, image=logo_photo, style="TFrame").pack(side="right", padx=6)
                break
            except Exception:
                continue
    theme_var = tk.StringVar(value="Light")
    ttk.Label(top_header, text="Theme", style="Subheader.TLabel").pack(side="right", padx=(0, 6))
    theme_combo = ttk.Combobox(top_header, textvariable=theme_var, values=list(palettes.keys()), width=10,
                               state="readonly")
    theme_combo.pack(side="right")

    ttk.Label(header, text="Configure your augmentation recipe and generate more training images.",
              style="Subheader.TLabel").pack(anchor="w", pady=(2, 0))
    ttk.Label(header, text="Developed by Vishal Maddeshiya", style="Subheader.TLabel").pack(anchor="w", pady=(0, 0))
    ttk.Label(header, text="Contact: vishalaidev7426@gmail.com", style="Subheader.TLabel").pack(anchor="w", pady=(0, 4))

    paths_frame = ttk.LabelFrame(content, text="Paths", style="Card.TLabelframe")
    paths_frame.pack(fill="x", padx=14, pady=8)
    ttk.Label(paths_frame, text="Input directory", style="Card.TLabel").grid(row=0, column=0, sticky="w", padx=6, pady=4)
    ttk.Entry(paths_frame, textvariable=input_var, width=70, style="Card.TEntry").grid(row=0, column=1, padx=6, pady=4)
    ttk.Button(paths_frame, text="Browse", style="Card.TButton",
               command=lambda: browse_dir(input_var, "Select input directory")).grid(row=0, column=2, padx=6, pady=4)
    ttk.Label(paths_frame, text="Output directory", style="Card.TLabel").grid(row=1, column=0, sticky="w", padx=6, pady=4)
    ttk.Entry(paths_frame, textvariable=output_var, width=70, style="Card.TEntry").grid(row=1, column=1, padx=6, pady=4)
    ttk.Button(paths_frame, text="Browse", style="Card.TButton",
               command=lambda: browse_dir(output_var, "Select output directory")).grid(row=1, column=2, padx=6, pady=4)

    top_frame = ttk.LabelFrame(content, text="Run settings", style="Card.TLabelframe")
    top_frame.pack(fill="x", padx=14, pady=8)
    ttk.Label(top_frame, text="Augmented per image", style="Card.TLabel").grid(row=0, column=0, sticky="w", padx=6, pady=4)
    ttk.Scale(top_frame, from_=1, to=20, orient="horizontal", variable=num_var, length=200).grid(row=0, column=1,
                                                                                                 padx=6, pady=4)
    ttk.Label(top_frame, textvariable=num_var, style="Card.TLabel").grid(row=0, column=2, sticky="w", padx=(6, 0))
    ttk.Label(top_frame, text="Seed (optional)", style="Card.TLabel").grid(row=0, column=3, sticky="w", padx=(12, 0), pady=4)
    ttk.Entry(top_frame, textvariable=seed_var, width=12, style="Card.TEntry").grid(row=0, column=4, padx=6, pady=4)

    opts_frame = ttk.LabelFrame(content, text="Augmentation settings", style="Card.TLabelframe")
    opts_frame.pack(fill="x", padx=14, pady=8)

    def add_slider(row, label_text, key, from_v=0.0, to_v=1.0, resolution=0.01):
        ttk.Label(opts_frame, text=label_text, style="Card.TLabel").grid(row=row, column=0, sticky="w", padx=6, pady=4)
        ttk.Scale(opts_frame, from_=from_v, to=to_v, orient="horizontal", variable=cfg_vars[key], length=180,
                  command=lambda v, k=key: cfg_vars[k].set(float(v))).grid(row=row, column=1, padx=6, pady=4)
        ttk.Label(opts_frame, textvariable=cfg_vars[key], style="Card.TLabel").grid(row=row, column=2, sticky="w",
                                                                                     padx=6, pady=4)

    add_slider(0, "Horizontal flip prob", "hflip_prob")
    add_slider(1, "Vertical flip prob", "vflip_prob")
    add_slider(2, "Rotation prob", "rotate_prob")
    add_slider(3, "Crop prob", "crop_prob")
    add_slider(4, "Brightness prob", "brightness_prob")
    add_slider(5, "Contrast prob", "contrast_prob")
    add_slider(6, "Color prob", "color_prob")
    add_slider(7, "Sharpness prob", "sharpness_prob")
    add_slider(8, "Blur prob", "blur_prob")
    add_slider(9, "Autocontrast prob", "autocontrast_prob")
    add_slider(10, "Solarize prob", "solarize_prob")
    add_slider(11, "Noise prob", "noise_prob")

    # Ranges remain numeric entries for finer control
    range_fields = [
        ("Max rotation deg", "max_rotate_deg"),
        ("Crop min scale", "crop_min_scale"),
        ("Brightness min", "brightness_min"),
        ("Brightness max", "brightness_max"),
        ("Contrast min", "contrast_min"),
        ("Contrast max", "contrast_max"),
        ("Color min", "color_min"),
        ("Color max", "color_max"),
        ("Sharpness min", "sharpness_min"),
        ("Sharpness max", "sharpness_max"),
        ("Blur max radius", "blur_max_radius"),
        ("Solarize min thresh", "solarize_min_thresh"),
        ("Solarize max thresh", "solarize_max_thresh"),
        ("Noise strength min", "noise_strength_min"),
        ("Noise strength max", "noise_strength_max"),
        ("Noise alpha min", "noise_alpha_min"),
        ("Noise alpha max", "noise_alpha_max"),
    ]

    base_row = 12
    for offset, (label, key) in enumerate(range_fields):
        row = base_row + offset
        ttk.Label(opts_frame, text=label, style="Card.TLabel").grid(row=row, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(opts_frame, textvariable=cfg_vars[key], width=12, style="Card.TEntry").grid(row=row, column=1,
                                                                                            padx=6, pady=4)

    status_var = tk.StringVar(value="Idle")
    current_file_var = tk.StringVar(value="")
    status_bar.configure(textvariable=status_var, style="Status.TLabel", anchor="w")
    status_bar.pack(fill="x", padx=14, pady=(6, 2), side="bottom")

    # Sandbox for interactive tuning
    sandbox_frame = ttk.LabelFrame(content, text="Sandbox (preview a single image with current settings)",
                                   style="Card.TLabelframe")
    sandbox_frame.pack(fill="x", padx=14, pady=8)
    ttk.Label(sandbox_frame, text="Image path", style="Card.TLabel").grid(row=0, column=0, sticky="w", padx=6, pady=4)
    ttk.Entry(sandbox_frame, textvariable=sandbox_path_var, width=50, style="Card.TEntry").grid(row=0, column=1,
                                                                                                padx=6, pady=4)
    ttk.Button(sandbox_frame, text="Browse", style="Card.TButton",
               command=lambda: browse_file(sandbox_path_var, sandbox_image)).grid(row=0, column=2, padx=6, pady=4)
    ttk.Label(sandbox_frame, text="Seed (optional)", style="Card.TLabel").grid(row=0, column=3, sticky="w",
                                                                               padx=(12, 0), pady=4)
    ttk.Entry(sandbox_frame, textvariable=sandbox_seed_var, width=10, style="Card.TEntry").grid(row=0, column=4,
                                                                                                padx=6, pady=4)
    ttk.Button(sandbox_frame, text="Preview augmentation", style="Card.TButton",
               command=lambda: apply_sandbox(cfg_vars)).grid(row=0, column=5, padx=6, pady=4)

    sandbox_images_frame = ttk.Frame(sandbox_frame, style="TFrame")
    sandbox_images_frame.grid(row=1, column=0, columnspan=6, pady=6, padx=6, sticky="w")
    sandbox_orig_label = ttk.Label(sandbox_images_frame, text="Original", style="Card.TLabel")
    sandbox_orig_label.grid(row=0, column=0, padx=8, pady=4)
    sandbox_aug_label = ttk.Label(sandbox_images_frame, text="Augmented", style="Card.TLabel")
    sandbox_aug_label.grid(row=0, column=1, padx=8, pady=4)
    sandbox_orig_photo = {"photo": None}
    sandbox_aug_photo = {"photo": None}

    preview_frame = ttk.LabelFrame(content, text="Live preview (batch run)", style="Card.TLabelframe")
    preview_frame.pack(fill="x", padx=14, pady=8)
    preview_label = ttk.Label(preview_frame, text="No image yet", style="Card.TLabel")
    preview_label.pack(padx=8, pady=8)
    preview_photo_holder = {"photo": None}

    progress_bar = ttk.Progressbar(content, mode="determinate")
    progress_bar.pack(fill="x", padx=14, pady=(0, 8))

    file_label = ttk.Label(content, textvariable=current_file_var, style="Status.TLabel", anchor="w")
    file_label.pack(fill="x", padx=14, pady=(0, 8))

    controls_frame = ttk.Frame(content, style="TFrame")
    controls_frame.pack(pady=6)
    start_button = ttk.Button(controls_frame, text="Start", style="Card.TButton")
    pause_button = ttk.Button(controls_frame, text="Pause", style="Card.TButton")
    stop_button = ttk.Button(controls_frame, text="Stop", style="Card.TButton")
    start_button.grid(row=0, column=0, padx=6)
    pause_button.grid(row=0, column=1, padx=6)
    stop_button.grid(row=0, column=2, padx=6)

    augment_thread = {"thread": None}
    stop_event = threading.Event()
    pause_event = threading.Event()

    def start():
        try:
            num_val = int(num_var.get())
            if num_val < 1:
                raise ValueError("num-per-image must be >= 1")
            seed_value = seed_var.get().strip()
            if seed_value:
                random.seed(int(seed_value))
            cfg_vals = {k: v.get() for k, v in cfg_vars.items()}
            cfg = AugmentConfig(
                hflip_prob=float(cfg_vals["hflip_prob"]),
                vflip_prob=float(cfg_vals["vflip_prob"]),
                rotate_prob=float(cfg_vals["rotate_prob"]),
                max_rotate_deg=float(cfg_vals["max_rotate_deg"]),
                crop_prob=float(cfg_vals["crop_prob"]),
                crop_min_scale=float(cfg_vals["crop_min_scale"]),
                brightness_prob=float(cfg_vals["brightness_prob"]),
                brightness_min=float(cfg_vals["brightness_min"]),
                brightness_max=float(cfg_vals["brightness_max"]),
                contrast_prob=float(cfg_vals["contrast_prob"]),
                contrast_min=float(cfg_vals["contrast_min"]),
                contrast_max=float(cfg_vals["contrast_max"]),
                color_prob=float(cfg_vals["color_prob"]),
                color_min=float(cfg_vals["color_min"]),
                color_max=float(cfg_vals["color_max"]),
                sharpness_prob=float(cfg_vals["sharpness_prob"]),
                sharpness_min=float(cfg_vals["sharpness_min"]),
                sharpness_max=float(cfg_vals["sharpness_max"]),
                blur_prob=float(cfg_vals["blur_prob"]),
                blur_max_radius=float(cfg_vals["blur_max_radius"]),
                autocontrast_prob=float(cfg_vals["autocontrast_prob"]),
                solarize_prob=float(cfg_vals["solarize_prob"]),
                solarize_min_thresh=int(float(cfg_vals["solarize_min_thresh"])),
                solarize_max_thresh=int(float(cfg_vals["solarize_max_thresh"])),
                noise_prob=float(cfg_vals["noise_prob"]),
                noise_strength_min=float(cfg_vals["noise_strength_min"]),
                noise_strength_max=float(cfg_vals["noise_strength_max"]),
                noise_alpha_min=float(cfg_vals["noise_alpha_min"]),
                noise_alpha_max=float(cfg_vals["noise_alpha_max"]),
            )
        except Exception as exc:
            messagebox.showerror("Invalid value", str(exc))
            return

        in_dir = Path(input_var.get()).expanduser()
        out_dir = Path(output_var.get()).expanduser()
        if not in_dir.exists():
            messagebox.showerror("Input missing", f"Input directory does not exist: {in_dir}")
            return

        stop_event.clear()
        pause_event.clear()
        start_button.config(state="disabled")
        pause_button.config(state="normal", text="Pause")
        stop_button.config(state="normal")
        status_var.set("Running...")
        current_file_var.set("")
        progress_bar["value"] = 0
        total_images = len([p for p in in_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS]) * num_val
        progress_bar["maximum"] = max(total_images, 1)

        def worker():
            try:
                def progress_cb(done, total, name):
                    root.after(0, lambda: update_progress(done, total, name))

                def preview_cb(image: Image.Image):
                    root.after(0, lambda: update_preview(image))

                augment_dataset(in_dir, out_dir, num_val, cfg, progress_cb=progress_cb, preview_cb=preview_cb,
                                stop_event=stop_event, pause_event=pause_event)
                if stop_event.is_set():
                    msg = "Stopped by user."
                    ok = False
                else:
                    msg = f"Done. Saved to {out_dir.resolve()}"
                    ok = True
            except Exception as exc:  # pragma: no cover - GUI path
                msg = f"Error: {exc}"
                ok = False
            root.after(0, lambda: finish(msg, ok))

        t = threading.Thread(target=worker, daemon=True)
        augment_thread["thread"] = t
        t.start()

    def finish(message: str, success: bool):
        status_var.set(message)
        start_button.config(state="normal")
        pause_button.config(state="disabled", text="Pause")
        stop_button.config(state="disabled")
        if success:
            messagebox.showinfo("Augmentation complete", message)
        else:
            messagebox.showerror("Augmentation error", message)

    def set_image_on_label(label: ttk.Label, holder: dict, image: Image.Image, max_size=(240, 240)):
        img_copy = image.copy()
        img_copy.thumbnail(max_size)
        photo = ImageTk.PhotoImage(img_copy)
        holder["photo"] = photo
        label.configure(image=photo, text="")

    def update_progress(done: int, total: int, name: str):
        progress_bar["value"] = done
        status_var.set(f"Processing {done}/{total}" if total else f"Processing {done}")
        current_file_var.set(f"Current file: {name}")

    def update_preview(image: Image.Image):
        set_image_on_label(preview_label, preview_photo_holder, image, max_size=(320, 320))

    def apply_sandbox(cfg_vars_local):
        if sandbox_image["img"] is None:
            messagebox.showerror("Sandbox", "Load an image first.")
            return
        seed_val = sandbox_seed_var.get().strip()
        if seed_val:
            random.seed(int(seed_val))
        cfg_vals = {k: v.get() for k, v in cfg_vars_local.items()}
        cfg_local = AugmentConfig(
            hflip_prob=float(cfg_vals["hflip_prob"]),
            vflip_prob=float(cfg_vals["vflip_prob"]),
            rotate_prob=float(cfg_vals["rotate_prob"]),
            max_rotate_deg=float(cfg_vals["max_rotate_deg"]),
            crop_prob=float(cfg_vals["crop_prob"]),
            crop_min_scale=float(cfg_vals["crop_min_scale"]),
            brightness_prob=float(cfg_vals["brightness_prob"]),
            brightness_min=float(cfg_vals["brightness_min"]),
            brightness_max=float(cfg_vals["brightness_max"]),
            contrast_prob=float(cfg_vals["contrast_prob"]),
            contrast_min=float(cfg_vals["contrast_min"]),
            contrast_max=float(cfg_vals["contrast_max"]),
            color_prob=float(cfg_vals["color_prob"]),
            color_min=float(cfg_vals["color_min"]),
            color_max=float(cfg_vals["color_max"]),
            sharpness_prob=float(cfg_vals["sharpness_prob"]),
            sharpness_min=float(cfg_vals["sharpness_min"]),
            sharpness_max=float(cfg_vals["sharpness_max"]),
            blur_prob=float(cfg_vals["blur_prob"]),
            blur_max_radius=float(cfg_vals["blur_max_radius"]),
            autocontrast_prob=float(cfg_vals["autocontrast_prob"]),
            solarize_prob=float(cfg_vals["solarize_prob"]),
            solarize_min_thresh=int(float(cfg_vals["solarize_min_thresh"])),
            solarize_max_thresh=int(float(cfg_vals["solarize_max_thresh"])),
            noise_prob=float(cfg_vals["noise_prob"]),
            noise_strength_min=float(cfg_vals["noise_strength_min"]),
            noise_strength_max=float(cfg_vals["noise_strength_max"]),
            noise_alpha_min=float(cfg_vals["noise_alpha_min"]),
            noise_alpha_max=float(cfg_vals["noise_alpha_max"]),
        )
        augmented = apply_random_transforms(sandbox_image["img"], cfg_local)
        set_image_on_label(sandbox_aug_label, sandbox_aug_photo, augmented)
        set_image_on_label(sandbox_orig_label, sandbox_orig_photo, sandbox_image["img"])

    def on_theme_change(event=None):
        apply_palette(theme_var.get())
        update_background_image()

    pause_button.config(state="disabled")
    stop_button.config(state="disabled")

    def toggle_pause():
        if augment_thread["thread"] is None or not augment_thread["thread"].is_alive():
            return
        if pause_event.is_set():
            pause_event.clear()
            pause_button.config(text="Pause")
            status_var.set("Resumed")
        else:
            pause_event.set()
            pause_button.config(text="Resume")
            status_var.set("Paused")

    def stop_run():
        if augment_thread["thread"] is None:
            return
        stop_event.set()
        pause_event.clear()
        pause_button.config(text="Pause", state="disabled")
        stop_button.config(state="disabled")
        status_var.set("Stopping...")

    start_button.config(command=start)
    pause_button.config(command=toggle_pause)
    stop_button.config(command=stop_run)
    theme_combo.bind("<<ComboboxSelected>>", on_theme_change)
    apply_palette(theme_var.get())
    root.mainloop()


if __name__ == "__main__":
    main()
