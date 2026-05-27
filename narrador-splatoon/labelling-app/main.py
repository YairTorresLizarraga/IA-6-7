from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

from PIL import Image, ImageTk


# Set the category to label here (must exist in dataset/parameters.json)
SELECTED_CATEGORY = "health"


@dataclass
class Section:
    x: int
    y: int
    width: int
    height: int


class ImageLabellerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Dataset Image Labeller")
        self.root.geometry("1280x820")

        self.project_root = Path(__file__).resolve().parents[1]
        self.dataset_dir = self.project_root / "dataset"
        self.parameters_path = self.dataset_dir / "parameters.jsonc"
        self.progress_path = self.dataset_dir / "labelling.json"
        self.pending_dir = self.dataset_dir / "pending_tag"

        self.category = SELECTED_CATEGORY
        self.labels: list[str] = []
        self.sections: list[Section] = []
        self.individual_sections: list[Section] = []
        self.progress_by_category: dict[str, str] = {}
        self.last_labeled_index: int | None = None

        self._load_parameters()
        self.image_paths = self._load_pending_images()
        self._load_progress()
        self.current_index = self._resolve_start_index()
        self.current_pil_image: Image.Image | None = None
        self.current_tk_image: ImageTk.PhotoImage | None = None

        self._build_ui()
        self._bind_keys()
        self._show_current_image()

    def _load_pending_images(self) -> list[Path]:
        if not self.pending_dir.exists():
            raise FileNotFoundError(f"Pending folder not found: {self.pending_dir}")

        extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        files = [
            path
            for path in sorted(self.pending_dir.iterdir())
            if path.is_file() and path.suffix.lower() in extensions
        ]
        return files

    def _load_progress(self) -> None:
        if not self.progress_path.exists():
            self.progress_by_category = {}
            self.last_labeled_index = None
            return

        try:
            with self.progress_path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
        except Exception:  # noqa: BLE001
            self.progress_by_category = {}
            self.last_labeled_index = None
            return

        if not isinstance(payload, dict):
            self.progress_by_category = {}
            self.last_labeled_index = None
            return

        self.progress_by_category = {
            str(key): str(value)
            for key, value in payload.items()
            if isinstance(key, str) and isinstance(value, str)
        }

        last_name = self.progress_by_category.get(self.category)
        if last_name is None:
            self.last_labeled_index = None
            return

        self.last_labeled_index = self._find_image_index_by_name(last_name)

    def _save_progress(self) -> None:
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)
        with self.progress_path.open("w", encoding="utf-8") as file:
            json.dump(self.progress_by_category, file, ensure_ascii=False, indent=2)

    def _resolve_start_index(self) -> int:
        if self.last_labeled_index is None:
            return 0

        return min(self.last_labeled_index + 1, len(self.image_paths))

    def _find_image_index_by_name(self, image_name: str) -> int | None:
        for idx, path in enumerate(self.image_paths):
            if path.name == image_name:
                return idx
        return None

    def _set_last_successful_image(self, image_path: Path) -> None:
        self.progress_by_category[self.category] = image_path.name
        self.last_labeled_index = self._find_image_index_by_name(image_path.name)
        self._save_progress()

    def _load_parameters(self) -> None:
        if not self.parameters_path.exists():
            raise FileNotFoundError(f"Parameters file not found: {self.parameters_path}")

        with self.parameters_path.open("r", encoding="utf-8") as file:
            parameters = json.load(file)

        if self.category not in parameters:
            available = ", ".join(sorted(parameters.keys()))
            raise ValueError(
                f"Category '{self.category}' not found in parameters.json. "
                f"Available categories: {available}"
            )

        category_cfg = parameters[self.category]
        self.labels = category_cfg.get("labels", [])
        self.sections = [Section(**section) for section in category_cfg.get("sections", [])]
        self.individual_sections = [
            Section(**section) for section in category_cfg.get("individual_sections", [])
        ]

        if not self.labels:
            raise ValueError(f"Category '{self.category}' has no labels configured")

    def _build_ui(self) -> None:
        self.info_var = tk.StringVar()
        self.help_var = tk.StringVar()

        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        top_bar = tk.Frame(main_frame)
        top_bar.pack(fill=tk.X, padx=10, pady=(10, 4))

        info_label = tk.Label(top_bar, textvariable=self.info_var, anchor="w", font=("Arial", 12))
        info_label.pack(fill=tk.X)

        help_label = tk.Label(
            top_bar,
            textvariable=self.help_var,
            anchor="w",
            justify=tk.LEFT,
            font=("Arial", 11),
        )
        help_label.pack(fill=tk.X, pady=(4, 0))

        self.image_label = tk.Label(main_frame, bg="#202020")
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=(6, 10))

        shortcuts = [f"{idx + 1}:{label}" for idx, label in enumerate(self.labels)]
        shortcuts_text = " | ".join(shortcuts)
        self.help_var.set(
            f"Select label with number keys -> {shortcuts_text} | "
            "S: skip | B: back to last labeled | D: delete | Esc: quit"
        )

        self.root.update_idletasks()

    def _bind_keys(self) -> None:
        self.root.bind("<Escape>", lambda _: self.root.destroy())
        self.root.bind("s", lambda _: self._skip_current_image())
        self.root.bind("S", lambda _: self._skip_current_image())
        self.root.bind("b", lambda _: self._go_to_last_successful())
        self.root.bind("B", lambda _: self._go_to_last_successful())
        self.root.bind("d", lambda _: self._delete_current_image())
        self.root.bind("D", lambda _: self._delete_current_image())
        for index in range(len(self.labels)):
            key = str(index + 1)
            self.root.bind(key, lambda _, i=index: self._label_current_image(i))

    def _show_current_image(self) -> None:
        total = len(self.image_paths)
        if total == 0:
            self.info_var.set(
                f"Category: {self.category} | No images found in {self.pending_dir.as_posix()}"
            )
            self.image_label.config(image="", text="No pending images", fg="white", font=("Arial", 24))
            return

        if self.current_index >= total:
            self.info_var.set(f"Category: {self.category} | Completed! {total}/{total} labeled")
            self.image_label.config(
                image="",
                text="All images labeled. Press Esc to close.",
                fg="white",
                font=("Arial", 24),
            )
            return

        image_path = self.image_paths[self.current_index]
        self.current_pil_image = Image.open(image_path).convert("RGB")

        canvas_w = max(self.image_label.winfo_width(), 300)
        canvas_h = max(self.image_label.winfo_height(), 300)

        display_image = self.current_pil_image.copy()
        display_image.thumbnail((canvas_w - 20, canvas_h - 20), Image.Resampling.LANCZOS)

        self.current_tk_image = ImageTk.PhotoImage(display_image)
        self.image_label.config(image=self.current_tk_image, text="")

        self.info_var.set(
            f"Category: {self.category} | Image {self.current_index + 1}/{total} | {image_path.name}"
        )

    def _label_current_image(self, label_index: int) -> None:
        if self.current_index >= len(self.image_paths):
            return

        selected_label = self.labels[label_index]
        image_path = self.image_paths[self.current_index]

        try:
            self._export_labeled_image(image_path, selected_label)
            self._remove_old_labeled_exports(image_path, selected_label)
            self._set_last_successful_image(image_path)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Export error", f"Failed to export '{image_path.name}':\n{exc}")
            return

        self.current_index += 1
        self._show_current_image()

    def _skip_current_image(self) -> None:
        if self.current_index >= len(self.image_paths):
            return

        self.current_index += 1
        self._show_current_image()

    def _go_to_last_successful(self) -> None:
        if self.last_labeled_index is None:
            messagebox.showinfo("Back", "No successful label yet for this category.")
            return

        self.current_index = self.last_labeled_index
        self._show_current_image()

    def _delete_current_image(self) -> None:
        if self.current_index >= len(self.image_paths):
            return

        image_path = self.image_paths[self.current_index]

        try:
            image_path.unlink()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Delete error", f"Failed to delete '{image_path.name}':\n{exc}")
            return

        del self.image_paths[self.current_index]
        self._show_current_image()

    def _export_labeled_image(self, image_path: Path, selected_label: str) -> None:
        output_dir = self.dataset_dir / self.category / selected_label
        output_dir.mkdir(parents=True, exist_ok=True)

        with Image.open(image_path).convert("RGB") as source:
            if self.individual_sections:
                crops = [self._crop_section(source, section) for section in self.individual_sections]
                for idx, crop in enumerate(crops, start=1):
                    out_name = f"{image_path.stem}_part_{idx:02d}.webp"
                    crop.save(output_dir / out_name, format="WEBP", quality=95)
            elif self.sections:
                crops = [self._crop_section(source, section) for section in self.sections]
                joined = self._join_horizontally(crops)
                joined.save(output_dir / f"{image_path.stem}.webp", format="WEBP", quality=95)
            else:
                source.save(output_dir / f"{image_path.stem}.webp", format="WEBP", quality=95)

    def _remove_old_labeled_exports(self, image_path: Path, selected_label: str) -> None:
        expected_output_names = self._expected_output_names(image_path)

        for label in self.labels:
            if label == selected_label:
                continue

            label_dir = self.dataset_dir / self.category / label
            if not label_dir.exists():
                continue

            for name in expected_output_names:
                candidate = label_dir / name
                if candidate.exists():
                    candidate.unlink()

    def _expected_output_names(self, image_path: Path) -> set[str]:
        if self.individual_sections:
            return {
                f"{image_path.stem}_part_{idx:02d}.webp"
                for idx in range(1, len(self.individual_sections) + 1)
            }

        return {f"{image_path.stem}.webp"}

    @staticmethod
    def _crop_section(image: Image.Image, section: Section) -> Image.Image:
        left = section.x
        top = section.y
        right = section.x + section.width
        bottom = section.y + section.height
        return image.crop((left, top, right, bottom))

    @staticmethod
    def _join_horizontally(images: list[Image.Image]) -> Image.Image:
        if not images:
            raise ValueError("No images to join")

        total_width = sum(img.width for img in images)
        max_height = max(img.height for img in images)
        canvas = Image.new("RGB", (total_width, max_height), color=(0, 0, 0))

        x_offset = 0
        for img in images:
            canvas.paste(img, (x_offset, 0))
            x_offset += img.width

        return canvas


def main() -> None:
    try:
        root = tk.Tk()
        ImageLabellerApp(root)
        root.mainloop()
    except Exception as exc:  # noqa: BLE001
        messagebox.showerror("Startup error", str(exc))


if __name__ == "__main__":
    main()