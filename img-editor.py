import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox


class ImageRegionSelectorApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Region Selector")

        self.image = None
        self.display_image = None
        self.overlay = None
        self.tk_image = None

        self.internal_tolerance = 10
        self.max_internal_tolerance = 20
        self.outline_thickness = 2

        # Canvas setup
        self.canvas = tk.Canvas(master, width=800, height=600, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        frame = tk.Frame(master)
        frame.pack(fill=tk.X)

        tk.Button(frame, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(frame, text="Clear Overlay", command=self.clear_overlay).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(frame, text="Save Overlay", command=self.save_overlay).pack(side=tk.LEFT, padx=5, pady=5)

        # Tolerance controls
        self.tol_label = tk.Label(frame, text=f"Tolerance: {self.internal_tolerance}")
        self.tol_label.pack(side=tk.LEFT, padx=(20, 5))

        self.tol_slider = tk.Scale(frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                   command=self.update_tolerance_from_slider, length=150)
        self.tol_slider.set(int(self.internal_tolerance / self.max_internal_tolerance * 100))
        self.tol_slider.pack(side=tk.LEFT, padx=5)

        tk.Label(frame, text="Manual:").pack(side=tk.LEFT, padx=(10, 2))
        self.tol_entry = tk.Entry(frame, width=4)
        self.tol_entry.insert(0, str(self.internal_tolerance))
        self.tol_entry.bind("<Return>", self.update_tolerance_from_entry)
        self.tol_entry.pack(side=tk.LEFT, padx=2)

        self.canvas.bind("<Button-1>", self.on_click)

    def update_tolerance_from_slider(self, val):
        self.internal_tolerance = round(float(val) / 100 * self.max_internal_tolerance, 2)
        self.tol_label.config(text=f"Tolerance: {self.internal_tolerance:.2f}")
        self.tol_entry.delete(0, tk.END)
        self.tol_entry.insert(0, str(self.internal_tolerance))

    def update_tolerance_from_entry(self, event):
        try:
            val = float(self.tol_entry.get())
        except ValueError:
            return
        val = max(0, min(self.max_internal_tolerance, val))
        self.internal_tolerance = val
        slider_val = int(val / self.max_internal_tolerance * 100)
        self.tol_slider.set(slider_val)
        self.tol_label.config(text=f"Tolerance: {val:.2f}")

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif"),
                       ("All Files", "*.*")]
        )
        if not file_path:
            return

        img = cv2.imread(file_path)
        if img is None:
            messagebox.showerror("Error", f"Failed to load image from:\n{file_path}")
            return

        self.image = cv2.bilateralFilter(img, d=5, sigmaColor=30, sigmaSpace=30)
        self.display_image = self.image.copy()
        self.overlay = np.zeros_like(self.image)
        self.show_image(self.display_image)
        messagebox.showinfo("Loaded", f"Loaded image:\n{file_path}")

    def flood_fill_region(self, x, y):
        h, w = self.image.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        diff = (self.internal_tolerance, self.internal_tolerance, self.internal_tolerance)
        flood_img = self.image.copy()
        fill_color = (0, 0, 255)
        flags = cv2.FLOODFILL_MASK_ONLY | (4 << 8)

        try:
            cv2.floodFill(flood_img, mask, (x, y), fill_color, diff, diff, flags)
        except Exception:
            return None

        region_mask = mask[1:h + 1, 1:w + 1]
        region_mask = (region_mask > 0).astype(np.uint8) * 255

        kernel = np.ones((3, 3), np.uint8)
        region_mask = cv2.morphologyEx(region_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return region_mask

    def on_click(self, event):
        if self.image is None:
            messagebox.showinfo("Info", "Load an image first.")
            return

        x = int(event.x * self.image.shape[1] / self.canvas.winfo_width())
        y = int(event.y * self.image.shape[0] / self.canvas.winfo_height())

        mask = self.flood_fill_region(x, y)
        if mask is None:
            return

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours is None or len(contours) == 0:
            return

        # ✅ Draw contours directly on a copy of the image for strong visibility
        combined = self.display_image.copy()
        outline_color = (0, 0, 0)
        for i, contour in enumerate(contours):
            cv2.drawContours(combined, contours, i, outline_color,
                             thickness=self.outline_thickness, hierarchy=hierarchy)

        # Store overlay separately (still black)
        for i, contour in enumerate(contours):
            cv2.drawContours(self.overlay, contours, i, outline_color,
                             thickness=self.outline_thickness, hierarchy=hierarchy)

        self.show_image(combined)

    def clear_overlay(self):
        if self.image is None:
            return
        self.overlay = np.zeros_like(self.image)
        self.show_image(self.image.copy())

    def show_image(self, image):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil = img_pil.resize((800, 600))
        self.tk_image = ImageTk.PhotoImage(img_pil)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

    def save_overlay(self):
        if self.overlay is None:
            return
        save_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG Image", "*.png")])
        if not save_path:
            return
        cv2.imwrite(save_path, self.overlay)
        messagebox.showinfo("Saved", f"Overlay saved to {save_path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageRegionSelectorApp(root)
    root.mainloop()
