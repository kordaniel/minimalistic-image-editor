#!/usr/bin/env python3
"""
Simple Image Text Editor

Dependencies:
    pip install opencv-python pillow numpy
"""

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, colorchooser
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
import os
from collections import deque


class ImageTextEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Text Editor")

        # Images and masks (original resolution)
        self.image = None
        self.display_image = None
        self.current_selection = None  # single-channel mask 0/255
        self.overlay = None

        # zoom/pan
        self.zoom_factor = 1.0
        self.min_zoom = 0.2
        self.max_zoom = 8.0
        self.zoom_step = 1.15

        # selection rendering
        self.internal_tolerance = 0.6
        self.max_internal_tolerance = 1.8
        self.outline_thickness = 3

        # pan state for shift-drag
        self._pan_last = None

        # Default margin for deselection expansion
        self.deselect_margin = 3
        self.fill_color = (255, 255, 255)
        self.picking_color = False

        # build UI
        self._build_ui()
        self._bind_events()


    def _build_ui(self):
        top_frame = tk.Frame(self.root)
        top_frame.pack(fill=tk.X, padx=4, pady=4)

        tk.Button(top_frame, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=3)
        tk.Button(top_frame, text="Clear Selection", command=self.clear_selection).pack(side=tk.LEFT, padx=3)
        tk.Button(top_frame, text="Add Text", command=self.add_text).pack(side=tk.LEFT, padx=3)
        tk.Button(top_frame, text="Save Edited Image", command=self.save_image).pack(side=tk.LEFT, padx=3)

        tk.Label(top_frame, text="Tolerance:").pack(side=tk.LEFT, padx=(10,3))
        self.tol_slider = tk.Scale(top_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                   command=self._tol_from_slider, length=160)
        self.tol_slider.set(int(self.internal_tolerance/self.max_internal_tolerance*100))
        self.tol_slider.pack(side=tk.LEFT, padx=3)

        tk.Label(top_frame, text="Manual:").pack(side=tk.LEFT, padx=(8,2))
        self.tol_entry = tk.Entry(top_frame, width=6)
        self.tol_entry.insert(0, f"{self.internal_tolerance:.2f}")
        self.tol_entry.bind("<Return>", self._tol_from_entry)
        self.tol_entry.pack(side=tk.LEFT, padx=2)

        self.canvas = tk.Canvas(self.root, bg="grey", width=900, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.tk_image = None

        # Deselect margin slider
        tk.Label(top_frame, text="Deselect Margin:").pack(anchor="w")
        self.margin_slider = tk.Scale(top_frame, from_=0, to=20, orient="horizontal",
                                      command=self._update_deselect_margin)
        self.margin_slider.set(self.deselect_margin)
        self.margin_slider.pack(fill="x", pady=(0, 8))

        # Fill color picker
        self.pick_color_btn = tk.Button(top_frame, text="Pick Fill Color", command=self.enable_color_pick)
        self.pick_color_btn.pack(fill="x", pady=(5, 0))

        # Fill color preview
        self.color_preview = tk.Label(top_frame, bg=self._rgb_to_hex(self.fill_color), width=10, relief="sunken")
        self.color_preview.pack(pady=(0, 10))

        # Fill button
        tk.Button(top_frame, text="Fill Selection", command=self.fill_selection).pack(fill="x", pady=(5, 10))

    def _update_deselect_margin(self, val):
        self.deselect_margin = float(val)

    def _rgb_to_hex(self, rgb):
        return "#%02x%02x%02x" % tuple(int(c) for c in rgb)

    def enable_color_pick(self):
        """Enable color picking mode."""
        self.picking_color = True
        self.canvas.config(cursor="cross")

    def _on_left_click(self, event):
        if self.image is None:
            return
        img_x, img_y = self._canvas_to_image_coords(event)

        if self.picking_color:
            # pick color
            bgr = self.image[img_y, img_x].tolist()
            self.fill_color = tuple(int(c) for c in bgr[::-1])  # convert BGR → RGB
            self.color_preview.config(bg=self._rgb_to_hex(self.fill_color))
            self.picking_color = False
            self.canvas.config(cursor="")
            return

        self.select_single_polygon(img_x, img_y, mode="outer")


    def _bind_events(self):
        # focus so wheel events delivered
        self.canvas.bind("<Enter>", lambda e: self.canvas.focus_set())

        # wheel: cross-platform
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)    # Windows/macOS
        self.canvas.bind("<Button-4>", self._on_mousewheel)      # Linux up
        self.canvas.bind("<Button-5>", self._on_mousewheel)      # Linux down

        # panning: middle button and Shift+Left-drag
        self.canvas.bind("<ButtonPress-2>", self._start_pan)
        self.canvas.bind("<B2-Motion>", self._do_pan)
        # Shift + left drag alternative for pan
        self.canvas.bind("<Shift-ButtonPress-1>", self._start_pan_shift)
        self.canvas.bind("<Shift-B1-Motion>", self._do_pan_shift)
        self.canvas.bind("<Shift-ButtonRelease-1>", self._end_pan_shift)

        # selection and subtract
        self.canvas.bind("<Button-1>", self._on_left_click)   # left click: single polygon select
        self.canvas.bind("<Button-3>", self._on_right_click)  # right click: subtract sub-area inside selection

        # redraw on resize
        self.root.bind("<Configure>", lambda e: self._redraw_canvas())

    # --------------------------
    # Load/Save
    # --------------------------
    def load_image(self):
        path = filedialog.askopenfilename(title="Open image",
                                          filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif"), ("All Files", "*.*")])
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", f"Could not load image: {path}")
            return
        self.image = cv2.bilateralFilter(img, d=5, sigmaColor=30, sigmaSpace=30)
        self.display_image = self.image.copy()
        h, w = self.image.shape[:2]
        self.current_selection = np.zeros((h, w), dtype=np.uint8)
        self.overlay = np.zeros_like(self.image)
        self.zoom_factor = 1.0
        self.canvas.delete("all")
        self._redraw_canvas()
        messagebox.showinfo("Loaded", f"Loaded: {os.path.basename(path)}")

    def save_image(self):
        if self.display_image is None:
            messagebox.showinfo("Info", "No image to save.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if not path:
            return
        cv2.imwrite(path, self.display_image)
        messagebox.showinfo("Saved", f"Saved edited image to:\n{path}")

    # --------------------------
    # Tolerance UI helpers
    # --------------------------
    def _tol_from_slider(self, val):
        try:
            v = float(val)
        except Exception:
            v = 0.0
        self.internal_tolerance = round(v/100.0 * self.max_internal_tolerance, 3)
        self.tol_entry.delete(0, tk.END)
        self.tol_entry.insert(0, f"{self.internal_tolerance:.3f}")

    def _tol_from_entry(self, event):
        try:
            v = float(self.tol_entry.get())
        except Exception:
            v = self.internal_tolerance
        v = max(0.0, min(self.max_internal_tolerance, v))
        self.internal_tolerance = v
        slider_val = int(round(v/self.max_internal_tolerance * 100))
        self.tol_slider.set(slider_val)
        self.tol_entry.delete(0, tk.END)
        self.tol_entry.insert(0, f"{self.internal_tolerance:.3f}")

    # --------------------------
    # Coordinate mapping
    # --------------------------
    def _canvas_to_image_coords(self, event):
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        img_x = int(round(canvas_x / self.zoom_factor))
        img_y = int(round(canvas_y / self.zoom_factor))
        if self.image is None:
            return img_x, img_y
        h, w = self.image.shape[:2]
        img_x = max(0, min(w-1, img_x))
        img_y = max(0, min(h-1, img_y))
        return img_x, img_y

    # --------------------------
    # Zoom & Pan
    # --------------------------
    def _on_mousewheel(self, event):
        if self.image is None:
            return
        delta = 0
        if hasattr(event, 'delta') and event.delta != 0:
            delta = event.delta
        else:
            if hasattr(event, 'num'):
                if event.num == 4:
                    delta = 120
                elif event.num == 5:
                    delta = -120
        if delta == 0:
            return
        factor = self.zoom_step if delta > 0 else 1.0/self.zoom_step
        new_zoom = self.zoom_factor * factor
        new_zoom = max(self.min_zoom, min(self.max_zoom, new_zoom))
        if abs(new_zoom - self.zoom_factor) < 1e-6:
            return

        # zoom around cursor: preserve image point under cursor
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        img_x_before = canvas_x / self.zoom_factor
        img_y_before = canvas_y / self.zoom_factor

        self.zoom_factor = new_zoom
        self._redraw_canvas()

        canvas_x_after = img_x_before * self.zoom_factor
        canvas_y_after = img_y_before * self.zoom_factor
        dx = canvas_x_after - canvas_x
        dy = canvas_y_after - canvas_y

        # scroll so the same image point stays under cursor
        # convert dx/dy into scroll units: use canvas.xview_moveto / relative adjustments
        # compute new left-top in fraction:
        bbox = self.canvas.bbox("all")
        if bbox:
            total_w = bbox[2]
            total_h = bbox[3]
            if total_w > 0:
                self.canvas.xview_moveto(max(0, (self.canvas.canvasx(0) + dx) / total_w))
            if total_h > 0:
                self.canvas.yview_moveto(max(0, (self.canvas.canvasy(0) + dy) / total_h))

    def _start_pan(self, event):
        # middle-button pan
        self.canvas.scan_mark(event.x, event.y)

    def _do_pan(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def _start_pan_shift(self, event):
        # Shift + left button pan: store last pos
        self._pan_last = (event.x, event.y)
        self.canvas.scan_mark(event.x, event.y)

    def _do_pan_shift(self, event):
        # simulate scan_dragto
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def _end_pan_shift(self, event):
        self._pan_last = None

    # --------------------------
    # Rendering
    # --------------------------
    def _redraw_canvas(self):
        if self.display_image is None:
            self.canvas.delete("all")
            return

        img = self.display_image.copy()
        h, w = img.shape[:2]
        new_w = max(1, int(round(w * self.zoom_factor)))
        new_h = max(1, int(round(h * self.zoom_factor)))
        zoomed = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # draw selection outlines scaled
        # draw selection outlines scaled
        if self.current_selection is not None:
            cnts, hierarchy = cv2.findContours(self.current_selection.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            if cnts is not None and len(cnts) > 0 and hierarchy is not None:
                for i, c in enumerate(cnts):
                    c = (c * self.zoom_factor).astype(np.int32)
                    # black for outer, white for inner hole outlines
                    color = (0, 0, 0) if hierarchy[0][i][3] < 0 else (255, 255, 255)
                    cv2.drawContours(zoomed, [c], -1, color, thickness=max(1, int(round(self.outline_thickness*self.zoom_factor))))


        img_rgb = cv2.cvtColor(zoomed, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)
        self.tk_image = ImageTk.PhotoImage(pil)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.canvas.config(scrollregion=(0, 0, new_w, new_h))

    def _on_right_click(self, event):
        """
        Right-click: deselect a sub-region inside the selection.
        The deselection region is slightly expanded to include neighboring pixels.
        """
        if self.image is None or self.current_selection is None:
            return
        img_x, img_y = self._canvas_to_image_coords(event)
        if self.current_selection[img_y, img_x] == 0:
            return
        self.select_single_polygon(img_x, img_y, mode="inner")


    def select_single_polygon(self, x, y, mode="outer"):
        """
        mode="outer": create a new selection mask.
        mode="inner": subtract a slightly expanded region (hole) from current selection.
        """
        h, w = self.image.shape[:2]
        ff_mask = np.zeros((h+2, w+2), np.uint8)
        flood_img = self.image.copy()
        diff = (int(self.internal_tolerance),) * 3
        flags = cv2.FLOODFILL_MASK_ONLY | (4 << 8)

        try:
            cv2.floodFill(flood_img, ff_mask, (x, y), (0, 0, 255), diff, diff, flags)
        except Exception as e:
            print("Flood fill failed:", e)
            return

        region = (ff_mask[1:h+1, 1:w+1] > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return

        chosen = None
        for c in contours:
            if cv2.pointPolygonTest(c, (x, y), False) >= 0:
                chosen = c
                break
        if chosen is None:
            chosen = max(contours, key=cv2.contourArea)

        # Base mask of the selected region
        region_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(region_mask, [chosen], -1, 255, thickness=cv2.FILLED)

        if mode == "outer":
            # Replace the current selection completely
            self.current_selection = region_mask

        elif mode == "inner":
            if self.current_selection is None:
                self.current_selection = np.zeros((h, w), dtype=np.uint8)

            expansion_radius = max(0, round(self.deselect_margin))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                               (expansion_radius * 2 + 1, expansion_radius * 2 + 1))
            expanded_mask = cv2.dilate(region_mask, kernel, iterations=1)

            self.current_selection[expanded_mask > 0] = 0

        # Clean up small artifacts
        kernel = np.ones((3, 3), np.uint8)
        self.current_selection = cv2.morphologyEx(self.current_selection, cv2.MORPH_OPEN, kernel)
        self._redraw_canvas()


    def subtract_area_inside_selection(self, x, y):
        """
        Robustly select a contiguous sub-area inside current_selection using the
        same color-tolerance rule as the initial selection, and subtract it.
        This uses a BFS (queue) and checks color difference per neighbor.
        """
        if self.current_selection is None:
            return
        h, w = self.current_selection.shape
        # Sanity clamp
        if x < 0 or y < 0 or x >= w or y >= h:
            return

        # If clicked pixel is not part of current selection, nothing to do
        if self.current_selection[y, x] == 0:
            return

        # Seed color at clicked pixel (BGR)
        seed_color = self.image[y, x].astype(int)

        # tolerance (we interpret internal tolerance as max per-channel difference)
        tol = int(round(self.internal_tolerance))
        if tol < 0:
            tol = 0

        # BFS queue
        q = deque()
        q.append((x, y))

        # visited mask to avoid revisiting
        visited = np.zeros((h, w), dtype=np.uint8)
        remove_mask = np.zeros((h, w), dtype=np.uint8)

        # 4-connectivity neighbors
        neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1))

        while q:
            cx, cy = q.popleft()
            if visited[cy, cx]:
                continue
            visited[cy, cx] = 1

            # must be within the current selection
            if self.current_selection[cy, cx] == 0:
                continue

            # color check: all channels difference <= tol
            pix = self.image[cy, cx].astype(int)
            # using max absolute channel difference (robust and fast)
            if np.max(np.abs(pix - seed_color)) <= tol:
                remove_mask[cy, cx] = 255
                # push neighbors
                for dx, dy in neighbors:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                        # only enqueue if inside current selection (speeds up)
                        if self.current_selection[ny, nx] != 0:
                            q.append((nx, ny))

        # If nothing found, do nothing
        if remove_mask.sum() == 0:
            return

        # Subtract found region from current_selection
        self.current_selection[remove_mask > 0] = 0

        # Small cleanup to remove noise and keep selection smooth
        kernel = np.ones((3, 3), np.uint8)
        self.current_selection = cv2.morphologyEx(self.current_selection, cv2.MORPH_OPEN, kernel, iterations=1)
        self.current_selection = cv2.morphologyEx(self.current_selection, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Redraw outlines
        self._redraw_canvas()


    def _flood_fill_binary_mask(self, mask, sx, sy):
        h, w = mask.shape
        out = np.zeros_like(mask)
        if mask[sy, sx] == 0:
            return out
        stack = [(sx, sy)]
        visited = np.zeros_like(mask, dtype=np.uint8)
        dirs = ((1,0),(-1,0),(0,1),(0,-1))
        while stack:
            cx, cy = stack.pop()
            if cx < 0 or cy < 0 or cx >= w or cy >= h:
                continue
            if visited[cy, cx]:
                continue
            visited[cy, cx] = 1
            if mask[cy, cx] == 0:
                continue
            out[cy, cx] = 255
            for dx, dy in dirs:
                nx, ny = cx+dx, cy+dy
                if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                    stack.append((nx, ny))
        return out

    # --------------------------
    # Fill selection with dominant color
    # --------------------------
    def fill_selection(self):
        if self.current_selection is None or self.display_image is None:
            messagebox.showinfo("Info", "No selection to fill.")
            return
        mask_bool = self.current_selection > 0
        if mask_bool.sum() == 0:
            messagebox.showinfo("Info", "Selection is empty.")
            return

        bgr_color = tuple(int(c) for c in self.fill_color[::-1])  # RGB → BGR
        filled_image = self.image.copy()
        filled_image[self.current_selection > 0] = bgr_color

        self.display_image = filled_image

        #pixels = self.display_image[mask_bool]
        #mean_col = tuple(np.round(pixels.mean(axis=0)).astype(int).tolist())
        #self.display_image[mask_bool] = mean_col
        self._redraw_canvas()

    # --------------------------
    # Add text (auto-fit & color)
    # --------------------------
    def add_text(self):
        if self.current_selection is None or self.display_image is None:
            messagebox.showinfo("Info", "No selection to add text.")
            return
        text = simpledialog.askstring("Add Text", "Enter replacement text:")
        if not text:
            return
        col = colorchooser.askcolor(title="Choose text color")
        if col is None or col[0] is None:
            text_color = (0,0,0)
        else:
            rgb = tuple(int(round(c)) for c in col[0])
            text_color = rgb

        ys, xs = np.where(self.current_selection > 0)
        if len(xs) == 0:
            messagebox.showinfo("Info", "Selection is empty.")
            return
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        box_w = x1 - x0 + 1
        box_h = y1 - y0 + 1

        pil_img = Image.fromarray(cv2.cvtColor(self.display_image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        def get_font(size):
            for fname in ("arial.ttf", "DejaVuSans.ttf"):
                try:
                    return ImageFont.truetype(fname, size=size)
                except Exception:
                    continue
            return ImageFont.load_default()

        max_font = min(box_h, 400)
        font_size = min(max_font, 200)
        wrapped_lines = [text]
        font = get_font(font_size)
        while font_size > 6:
            font = get_font(font_size)
            w, h = draw.textsize(text, font=font)
            if w <= box_w * 0.95 and h <= box_h * 0.95:
                wrapped_lines = [text]
                break
            words = text.split()
            if len(words) > 1:
                lines = []
                cur = words[0]
                for wword in words[1:]:
                    test = cur + " " + wword
                    tw, th = draw.textsize(test, font=font)
                    if tw <= box_w * 0.95:
                        cur = test
                    else:
                        lines.append(cur)
                        cur = wword
                lines.append(cur)
                total_h = sum(draw.textsize(line, font=font)[1] for line in lines)
                if total_h <= box_h * 0.95:
                    wrapped_lines = lines
                    break
            font_size -= 2

        line_sizes = [draw.textsize(line, font=font) for line in wrapped_lines]
        total_h = sum(h for (_, h) in line_sizes)
        start_y = y0 + (box_h - total_h) // 2

        y = start_y
        for i, line in enumerate(wrapped_lines):
            tw, th = line_sizes[i]
            x = x0 + (box_w - tw) // 2
            draw.text((x, y), line, fill=text_color, font=font)
            y += th

        self.display_image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        self._redraw_canvas()

    # --------------------------
    # Utilities
    # --------------------------
    def clear_selection(self):
        if self.image is None:
            return
        h, w = self.image.shape[:2]
        self.current_selection = np.zeros((h, w), dtype=np.uint8)
        self._redraw_canvas()


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageTextEditor(root)
    root.mainloop()
