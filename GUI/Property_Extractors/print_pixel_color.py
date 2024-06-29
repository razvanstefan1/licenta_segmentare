import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Pixel Color Viewer")

        self.open_button = tk.Button(root, text="Open BMP Image", command=self.open_image)
        self.open_button.pack(pady=10)

        self.image_label = tk.Label(root)
        self.image_label.pack()

        self.image = None

    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("BMP files", "*.bmp")])
        if file_path:
            self.image = Image.open(file_path)
            photo = ImageTk.PhotoImage(self.image)
            self.image_label.config(image=photo)
            self.image_label.image = photo
            self.image_label.bind("<Button-1>", self.print_pixel_color)

    def print_pixel_color(self, event):
        if self.image:
            x, y = event.x, event.y
            # Convert to the image coordinate system
            image_x = int(x * self.image.width / self.image_label.winfo_width())
            image_y = int(y * self.image.height / self.image_label.winfo_height())
            pixel = self.image.getpixel((image_x, image_y))
            print(f"Pixel color at ({image_x}, {image_y}): {pixel}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()
