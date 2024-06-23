#
# import tkinter as tk
# from tkinter import filedialog, ttk
# from PIL import Image, ImageTk
# import cv2
# import numpy as np
# from collections import Counter
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
#
#
# class ImageApp:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("Image Segmentation Viewer")
#         self.root.geometry("1200x800")  # Set the window size
#
#         # Frame for the buttons
#         self.button_frame = tk.Frame(root)
#         self.button_frame.grid(row=0, column=0, padx=10, pady=10, sticky="n")
#
#         # Add buttons to the button frame
#         self.open_button = ttk.Button(self.button_frame, text="Open Image", command=self.open_image)
#         self.open_button.pack(pady=5)
#
#         self.calculate_button = ttk.Button(self.button_frame, text="Calculate Histogram", command=self.calculate_histogram)
#         self.calculate_button.pack(pady=5)
#
#         self.color_button = ttk.Button(self.button_frame, text="Color Predictions", command=self.ColorPredictions)
#         self.color_button.pack(pady=5)
#
#         # Frame for the image
#         self.image_frame = tk.Frame(root)
#         self.image_frame.grid(row=0, column=1, padx=10, pady=10)
#
#         # Add a label to display the image
#         self.image_label = tk.Label(self.image_frame)
#         self.image_label.pack()
#         self.image_label.bind("<Button-1>", self.print_pixel_color)  # Bind left mouse click event
#
#         # Frame for the histogram
#         self.histogram_frame = tk.Frame(root)
#         self.histogram_frame.grid(row=0, column=2, padx=10, pady=10)
#
#     def open_image(self):
#         # Open a file dialog to select an image file
#         file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.gif;*.bmp")])
#         if file_path:
#             # Open the image file with PIL
#             self.image = Image.open(file_path)
#             # Convert the image to a format Tkinter can use
#             photo = ImageTk.PhotoImage(self.image)
#             # Update the label with the new image
#             self.image_label.config(image=photo)
#             self.image_label.image = photo
#
#             # Load the image with OpenCV
#             self.cv_image = cv2.imread(file_path)
#             self.cv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
#
#     def calculate_histogram(self):
#         if hasattr(self, 'cv_image'):
#             # Flatten the image array and convert to a list of tuples
#             pixels = [tuple(pixel) for pixel in self.cv_image.reshape(-1, 3)]
#             # Count the occurrences of each color
#             color_count = Counter(pixels)
#
#             # Print out the RGB values and their counts
#             for color, count in color_count.items():
#                 print(f"Color: {color}, Count: {count}")
#
#             # Define the colors we are interested in
#             predefined_colors = {
#                 (0, 0, 0): 'background',
#                 (1, 1, 1): 'capillary',
#                 (2, 2, 2): 'artery',
#                 (3, 3, 3): 'vein',
#                 (4, 4, 4): 'FAZ'
#             }
#
#             # Count the occurrences of the predefined colors
#             color_histogram = {color: 0 for color in predefined_colors.values()}
#             for color, count in color_count.items():
#                 if color in predefined_colors:
#                     color_histogram[predefined_colors[color]] += count
#
#             # Calculate the total number of pixels
#             total_pixels = self.cv_image.shape[0] * self.cv_image.shape[1]
#             # Calculate the area per pixel (since the image is 6mm x 6mm)
#             area_per_pixel = (6 * 6) / total_pixels
#
#             # Calculate the surface area for each zone
#             surface_areas = {color: count * area_per_pixel for color, count in color_histogram.items()}
#
#             # Display the surface area
#             for color, area in surface_areas.items():
#                 print(f"{color.capitalize()} area: {area:.2f} mm^2")
#
#             # Plot histogram
#             fig, ax = plt.subplots()
#             ax.set_facecolor('lightgreen')  # Set the background color of the plot area
#             ax.bar(color_histogram.keys(), color_histogram.values(), color=['black', 'white', 'purple', 'blue', 'red'])
#             ax.set_xlabel('Colors')
#             ax.set_ylabel('Pixel Count')
#             ax.set_title('Color Histogram')
#
#             # Display the histogram in the histogram frame
#             for widget in self.histogram_frame.winfo_children():
#                 widget.destroy()
#             canvas = FigureCanvasTkAgg(fig, master=self.histogram_frame)
#             canvas.draw()
#             canvas.get_tk_widget().pack()
#
#         else:
#             print("No image loaded!")
#
#     def print_pixel_color(self, event):
#         if hasattr(self, 'cv_image'):
#             # Get the coordinates of the clicked point
#             x, y = event.x, event.y
#             # Convert coordinates if the image was resized in the label
#             width, height = self.image.size
#             label_width, label_height = self.image_label.winfo_width(), self.image_label.winfo_height()
#             scale_x, scale_y = width / label_width, height / label_height
#             x = int(x * scale_x)
#             y = int(y * scale_y)
#
#             # Get the color of the pixel at the clicked coordinates
#             color = self.cv_image[y, x]
#             print(f"Clicked at ({x}, {y}), Color: {color}")
#
#     def ColorPredictions(self):  #imaginile sunt 400 pe 400 si 6mm pe 6mm.
#         if hasattr(self, 'cv_image'):
#             # Define the color mappings
#             color_map = {
#                 (0, 0, 0): (0, 0, 0),  # Fundal, negru
#                 (1, 1, 1): (222, 222, 186),  # Capilare, alb galbui
#                 (2, 2, 2): (153, 0, 153),  # Artere, mov
#                 (3, 3, 3): (0, 102, 255),  # Vene, albastru
#                 (4, 4, 4): (179,0,0)  # FAZ, galben ish
#             }
#
#             # Create an empty image for the colorized output
#             colorized_image = np.zeros((self.cv_image.shape[0], self.cv_image.shape[1], 3), dtype=np.uint8)
#
#             # Map the original pixel values to the new colors
#             for i in range(self.cv_image.shape[0]):
#                 for j in range(self.cv_image.shape[1]):
#                     pixel_value = tuple(self.cv_image[i, j])
#                     if pixel_value in color_map:
#                         colorized_image[i, j] = color_map[pixel_value]
#
#             # Display the colorized image using Tkinter
#             colorized_image_pil = Image.fromarray(colorized_image)
#             photo = ImageTk.PhotoImage(colorized_image_pil)
#             self.image_label.config(image=photo)
#             self.image_label.image = photo
#             print(self.image.size)
#
#
# if __name__ == "__main__":
#     root = tk.Tk()
#     app = ImageApp(root)
#     root.mainloop()





import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Segmentation Viewer")
        self.root.geometry("1600x800")  # Set the window size

        # Frame for the buttons
        self.button_frame = tk.Frame(root)
        self.button_frame.grid(row=0, column=0, padx=10, pady=10, sticky="n")

        # Add buttons to the button frame
        self.open_button = ttk.Button(self.button_frame, text="Open Image", command=self.open_image)
        self.open_button.pack(pady=5)

        self.calculate_button = ttk.Button(self.button_frame, text="Calculate Histogram", command=self.calculate_histogram)
        self.calculate_button.pack(pady=5)

        self.color_button = ttk.Button(self.button_frame, text="Color Predictions", command=self.ColorPredictions)
        self.color_button.pack(pady=5)

        # Frame for the image
        self.image_frame = tk.Frame(root)
        self.image_frame.grid(row=0, column=1, padx=10, pady=10)

        # Add a label to display the image
        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack()
        self.image_label.bind("<Button-1>", self.print_pixel_color)  # Bind left mouse click event

        # Frame for the histogram
        self.histogram_frame = tk.Frame(root)
        self.histogram_frame.grid(row=0, column=2, padx=10, pady=10)

    def open_image(self):
        # Open a file dialog to select an image file
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.gif;*.bmp")])
        if file_path:
            # Open the image file with PIL
            self.image = Image.open(file_path)
            # Convert the image to a format Tkinter can use
            photo = ImageTk.PhotoImage(self.image)
            # Update the label with the new image
            self.image_label.config(image=photo)
            self.image_label.image = photo

            # Load the image with OpenCV
            self.cv_image = cv2.imread(file_path)
            self.cv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)

    def calculate_histogram(self):
        if hasattr(self, 'cv_image'):
            # Flatten the image array and convert to a list of tuples
            pixels = [tuple(pixel) for pixel in self.cv_image.reshape(-1, 3)]
            # Count the occurrences of each color
            color_count = Counter(pixels)

            # Print out the RGB values and their counts
            for color, count in color_count.items():
                print(f"Color: {color}, Count: {count}")

            # Define the colors we are interested in
            predefined_colors = {
                (0, 0, 0): 'background',
                (1, 1, 1): 'capillary',
                (2, 2, 2): 'artery',
                (3, 3, 3): 'vein',
                (4, 4, 4): 'FAZ'
            }

            # Count the occurrences of the predefined colors
            color_histogram = {color: 0 for color in predefined_colors.values()}
            for color, count in color_count.items():
                if color in predefined_colors:
                    color_histogram[predefined_colors[color]] += count

            # Calculate the total number of pixels
            total_pixels = self.cv_image.shape[0] * self.cv_image.shape[1]
            # Calculate the area per pixel (since the image is 6mm x 6mm)
            area_per_pixel = (6 * 6) / total_pixels

            # Calculate the surface area for each zone
            surface_areas = {color: count * area_per_pixel for color, count in color_histogram.items()}

            # Plot histogram
            fig, ax = plt.subplots()
            ax.set_facecolor('lightgreen')  # Set the background color of the plot area
            ax.bar(color_histogram.keys(), color_histogram.values(), color=['black', 'white', 'purple', 'blue', 'red'])
            ax.set_xlabel('Colors')
            ax.set_ylabel('Pixel Count')
            ax.set_title('Color Histogram')

            # Display the histogram in the histogram frame
            for widget in self.histogram_frame.winfo_children():
                widget.destroy()
            canvas = FigureCanvasTkAgg(fig, master=self.histogram_frame)
            canvas.draw()
            canvas.get_tk_widget().pack()

            # Recreate the area frame and display the surface area
            self.area_frame = tk.Frame(self.histogram_frame)
            self.area_frame.pack(pady=10)
            for color, area in surface_areas.items():
                label = tk.Label(self.area_frame, text=f"{color.capitalize()} area: {area:.2f} mm^2")
                label.pack()

        else:
            print("No image loaded!")

    def print_pixel_color(self, event):
        if hasattr(self, 'cv_image'):
            # Get the coordinates of the clicked point
            x, y = event.x, event.y
            # Convert coordinates if the image was resized in the label
            width, height = self.image.size
            label_width, label_height = self.image_label.winfo_width(), self.image_label.winfo_height()
            scale_x, scale_y = width / label_width, height / label_height
            x = int(x * scale_x)
            y = int(y * scale_y)

            # Get the color of the pixel at the clicked coordinates
            color = self.cv_image[y, x]
            print(f"Clicked at ({x}, {y}), Color: {color}")

    def ColorPredictions(self):  #imaginile sunt 400 pe 400 si 6mm pe 6mm.
        if hasattr(self, 'cv_image'):
            # Define the color mappings
            color_map = {
                (0, 0, 0): (0, 0, 0),  # Fundal, negru
                (1, 1, 1): (222, 222, 186),  # Capilare, alb galbui
                (2, 2, 2): (153, 0, 153),  # Artere, mov
                (3, 3, 3): (0, 102, 255),  # Vene, albastru
                (4, 4, 4): (179,0,0)  # FAZ, galben ish
            }

            # Create an empty image for the colorized output
            colorized_image = np.zeros((self.cv_image.shape[0], self.cv_image.shape[1], 3), dtype=np.uint8)

            # Map the original pixel values to the new colors
            for i in range(self.cv_image.shape[0]):
                for j in range(self.cv_image.shape[1]):
                    pixel_value = tuple(self.cv_image[i, j])
                    if pixel_value in color_map:
                        colorized_image[i, j] = color_map[pixel_value]

            # Display the colorized image using Tkinter
            colorized_image_pil = Image.fromarray(colorized_image)
            photo = ImageTk.PhotoImage(colorized_image_pil)
            self.image_label.config(image=photo)
            self.image_label.image = photo
            print(self.image.size)


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()
