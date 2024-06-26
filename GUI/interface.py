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
        self.image_frame.grid(row=0, column=1, padx=10, pady=10, sticky="n")

        # Add a label to display the image
        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack()
        self.image_label.bind("<Button-1>", self.print_pixel_color)  # Bind left mouse click event

        # Frame for the histogram
        self.histogram_frame = tk.Frame(root)
        self.histogram_frame.grid(row=0, column=2, padx=10, pady=10, sticky="n")

        # Frame for the area information
        self.area_frame = tk.Frame(root)
        self.area_frame.grid(row=0, column=3, padx=10, pady=10, sticky="n")

        #adaugam frame pentru checkboxes
        self.checkbox_frame = tk.Frame(self.button_frame)
        self.checkbox_frame.pack(pady=5)

        #creem checkboxes
        # Variables to store checkbox states
        self.show_capillary = tk.IntVar(value=1)
        self.show_vein = tk.IntVar(value=1)
        self.show_artery = tk.IntVar(value=1)
        self.show_faz = tk.IntVar(value=1)

        # Create checkboxes
        self.capillary_checkbox = ttk.Checkbutton(self.checkbox_frame, text="Capillary", variable=self.show_capillary)
        self.capillary_checkbox.pack(anchor="w")

        self.vein_checkbox = ttk.Checkbutton(self.checkbox_frame, text="Vein", variable=self.show_vein)
        self.vein_checkbox.pack(anchor="w")

        self.artery_checkbox = ttk.Checkbutton(self.checkbox_frame, text="Artery", variable=self.show_artery)
        self.artery_checkbox.pack(anchor="w")

        self.faz_checkbox = ttk.Checkbutton(self.checkbox_frame, text="FAZ", variable=self.show_faz)
        self.faz_checkbox.pack(anchor="w")

        # Bind the checkboxes to the HideOrShowComponents method
        # self.show_capillary.trace_add("write", lambda *args: self.HideOrShowComponents('capillary', self.show_capillary.get()))
        # self.show_vein.trace_add("write", lambda *args: self.HideOrShowComponents('vein', self.show_vein.get()))
        # self.show_artery.trace_add("write", lambda *args: self.HideOrShowComponents('artery', self.show_artery.get()))
        # self.show_faz.trace_add("write", lambda *args: self.HideOrShowComponents('faz', self.show_faz.get()))
        # Bind the checkboxes to the HideOrShowComponents method
        self.show_capillary.trace_add("write", lambda *args: self.HideOrShowComponents())
        self.show_vein.trace_add("write", lambda *args: self.HideOrShowComponents())
        self.show_artery.trace_add("write", lambda *args: self.HideOrShowComponents())
        self.show_faz.trace_add("write", lambda *args: self.HideOrShowComponents())
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
                (0, 0, 0): 'bg',
                (1, 1, 1): 'cap',
                (2, 2, 2): 'art',
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
            fig.set_size_inches(4.7,2.5)
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

            # Clear previous contents in the area_frame
            for widget in self.area_frame.winfo_children():
                widget.destroy()

            # Display the surface area
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
#image_label e labelul folosit pentru a arata imaginea. cv_image e imaginea folosita de libraria opencv pt a o procesa
            # Map the original pixel values to the new colors
            for i in range(self.cv_image.shape[0]):
                for j in range(self.cv_image.shape[1]):
                    pixel_value = tuple(self.cv_image[i, j])
                    # if pixel_value in color_map: scot check-ul pt o viteza mai mare
                    colorized_image[i, j] = color_map[pixel_value]

            # Display the colorized image using Tkinter
            colorized_image_pil = Image.fromarray(colorized_image)
            photo = ImageTk.PhotoImage(colorized_image_pil)
            self.image_label.config(image=photo)
            self.image_label.image = photo
            print(self.image.size)

    # def HideOrShowComponents(self, component, action):
    #     # Map components to their corresponding colors
    #     print(component, action)
    #     color_map = {
    #         'capillary': (222, 222, 186),  # Capillaries color
    #         'vein': (0, 102, 255),  # Veins color
    #         'artery': (153, 0, 153),  # Arteries color
    #         'faz': (179, 0, 0)  # FAZ color
    #     }
    #     cv_color_map = {
    #         'capillary': (1, 1, 1),  # Capilare, alb galbui
    #         'artery': (2, 2, 2),  # Artere, mov
    #         'vein': (3, 3, 3),  # Vene, albastru
    #         'faz': (4, 4, 4)  # FAZ, galben ish
    #     }
    #
    #     nujce_color_map = {
    #         (0, 0, 0): (0, 0, 0),  # Fundal, negru
    #         (1, 1, 1): (222, 222, 186),  # Capilare, alb galbui
    #         (2, 2, 2): (153, 0, 153),  # Artere, mov
    #         (3, 3, 3): (0, 102, 255),  # Vene, albastru
    #         (4, 4, 4): (179, 0, 0)  # FAZ, galben ish
    #     }
    #
    #     # Get the target color for the specified component
    #     target_color = color_map[component]
    #     cv_target_color = cv_color_map[component]
    #     # Create a modified image based on the action
    #     modified_image = np.copy(self.cv_image)
    #
    #     if action == 0:
    #         for i in range(modified_image.shape[0]):
    #             for j in range(modified_image.shape[1]):
    #                 if tuple(modified_image[i, j]) == cv_target_color:
    #                     modified_image[i, j] = (0, 0, 0)
    #                 else:
    #                     modified_image[i, j] = nujce_color_map[modified_image[i,j]]
    #
    #     # if action == 1:
    #     #     for i in range(modified_image.shape[0]):
    #     #         for j in range(modified_image.shape[1]):
    #     #             if tuple(modified_image[i, j]) == cv_target_color:
    #     #                 modified_image[i, j] = nujce_color_map[modified_image[i, j]]
    #     #             else:
    #     #                 modified_image[i, j] = nujce_color_map[modified_image[i, j]]
    #
    #
    #                     # Display the modified image using Tkinter
    #     modified_image_pil = Image.fromarray(modified_image)
    #     photo = ImageTk.PhotoImage(modified_image_pil)
    #     self.image_label.config(image=photo)
    #     self.image_label.image = photo
    #
    #     print(component, action)

    def HideOrShowComponents(self):  # imaginile sunt 400 pe 400 si 6mm pe 6mm.
        if hasattr(self, 'cv_image'):
            # Define the color mappings
            color_map = {
                (0, 0, 0): (0, 0, 0),  # Fundal, negru
                (1, 1, 1):  (0, 0, 0) if self.show_capillary.get() == 0 else (222, 222, 186),  # Capilare, alb galbui
                (2, 2, 2): (0, 0, 0) if self.show_artery.get() == 0 else (153, 0, 153),  # Artere, mov
                (3, 3, 3): (0, 0, 0) if self.show_vein.get() == 0 else (0, 102, 255),  # Vene, albastru
                (4, 4, 4): (0, 0, 0) if self.show_faz.get() == 0 else (179, 0, 0)  # FAZ, galben ish
            }

            # Create an empty image for the colorized output
            colorized_image = np.zeros((self.cv_image.shape[0], self.cv_image.shape[1], 3), dtype=np.uint8)
            # image_label e labelul folosit pentru a arata imaginea. cv_image e imaginea folosita de libraria opencv pt a o procesa
            # Map the original pixel values to the new colors
            for i in range(self.cv_image.shape[0]):
                for j in range(self.cv_image.shape[1]):
                    pixel_value = tuple(self.cv_image[i, j])
                    # if pixel_value in color_map: scot check-ul pt o viteza mai mare
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











#
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
#         self.root.geometry("1600x800")  # Set the window size
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
#         self.image_frame.grid(row=0, column=1, padx=10, pady=10, sticky="n")
#
#         # Add a label to display the image
#         self.image_label = tk.Label(self.image_frame)
#         self.image_label.pack()
#         self.image_label.bind("<Button-1>", self.print_pixel_color)  # Bind left mouse click event
#
#         # Frame for the histogram
#         self.histogram_frame = tk.Frame(root)
#         self.histogram_frame.grid(row=0, column=2, padx=10, pady=10, sticky="n")
#
#         # Frame for the area information
#         self.area_frame = tk.Frame(root)
#         self.area_frame.grid(row=0, column=3, padx=10, pady=10, sticky="n")
#
#         #adaugam frame pentru checkboxes
#         self.checkbox_frame = tk.Frame(self.button_frame)
#         self.checkbox_frame.pack(pady=5)
#
#         #creem checkboxes
#         # Variables to store checkbox states
#         self.show_capillary = tk.IntVar(value=1)
#         self.show_vein = tk.IntVar(value=1)
#         self.show_artery = tk.IntVar(value=1)
#         self.show_faz = tk.IntVar(value=1)
#
#         # Create checkboxes
#         self.capillary_checkbox = ttk.Checkbutton(self.checkbox_frame, text="Capillary", variable=self.show_capillary)
#         self.capillary_checkbox.pack(anchor="w")
#
#         self.vein_checkbox = ttk.Checkbutton(self.checkbox_frame, text="Vein", variable=self.show_vein)
#         self.vein_checkbox.pack(anchor="w")
#
#         self.artery_checkbox = ttk.Checkbutton(self.checkbox_frame, text="Artery", variable=self.show_artery)
#         self.artery_checkbox.pack(anchor="w")
#
#         self.faz_checkbox = ttk.Checkbutton(self.checkbox_frame, text="FAZ", variable=self.show_faz)
#         self.faz_checkbox.pack(anchor="w")
#
#         # Bind the checkboxes to the HideOrShowComponents method
#         # self.show_capillary.trace_add("write", lambda *args: self.HideOrShowComponents('capillary', self.show_capillary.get()))
#         # self.show_vein.trace_add("write", lambda *args: self.HideOrShowComponents('vein', self.show_vein.get()))
#         # self.show_artery.trace_add("write", lambda *args: self.HideOrShowComponents('artery', self.show_artery.get()))
#         # self.show_faz.trace_add("write", lambda *args: self.HideOrShowComponents('faz', self.show_faz.get()))
#         # Bind the checkboxes to the HideOrShowComponents method
#         self.show_capillary.trace_add("write", lambda *args: self.HideOrShowComponents())
#         self.show_vein.trace_add("write", lambda *args: self.HideOrShowComponents())
#         self.show_artery.trace_add("write", lambda *args: self.HideOrShowComponents())
#         self.show_faz.trace_add("write", lambda *args: self.HideOrShowComponents())
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
#             # Precompute images
#             self.precompute_images()
#
#     def precompute_images(self):
#         self.precomputed_images = {}
#         checkbox_states = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]
#
#         for states in checkbox_states:
#             show_capillary, show_artery, show_vein, show_faz = states
#
#             color_map = {
#                 (0, 0, 0): (0, 0, 0),  # Fundal, negru
#                 (1, 1, 1): (0, 0, 0) if show_capillary == 0 else (222, 222, 186),  # Capilare, alb galbui
#                 (2, 2, 2): (0, 0, 0) if show_artery == 0 else (153, 0, 153),  # Artere, mov
#                 (3, 3, 3): (0, 0, 0) if show_vein == 0 else (0, 102, 255),  # Vene, albastru
#                 (4, 4, 4): (0, 0, 0) if show_faz == 0 else (179, 0, 0)  # FAZ, galben ish
#             }
#
#             colorized_image = np.zeros((self.cv_image.shape[0], self.cv_image.shape[1], 3), dtype=np.uint8)
#             for i in range(self.cv_image.shape[0]):
#                 for j in range(self.cv_image.shape[1]):
#                     pixel_value = tuple(self.cv_image[i, j])
#                     colorized_image[i, j] = color_map[pixel_value]
#
#             colorized_image_pil = Image.fromarray(colorized_image)
#             self.precomputed_images[states] = ImageTk.PhotoImage(colorized_image_pil)
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
#                 (0, 0, 0): 'bg',
#                 (1, 1, 1): 'cap',
#                 (2, 2, 2): 'art',
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
#             # Plot histogram
#             fig, ax = plt.subplots()
#             fig.set_size_inches(4.7,2.5)
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
#             # Clear previous contents in the area_frame
#             for widget in self.area_frame.winfo_children():
#                 widget.destroy()
#
#             # Display the surface area
#             for color, area in surface_areas.items():
#                 label = tk.Label(self.area_frame, text=f"{color.capitalize()} area: {area:.2f} mm^2")
#                 label.pack()
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
# #image_label e labelul folosit pentru a arata imaginea. cv_image e imaginea folosita de libraria opencv pt a o procesa
#             # Map the original pixel values to the new colors
#             for i in range(self.cv_image.shape[0]):
#                 for j in range(self.cv_image.shape[1]):
#                     pixel_value = tuple(self.cv_image[i, j])
#                     # if pixel_value in color_map: scot check-ul pt o viteza mai mare
#                     colorized_image[i, j] = color_map[pixel_value]
#
#             # Display the colorized image using Tkinter
#             colorized_image_pil = Image.fromarray(colorized_image)
#             photo = ImageTk.PhotoImage(colorized_image_pil)
#             self.image_label.config(image=photo)
#             self.image_label.image = photo
#             print(self.image.size)
#
#     def HideOrShowComponents(self):
#         states = (
#             self.show_capillary.get(),
#             self.show_artery.get(),
#             self.show_vein.get(),
#             self.show_faz.get()
#         )
#
#         if states in self.precomputed_images:
#             self.image_label.config(image=self.precomputed_images[states])
#             self.image_label.image = self.precomputed_images[states]
#
#     def HideOrShowComponents(self):  # imaginile sunt 400 pe 400 si 6mm pe 6mm.
#         if hasattr(self, 'cv_image'):
#             # Define the color mappings
#             color_map = {
#                 (0, 0, 0): (0, 0, 0),  # Fundal, negru
#                 (1, 1, 1):  (0, 0, 0) if self.show_capillary.get() == 0 else (222, 222, 186),  # Capilare, alb galbui
#                 (2, 2, 2): (0, 0, 0) if self.show_artery.get() == 0 else (153, 0, 153),  # Artere, mov
#                 (3, 3, 3): (0, 0, 0) if self.show_vein.get() == 0 else (0, 102, 255),  # Vene, albastru
#                 (4, 4, 4): (0, 0, 0) if self.show_faz.get() == 0 else (179, 0, 0)  # FAZ, galben ish
#             }
#
#             # Create an empty image for the colorized output
#             colorized_image = np.zeros((self.cv_image.shape[0], self.cv_image.shape[1], 3), dtype=np.uint8)
#             # image_label e labelul folosit pentru a arata imaginea. cv_image e imaginea folosita de libraria opencv pt a o procesa
#             # Map the original pixel values to the new colors
#             for i in range(self.cv_image.shape[0]):
#                 for j in range(self.cv_image.shape[1]):
#                     pixel_value = tuple(self.cv_image[i, j])
#                     # if pixel_value in color_map: scot check-ul pt o viteza mai mare
#                     colorized_image[i, j] = color_map[pixel_value]
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
#
