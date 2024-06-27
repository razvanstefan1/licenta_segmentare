import os
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import GenerateSegmentation


# imaginea pe care o deschide sa fie displayed. Cand da click pe altele sa apara alea.
# trimitem image number la generatesegmentation precum si rezolutia
class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Segmentation Viewer")
        self.root.geometry("1920x1080")  # Set the window size

        # frame butoane
        self.button_frame = tk.Frame(root)
        self.button_frame.grid(row=0, column=0, padx=10, pady=10, sticky="n")

        # adaugarea btoanelor la frame urile lor
        self.open_button = ttk.Button(self.button_frame, text="Open Image", command=self.open_image)
        self.open_button.pack(pady=5)

        self.calculate_button = ttk.Button(self.button_frame, text="Calculate Histogram",
                                           command=self.calculate_histogram)
        self.calculate_button.pack(pady=5)

        self.color_button = ttk.Button(self.button_frame, text="Color Predictions", command=self.ColorPredictions)
        self.color_button.pack(pady=5)

        # frame pentru imagine
        self.image_frame = tk.Frame(root)
        self.image_frame.grid(row=0, column=1, padx=10, pady=10, sticky="n")

        # label in care e aratata inaginea
        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack()
        self.image_label.bind("<Button-1>", self.print_pixel_color)  # Bind left mouse click event

        # frame pt sitogt
        self.histogram_frame = tk.Frame(root)
        self.histogram_frame.grid(row=0, column=2, padx=10, pady=10, sticky="n")

        self.area_frame = tk.Frame(root)
        self.area_frame.grid(row=0, column=3, padx=10, pady=10, sticky="n")

        self.checkbox_frame = tk.Frame(self.button_frame)
        self.checkbox_frame.pack(pady=5)

        self.show_capillary = tk.IntVar(value=1)
        self.show_vein = tk.IntVar(value=1)
        self.show_artery = tk.IntVar(value=1)
        self.show_faz = tk.IntVar(value=1)

        self.capillary_checkbox = ttk.Checkbutton(self.checkbox_frame, text="Capillary", variable=self.show_capillary)
        self.capillary_checkbox.pack(anchor="w")

        self.vein_checkbox = ttk.Checkbutton(self.checkbox_frame, text="Vein", variable=self.show_vein)
        self.vein_checkbox.pack(anchor="w")

        self.artery_checkbox = ttk.Checkbutton(self.checkbox_frame, text="Artery", variable=self.show_artery)
        self.artery_checkbox.pack(anchor="w")

        self.faz_checkbox = ttk.Checkbutton(self.checkbox_frame, text="FAZ", variable=self.show_faz)
        self.faz_checkbox.pack(anchor="w")

        self.show_capillary.trace_add("write", lambda *args: self.HideOrShowComponents())
        self.show_vein.trace_add("write", lambda *args: self.HideOrShowComponents())
        self.show_artery.trace_add("write", lambda *args: self.HideOrShowComponents())
        self.show_faz.trace_add("write", lambda *args: self.HideOrShowComponents())

    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.gif;*.bmp")])
        if file_path:

            # imaginea afisata e cea deschisa
            self.image = Image.open(file_path)
            photo = ImageTk.PhotoImage(self.image)
            self.image_label.config(image=photo)
            self.image_label.image = photo
            width, height = self.image.size

            filename = os.path.basename(file_path)

            test_results_path = os.path.join('C:\\Users\\brolz\\Desktop\\FACULTATE\\LICENTA\\COD_LICENTA_SEGMENTARE',
                                             'saveroot', 'test_results', filename)

            # imaginea pe care se fac calculele e aia cu labelu de la pathu asta
            # self.cv_image = cv2.imread(test_results_path)
            # if self.cv_image is not None:
            #     self.cv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
            # else:
            #     print(f"Image not found in the test_results directory: {test_results_path}")
            self.cv_image = GenerateSegmentation.generateSegmentationExternal(image_number=filename, ext_output_path='C:\\Users\\brolz\\Desktop\\FACULTATE\\LICENTA\\generated',
                                                                              image_resolution=(width, height),
                                                                              modality_root_dir='C:\\Users\\brolz\\Desktop\\FACULTATE\\LICENTA\\MODALITIES\\')
            if self.cv_image is not None:
                self.cv_image = cv2.cvtColor (self.cv_image, cv2.COLOR_BGR2RGB)
            else:
                print(f"Image not found in the test_results directory: {test_results_path}")

    def calculate_histogram(self):
        if hasattr(self, 'cv_image'):
            pixels = [tuple(pixel) for pixel in self.cv_image.reshape(-1, 3)]

            color_count = Counter(pixels)

            for color, count in color_count.items():
                print(f"Color: {color}, Count: {count}")

            predefined_colors = {
                (0, 0, 0): 'bg',
                (1, 1, 1): 'cap',
                (2, 2, 2): 'art',
                (3, 3, 3): 'vein',
                (4, 4, 4): 'FAZ'
            }

            color_histogram = {color: 0 for color in predefined_colors.values()}
            for color, count in color_count.items():
                if color in predefined_colors:
                    color_histogram[predefined_colors[color]] += count

            total_pixels = self.cv_image.shape[0] * self.cv_image.shape[1]

            img_dim = 6
            if self.cv_image.shape[0] == 304:  # daca e 3 pe 3 mm
                img_dim = 3

            area_per_pixel = (img_dim * img_dim) / total_pixels

            surface_areas = {color: count * area_per_pixel for color, count in color_histogram.items()}

            fig, ax = plt.subplots()
            fig.set_size_inches(4.7, 2.5)
            ax.set_facecolor('lightgreen')
            ax.bar(color_histogram.keys(), color_histogram.values(), color=['black', 'white', 'purple', 'blue', 'red'])
            ax.set_xlabel('Colors')
            ax.set_ylabel('Pixel Count')
            ax.set_title('Color Histogram')

            for widget in self.histogram_frame.winfo_children():
                widget.destroy()
            canvas = FigureCanvasTkAgg(fig, master=self.histogram_frame)
            canvas.draw()
            canvas.get_tk_widget().pack()

            for widget in self.area_frame.winfo_children():
                widget.destroy()

            for color, area in surface_areas.items():
                label = tk.Label(self.area_frame, text=f"{color.capitalize()} area: {area:.2f} mm^2")
                label.pack()

        else:
            print("No image loaded!")

    def print_pixel_color(self, event):
        if hasattr(self, 'cv_image'):
            x, y = event.x, event.y

            width, height = self.image.size
            label_width, label_height = self.image_label.winfo_width(), self.image_label.winfo_height()
            scale_x, scale_y = width / label_width, height / label_height
            x = int(x * scale_x)
            y = int(y * scale_y)

            color = self.cv_image[y, x]
            print(f"Clicked at ({x}, {y}), Color: {color}")

    def ColorPredictions(self):
        if hasattr(self, 'cv_image'):

            color_map = {
                (0, 0, 0): (0, 0, 0),  # Fundal, negru
                (1, 1, 1): (222, 222, 186),  # Capilare, alb galbui
                (2, 2, 2): (153, 0, 153),  # Artere, mov
                (3, 3, 3): (0, 102, 255),  # Vene, albastru
                (4, 4, 4): (179, 0, 0)  # FAZ, galben ish
            }

            colorized_image = np.zeros((self.cv_image.shape[0], self.cv_image.shape[1], 3), dtype=np.uint8)
            # image_label e labelul folosit pentru a arata imaginea. cv_image e imaginea folosita de libraria opencv pt a o procesa

            for i in range(self.cv_image.shape[0]):
                for j in range(self.cv_image.shape[1]):
                    pixel_value = tuple(self.cv_image[i, j])
                    # if pixel_value in color_map: scot check-ul pt o viteza mai mare
                    colorized_image[i, j] = color_map[pixel_value]

            colorized_image_pil = Image.fromarray(colorized_image)
            photo = ImageTk.PhotoImage(colorized_image_pil)
            self.image_label.config(image=photo)
            self.image_label.image = photo
            print(self.image.size)

    def HideOrShowComponents(self):  # imaginile sunt 400 pe 400 si 6mm pe 6mm.
        if hasattr(self, 'cv_image'):

            color_map = {
                (0, 0, 0): (0, 0, 0),  # Fundal, negru
                (1, 1, 1): (0, 0, 0) if self.show_capillary.get() == 0 else (222, 222, 186),  # Capilare, alb galbui
                (2, 2, 2): (0, 0, 0) if self.show_artery.get() == 0 else (153, 0, 153),  # Artere, mov
                (3, 3, 3): (0, 0, 0) if self.show_vein.get() == 0 else (0, 102, 255),  # Vene, albastru
                (4, 4, 4): (0, 0, 0) if self.show_faz.get() == 0 else (179, 0, 0)  # FAZ, galben ish
            }

            colorized_image = np.zeros((self.cv_image.shape[0], self.cv_image.shape[1], 3), dtype=np.uint8)
            # image_label e labelul folosit pentru a arata imaginea. cv_image e imaginea folosita de libraria opencv pt a o procesa

            for i in range(self.cv_image.shape[0]):
                for j in range(self.cv_image.shape[1]):
                    pixel_value = tuple(self.cv_image[i, j])
                    # if pixel_value in color_map: scot check-ul pt o viteza mai mare
                    colorized_image[i, j] = color_map[pixel_value]

            colorized_image_pil = Image.fromarray(colorized_image)
            photo = ImageTk.PhotoImage(colorized_image_pil)
            self.image_label.config(image=photo)
            self.image_label.image = photo
            print(self.image.size)


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()

# import os
# import tkinter as tk
# from tkinter import filedialog, ttk
# from PIL import Image, ImageTk
# import cv2
# import numpy as np
# from collections import Counter
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# import GenerateSegmentation
# #imaginea pe care o deschide sa fie displayed. Cand da click pe altele sa apara alea.
# #trimitem image number la generatesegmentation precum si rezolutia
# class ImageApp:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("Image Segmentation Viewer")
#         self.root.geometry("1920x1080")  # Set the window size
#
#         # frame butoane
#         self.button_frame = tk.Frame(root)
#         self.button_frame.grid(row=0, column=0, padx=10, pady=10, sticky="n")
#
#         # adaugarea btoanelor la frame urile lor
#         self.open_button = ttk.Button(self.button_frame, text="Open Image", command=self.open_image)
#         self.open_button.pack(pady=5)
#
#         self.calculate_button = ttk.Button(self.button_frame, text="Calculate Histogram", command=self.calculate_histogram)
#         self.calculate_button.pack(pady=5)
#
#         self.color_button = ttk.Button(self.button_frame, text="Color Predictions", command=self.ColorPredictions)
#         self.color_button.pack(pady=5)
#
#         # frame pentru imagine
#         self.image_frame = tk.Frame(root)
#         self.image_frame.grid(row=0, column=1, padx=10, pady=10, sticky="n")
#
#         # label in care e aratata inaginea
#         self.image_label = tk.Label(self.image_frame)
#         self.image_label.pack()
#         self.image_label.bind("<Button-1>", self.print_pixel_color)  # Bind left mouse click event
#
#         # frame pt sitogt
#         self.histogram_frame = tk.Frame(root)
#         self.histogram_frame.grid(row=0, column=2, padx=10, pady=10, sticky="n")
#
#
#         self.area_frame = tk.Frame(root)
#         self.area_frame.grid(row=0, column=3, padx=10, pady=10, sticky="n")
#
#
#         self.checkbox_frame = tk.Frame(self.button_frame)
#         self.checkbox_frame.pack(pady=5)
#
#
#
#         self.show_capillary = tk.IntVar(value=1)
#         self.show_vein = tk.IntVar(value=1)
#         self.show_artery = tk.IntVar(value=1)
#         self.show_faz = tk.IntVar(value=1)
#
#
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
#         self.show_capillary.trace_add("write", lambda *args: self.HideOrShowComponents())
#         self.show_vein.trace_add("write", lambda *args: self.HideOrShowComponents())
#         self.show_artery.trace_add("write", lambda *args: self.HideOrShowComponents())
#         self.show_faz.trace_add("write", lambda *args: self.HideOrShowComponents())
#
#     def open_image(self):
#         file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.gif;*.bmp")])
#         if file_path:
#
#             #imaginea afisata e cea deschisa
#             self.image = Image.open(file_path)
#             photo = ImageTk.PhotoImage(self.image)
#             self.image_label.config(image=photo)
#             self.image_label.image = photo
#
#             filename = os.path.basename(file_path)
#
#             test_results_path = os.path.join('C:\\Users\\brolz\\Desktop\\FACULTATE\\LICENTA\\COD_LICENTA_SEGMENTARE', 'saveroot', 'test_results', filename)
#
#             #imaginea pe care se fac calculele e aia cu labelu de la pathu asta
#             self.cv_image = cv2.imread(test_results_path)
#             if self.cv_image is not None:
#                 self.cv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
#             else:
#                 print(f"Image not found in the test_results directory: {test_results_path}")
#
#     def calculate_histogram(self):
#         if hasattr(self, 'cv_image'):
#             pixels = [tuple(pixel) for pixel in self.cv_image.reshape(-1, 3)]
#
#             color_count = Counter(pixels)
#
#             for color, count in color_count.items():
#                 print(f"Color: {color}, Count: {count}")
#
#             predefined_colors = {
#                 (0, 0, 0): 'bg',
#                 (1, 1, 1): 'cap',
#                 (2, 2, 2): 'art',
#                 (3, 3, 3): 'vein',
#                 (4, 4, 4): 'FAZ'
#             }
#
#
#             color_histogram = {color: 0 for color in predefined_colors.values()}
#             for color, count in color_count.items():
#                 if color in predefined_colors:
#                     color_histogram[predefined_colors[color]] += count
#
#
#             total_pixels = self.cv_image.shape[0] * self.cv_image.shape[1]
#
#             img_dim = 6
#             if self.cv_image.shape[0] == 304: #daca e 3 pe 3 mm
#                 img_dim = 3
#
#             area_per_pixel = (img_dim * img_dim) / total_pixels
#
#
#             surface_areas = {color: count * area_per_pixel for color, count in color_histogram.items()}
#
#
#             fig, ax = plt.subplots()
#             fig.set_size_inches(4.7,2.5)
#             ax.set_facecolor('lightgreen')
#             ax.bar(color_histogram.keys(), color_histogram.values(), color=['black', 'white', 'purple', 'blue', 'red'])
#             ax.set_xlabel('Colors')
#             ax.set_ylabel('Pixel Count')
#             ax.set_title('Color Histogram')
#
#
#             for widget in self.histogram_frame.winfo_children():
#                 widget.destroy()
#             canvas = FigureCanvasTkAgg(fig, master=self.histogram_frame)
#             canvas.draw()
#             canvas.get_tk_widget().pack()
#
#
#             for widget in self.area_frame.winfo_children():
#                 widget.destroy()
#
#
#             for color, area in surface_areas.items():
#                 label = tk.Label(self.area_frame, text=f"{color.capitalize()} area: {area:.2f} mm^2")
#                 label.pack()
#
#         else:
#             print("No image loaded!")
#
#     def print_pixel_color(self, event):
#         if hasattr(self, 'cv_image'):
#
#             x, y = event.x, event.y
#
#             width, height = self.image.size
#             label_width, label_height = self.image_label.winfo_width(), self.image_label.winfo_height()
#             scale_x, scale_y = width / label_width, height / label_height
#             x = int(x * scale_x)
#             y = int(y * scale_y)
#
#
#             color = self.cv_image[y, x]
#             print(f"Clicked at ({x}, {y}), Color: {color}")
#
#     def ColorPredictions(self):
#         if hasattr(self, 'cv_image'):
#
#             color_map = {
#                 (0, 0, 0): (0, 0, 0),  # Fundal, negru
#                 (1, 1, 1): (222, 222, 186),  # Capilare, alb galbui
#                 (2, 2, 2): (153, 0, 153),  # Artere, mov
#                 (3, 3, 3): (0, 102, 255),  # Vene, albastru
#                 (4, 4, 4): (179,0,0)  # FAZ, galben ish
#             }
#
#
#             colorized_image = np.zeros((self.cv_image.shape[0], self.cv_image.shape[1], 3), dtype=np.uint8)
# #image_label e labelul folosit pentru a arata imaginea. cv_image e imaginea folosita de libraria opencv pt a o procesa
#
#             for i in range(self.cv_image.shape[0]):
#                 for j in range(self.cv_image.shape[1]):
#                     pixel_value = tuple(self.cv_image[i, j])
#                     # if pixel_value in color_map: scot check-ul pt o viteza mai mare
#                     colorized_image[i, j] = color_map[pixel_value]
#
#
#             colorized_image_pil = Image.fromarray(colorized_image)
#             photo = ImageTk.PhotoImage(colorized_image_pil)
#             self.image_label.config(image=photo)
#             self.image_label.image = photo
#             print(self.image.size)
#
#     def HideOrShowComponents(self):  # imaginile sunt 400 pe 400 si 6mm pe 6mm.
#         if hasattr(self, 'cv_image'):
#
#             color_map = {
#                 (0, 0, 0): (0, 0, 0),  # Fundal, negru
#                 (1, 1, 1):  (0, 0, 0) if self.show_capillary.get() == 0 else (222, 222, 186),  # Capilare, alb galbui
#                 (2, 2, 2): (0, 0, 0) if self.show_artery.get() == 0 else (153, 0, 153),  # Artere, mov
#                 (3, 3, 3): (0, 0, 0) if self.show_vein.get() == 0 else (0, 102, 255),  # Vene, albastru
#                 (4, 4, 4): (0, 0, 0) if self.show_faz.get() == 0 else (179, 0, 0)  # FAZ, galben ish
#             }
#
#
#             colorized_image = np.zeros((self.cv_image.shape[0], self.cv_image.shape[1], 3), dtype=np.uint8)
#             # image_label e labelul folosit pentru a arata imaginea. cv_image e imaginea folosita de libraria opencv pt a o procesa
#
#             for i in range(self.cv_image.shape[0]):
#                 for j in range(self.cv_image.shape[1]):
#                     pixel_value = tuple(self.cv_image[i, j])
#                     # if pixel_value in color_map: scot check-ul pt o viteza mai mare
#                     colorized_image[i, j] = color_map[pixel_value]
#
#
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
#
#
