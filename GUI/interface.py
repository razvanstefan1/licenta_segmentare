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
from GUI.Property_Extractors import ExtractProperties



# trimitem image number la generatesegmentation precum si rezolutia
class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Segmentation Viewer")
        self.root.geometry("1920x1080")  # Set the window size

        # frame pt butoane si imaginea principala
        self.upper_left_frame = tk.Frame(root)
        self.upper_left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nw")

        # frame butoane
        self.button_frame = tk.Frame(self.upper_left_frame)
        self.button_frame.grid(row=0, column=0, padx=10, pady=10, sticky="n")

        # adaugarea btoanelor la frame urile lor
        self.open_button = ttk.Button(self.button_frame, text="Open Image", command=self.open_image)
        self.open_button.pack(pady=5)

        self.calculate_button = ttk.Button(self.button_frame, text="Calculate Histogram",
                                           command=self.calculate_histogram)
        self.calculate_button.pack(pady=5)

        self.color_button = ttk.Button(self.button_frame, text="Color Predictions", command=self.ShowColoredPredictions)
        self.color_button.pack(pady=5)

        # frame pentru imagine
        self.main_img_frame = tk.Frame(self.upper_left_frame)
        self.main_img_frame.grid(row=0, column=1, padx=10, pady=10, sticky="n")

        # label in care e aratam imaginea princpala
        self.main_img_label = tk.Label(self.main_img_frame)
        self.main_img_label.pack()  # punem label ul in centrul frame-ului
        self.main_img_label.bind("<Button-1>", self.print_pixel_color)  # pt click ca sa printeze culoarea
        self.main_img_text_label = tk.Label(self.main_img_frame, text="Main Image")
        self.main_img_text_label.pack_forget()

        # frame nou pentru cele 6 modalitati (lower left corner )
        self.modality_frame = tk.Frame(self.root)
        self.modality_frame.grid(row=1, column=0, padx=10, pady=10, sticky="sw")

        # frame + labels pt toate modalitatile
        self.oct_full_frame = tk.Frame(self.modality_frame)
        self.oct_full_frame.grid(row=0, column=0, padx=10, pady=10, sticky="n")
        self.oct_full_text_label = tk.Label(self.oct_full_frame, text="OCT(FULL)")
        self.oct_full_text_label.pack_forget()
        self.oct_full_label = tk.Label(self.oct_full_frame)
        self.oct_full_label.pack()
        self.oct_full_label.bind("<Button-1>", lambda e: self.change_main_image(self.oct_full_image, "OCT_FULL"))

        self.oct_ilm_opl_frame = tk.Frame(self.modality_frame)
        self.oct_ilm_opl_frame.grid(row=0, column=1, padx=10, pady=10, sticky="n")
        self.oct_ilm_opl_text_label = tk.Label(self.oct_ilm_opl_frame, text="OCT(ILM_OPL)")
        self.oct_ilm_opl_text_label.pack_forget()
        self.oct_ilm_opl_label = tk.Label(self.oct_ilm_opl_frame)
        self.oct_ilm_opl_label.pack()
        self.oct_ilm_opl_label.bind("<Button-1>", lambda e: self.change_main_image(self.oct_ilm_opl_image, "OCT_ILM_OPL"))

        self.oct_opl_bm_frame = tk.Frame(self.modality_frame)
        self.oct_opl_bm_frame.grid(row=0, column=2, padx=10, pady=10, sticky="n")
        self.oct_opl_bm_text_label = tk.Label(self.oct_opl_bm_frame, text="OCT(OPL_BM)")
        self.oct_opl_bm_text_label.pack_forget()
        self.oct_opl_bm_label = tk.Label(self.oct_opl_bm_frame)
        self.oct_opl_bm_label.pack()
        self.oct_opl_bm_label.bind("<Button-1>", lambda e: self.change_main_image(self.oct_opl_bm_image, "OCT_OPL_BM"))

        self.octa_full_frame = tk.Frame(self.modality_frame)
        self.octa_full_frame.grid(row=1, column=0, padx=10, pady=10, sticky="n")
        self.octa_full_text_label = tk.Label(self.octa_full_frame, text="OCTA(FULL)")
        self.octa_full_text_label.pack_forget()
        self.octa_full_label = tk.Label(self.octa_full_frame)
        self.octa_full_label.pack()
        self.octa_full_label.bind("<Button-1>", lambda e: self.change_main_image(self.octa_full_image, "OCTA_FULL"))

        self.octa_ilm_opl_frame = tk.Frame(self.modality_frame)
        self.octa_ilm_opl_frame.grid(row=1, column=1, padx=10, pady=10, sticky="n")
        self.octa_ilm_opl_text_label = tk.Label(self.octa_ilm_opl_frame, text="OCTA(ILM_OPL)")
        self.octa_ilm_opl_text_label.pack_forget()
        self.octa_ilm_opl_label = tk.Label(self.octa_ilm_opl_frame)
        self.octa_ilm_opl_label.pack()
        self.octa_ilm_opl_label.bind("<Button-1>", lambda e: self.change_main_image(self.octa_ilm_opl_image, "OCTA_ILM_OPL"))

        self.octa_opl_bm_frame = tk.Frame(self.modality_frame)
        self.octa_opl_bm_frame.grid(row=1, column=2, padx=10, pady=10, sticky="n")
        self.octa_opl_bm_text_label = tk.Label(self.octa_opl_bm_frame, text="OCTA(OPL_BM)")
        self.octa_opl_bm_text_label.pack_forget()
        self.octa_opl_bm_label = tk.Label(self.octa_opl_bm_frame)
        self.octa_opl_bm_label.pack()
        self.octa_opl_bm_label.bind("<Button-1>", lambda e: self.change_main_image(self.octa_opl_bm_image, "OCTA_OPL_BM"))

        # frame pt histograma
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

        # ascundem histograma si ariile cand deschidem alta poza ca sa nu ramana cea de la poza trecuta
        self.histogram_frame.grid_forget()
        self.area_frame.grid_forget()

        if file_path:
            self.image = Image.open(file_path)
            photo = ImageTk.PhotoImage(self.image)
            self.main_img_label.config(image=photo)
            self.main_img_label.image = photo
            self.main_img_text_label.pack()#label vizibil dupa ce deschidem

            width, height = self.image.size

            # numele pozei efectiv gen 10000.bmp
            filename = os.path.basename(file_path)
            self.filename = filename #ca sa fie accesibil numarul pozei cand o schimb sa adaug modalitatea


            # Construim calea spre root directorul de la modalities
            path_parts = file_path.split('/')
            modality_path_components = path_parts[:-2]
            modality_path = '\\'.join(modality_path_components)
            modality_path += '\\'

            #afisam modalitatea primei imagini deschise
            self.main_img_text_label.config(text=f'{filename}   {path_parts[-2]}')

            # deschidem toate cele 6 imagini
            self.root.after(10, self.open_modalities, filename, modality_path)

            test_results_path = os.path.join('C:\\Users\\brolz\\Desktop\\FACULTATE\\LICENTA\\COD_LICENTA_SEGMENTARE',
                                             'saveroot', 'test_results', filename)

            # programam apelul la segmentare dupa ce se afiseaza imaginile (sa nu fie delay asa mare)
            self.root.after(100, self.call_generate_segmentation_external, filename, (width, height), modality_path,
                          test_results_path)

    def open_modalities(self, filename, modality_path):
        # deschidem toate cele 6 imagini
        # deschidem oct full
        self.oct_full_image = Image.open(modality_path + 'OCT(FULL)\\' + filename)
        self.oct_full_image_smaller = self.oct_full_image.resize((200, 200))
        photo = ImageTk.PhotoImage(self.oct_full_image_smaller)
        self.oct_full_label.config(image=photo)
        self.oct_full_label.image = photo
        self.oct_full_text_label.pack()  #facem label ul vizibil
        # deschidem oct ilm opl
        self.oct_ilm_opl_image = Image.open(modality_path + 'OCT(ILM_OPL)\\' + filename)
        self.oct_ilm_opl_image_smaller = self.oct_ilm_opl_image.resize((200, 200))
        photo = ImageTk.PhotoImage(self.oct_ilm_opl_image_smaller)
        self.oct_ilm_opl_label.config(image=photo)
        self.oct_ilm_opl_label.image = photo
        self.oct_ilm_opl_text_label.pack()  # facem label ul vizibil
        # deschidem oct opl bm
        self.oct_opl_bm_image = Image.open(modality_path + 'OCT(OPL_BM)\\' + filename)
        self.oct_opl_bm_image_smaller = self.oct_opl_bm_image.resize((200, 200))
        photo = ImageTk.PhotoImage(self.oct_opl_bm_image_smaller)
        self.oct_opl_bm_label.config(image=photo)
        self.oct_opl_bm_label.image = photo
        self.oct_opl_bm_text_label.pack()  # facem label ul vizibil

        # deschidem octa full
        self.octa_full_image = Image.open(modality_path + 'OCTA(FULL)\\' + filename)
        self.octa_full_image_smaller = self.octa_full_image.resize((200, 200))
        photo = ImageTk.PhotoImage(self.octa_full_image_smaller)
        self.octa_full_label.config(image=photo)
        self.octa_full_label.image = photo
        self.octa_full_text_label.pack()  # facem label ul vizibil

        # deschidem octa ilm opl
        self.octa_ilm_opl_image = Image.open(modality_path + 'OCTA(ILM_OPL)\\' + filename)
        self.octa_ilm_opl_image_smaller = self.octa_ilm_opl_image.resize((200, 200))
        photo = ImageTk.PhotoImage(self.octa_ilm_opl_image_smaller)
        self.octa_ilm_opl_label.config(image=photo)
        self.octa_ilm_opl_label.image = photo
        self.octa_ilm_opl_text_label.pack()  # facem label ul vizibil

        # deschidem octa opl bm
        self.octa_opl_bm_image = Image.open(modality_path + 'OCTA(OPL_BM)\\' + filename)
        self.octa_opl_bm_image_smaller = self.octa_opl_bm_image.resize((200, 200))
        photo = ImageTk.PhotoImage(self.octa_opl_bm_image_smaller)
        self.octa_opl_bm_label.config(image=photo)
        self.octa_opl_bm_label.image = photo
        self.octa_opl_bm_text_label.pack()  # facem label ul vizibil

    #sa se schimbe imaginea principala cu cea pe care se da click
    def change_main_image(self, image, main_label_text):
        photo = ImageTk.PhotoImage(image)
        self.main_img_label.config(image=photo)
        self.main_img_label.image = photo
        self.main_img_text_label.config(text=f'{self.filename}   {main_label_text}')

    #se face call si la property extractor dupa ce se obtine imaginea aia segmentata cu valori 0,1,2,3. Se face call si la generarea de color predictions
    def call_generate_segmentation_external(self, filename, resolution, modality_path, test_results_path):
        self.cv_image = GenerateSegmentation.generateSegmentationExternal(
            image_number=filename,
            ext_output_path='C:\\Users\\brolz\\Desktop\\FACULTATE\\LICENTA\\generated',
            image_resolution=resolution,
            # modality_root_dir='C:\\Users\\brolz\\Desktop\\FACULTATE\\LICENTA\\MODALITIES\\'
            modality_root_dir=modality_path
        )
        if self.cv_image is not None:
            self.cv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
            #call la property extractor
            ExtractProperties.calculate_properties(self.cv_image)
            #call la color predictions
            self.ColorPredictions()
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

            #aratam din nou histograma si aria (sa se faca refresh cand schimbi imaginea)
            self.histogram_frame.grid(row=0, column=2, padx=10, pady=10, sticky="n")
            self.area_frame.grid(row=0, column=3, padx=10, pady=10, sticky="n")

        else:
            print("No image loaded!")

    def print_pixel_color(self, event):
        if hasattr(self, 'cv_image'):
            x, y = event.x, event.y

            width, height = self.image.size
            label_width, label_height = self.main_img_label.winfo_width(), self.main_img_label.winfo_height()
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
            self.colored_photo = ImageTk.PhotoImage(colorized_image_pil)


    def ShowColoredPredictions(self):
        self.main_img_label.config(image=self.colored_photo)
        self.main_img_label.image = self.colored_photo

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
            self.main_img_label.config(image=photo)
            self.main_img_label.image = photo
            print(self.image.size)


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()


