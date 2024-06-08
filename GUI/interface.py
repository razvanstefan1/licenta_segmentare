
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt


class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Viewer")
        self.root.geometry("1000x800")  # Set the window size

        # Frame for the image
        self.image_frame = tk.Frame(root)
        self.image_frame.grid(row=0, column=0, padx=10, pady=10)

        # Add a button to open an image
        self.open_button = tk.Button(self.image_frame, text="Open Image", command=self.open_image)
        self.open_button.pack()

        # Add a label to display the image
        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack()
        self.image_label.bind("<Button-1>", self.print_pixel_color)  # Bind left mouse click event

        # Add a button to calculate histogram
        self.calculate_button = tk.Button(self.image_frame, text="Calculate", command=self.calculate_histogram)
        self.calculate_button.pack()

        # Add a button to colorize predictions
        self.color_button = tk.Button(self.image_frame, text="Color", command=self.ColorPredictions)
        self.color_button.pack()

        # Frame for the text boxes
        self.text_frame = tk.Frame(root)
        self.text_frame.grid(row=0, column=1, padx=10, pady=10, sticky="n")

        # Add text boxes with lorem ipsum text
        self.text_boxes = []
        lorem_ipsum = (
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
            "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
            "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris.",
            "Nisi ut aliquip ex ea commodo consequat.",
            "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum."
        )
        for text in lorem_ipsum:
            textbox = tk.Text(self.text_frame, height=4, width=40)
            textbox.insert(tk.END, text)
            textbox.config(state=tk.DISABLED)  # Make the text box read-only
            textbox.pack(pady=5)
            self.text_boxes.append(textbox)

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
                (255, 0, 0): 'red',  # Red
                (0, 0, 255): 'blue',  # Blue
                (255, 255, 255): 'white',  # White
                (0, 255, 0): 'lime'  # Lime
            }

            # Count the occurrences of the predefined colors
            color_histogram = {color: 0 for color in predefined_colors.values()}
            for color, count in color_count.items():
                if color in predefined_colors:
                    color_histogram[predefined_colors[color]] += count

            # Plot histogram
            plt.bar(color_histogram.keys(), color_histogram.values(), color=['red', 'blue', 'white', 'lime'])
            plt.xlabel('Colors')
            plt.ylabel('Pixel Count')
            plt.title('Color Histogram')
            plt.show()
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

            #0, 0, 0 - fundal
            #1, 1, 1 - capilare
            #2, 2, 2 - artere
            #3, 3, 3 - vene

            # Create an empty image for the colorized output
            colorized_image = np.zeros((self.cv_image.shape[0], self.cv_image.shape[1], 3), dtype=np.uint8)

            # Map the original pixel values to the new colors
            for i in range(self.cv_image.shape[0]):
                for j in range(self.cv_image.shape[1]):
                    pixel_value = tuple(self.cv_image[i, j])
                    if pixel_value in color_map:
                        colorized_image[i, j] = color_map[pixel_value]

            # Save the colorized image
           # colorized_image_path = os.path.join(opt.saveroot, 'colorized_image.png')
           # cv2.imwrite(colorized_image_path, cv2.cvtColor(colorized_image, cv2.COLOR_RGB2BGR))

            #convert colorized_image to rgb
            #colorized_image = cv2.cvtColor(colorized_image, cv2.COLOR_BGR2RGB)
            #cv2.imshow("Color Predictions", colorized_image)  #e inversata daca nu fac conversia de mai sus

            # Display the colorized image using Tkinter
            colorized_image_pil = Image.fromarray(colorized_image)
            photo = ImageTk.PhotoImage(colorized_image_pil)
            self.image_label.config(image=photo)
            self.image_label.image = photo
            print (self.image.size)


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()
