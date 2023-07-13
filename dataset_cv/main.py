import tkinter as tk
from PIL import Image, ImageTk

# List of class labels
class_labels = [
    "pasta with pesto",
    "pasta with tomato sauce",
    "pasta with meat sauce",
    "pasta with clams and mussels",
    "pilaw rice with peppers and peas",
    "grilled pork cutlet",
    "fish cutlet",
    "rabbit",
    "seafood salad",
    "beans",
    "basil potatoes",
    "salad",
    "bread",
    "Not known"
]

# Function to handle button click event
def classify_image():
    # Get the selected class label
    class_label = class_labels[class_var.get()]

    # Write the assigned class to the text file
    with open("image_classes.txt", "a") as file:
        file.write(f"{image_filename},{class_label}\n")

    # Close the image window
    window.destroy()

# Read and classify each image
for i in range(97, 114):
    # Load the image
    image_filename = f"{i}.jpg"
    image = Image.open(image_filename)

    # Create a tkinter window
    window = tk.Tk()

    # Set up a tkinter window for the image
    image_window = tk.Label(window)
    image_window.pack()

    # Display the image in the window
    image_tk = ImageTk.PhotoImage(image)
    image_window.configure(image=image_tk)
    image_window.image = image_tk

    # Create a tkinter variable for the selected class
    class_var = tk.IntVar()

    # Create radio buttons for each class
    for index, label in enumerate(class_labels):
        radio_button = tk.Radiobutton(window, text=label, variable=class_var, value=index)
        radio_button.pack()

    # Create a button to classify the image
    classify_button = tk.Button(window, text="Classify", command=classify_image)
    classify_button.pack()

    # Start the tkinter event loop
    window.mainloop()
