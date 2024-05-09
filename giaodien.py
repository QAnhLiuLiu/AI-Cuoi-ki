import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np


from tensorflow.keras.models import load_model
model = load_model('benh_la_lua.h5')


class PlantDiseaseApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Ứng dụng Dự đoán Bệnh Lúa")
        self.master.geometry("400x400")

        self.label = tk.Label(self.master, text="Chọn hình ảnh lá cây lúa để dự đoán:")
        self.label.pack()

        self.select_button = tk.Button(self.master, text="Chọn ảnh", command=self.select_image)
        self.select_button.pack()

        self.image_label = tk.Label(self.master)
        self.image_label.pack()

        self.predict_button = tk.Button(self.master, text="Dự đoán", command=self.predict)
        self.predict_button.pack()

        self.result_label = tk.Label(self.master, text="")
        self.result_label.pack()

    def select_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = Image.open(file_path)
            self.image = self.image.resize((150, 150))
            self.photo = ImageTk.PhotoImage(self.image)
            self.image_label.config(image=self.photo)

            self.image = np.expand_dims(self.image, axis=0)

    def predict(self):
        if hasattr(self, 'image'):
            prediction = model.predict(self.image)
            disease = np.argmax(prediction)
            if disease == 0:
                result = "Bệnh bạc lá"
            elif disease == 1:
                result = "Bệnh đốm nâu"
            elif disease == 2:
                result = "Bệnh tungro"
            elif disease == 3:
                result = "Bệnh cháy lá"
            self.result_label.config(text="Dự đoán: " + result)
        else:
            messagebox.showerror("Lỗi", "Vui lòng chọn hình ảnh trước khi dự đoán.")

root = tk.Tk()
app = PlantDiseaseApp(root)
root.mainloop()