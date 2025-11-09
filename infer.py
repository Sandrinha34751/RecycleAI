import json
import time
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk
from tkinter import Tk, Label, Button, filedialog, Frame, Canvas
from filters import apply_filters_as_channels

def load_model(path='recycle_model.h5'):
    return tf.keras.models.load_model(path)

def predict_image(model, img_path, size=(128,128), class_json='class_indices.json'):
    img = Image.open(img_path).convert('RGB')
    arr = apply_filters_as_channels(img, size=size)
    x = np.expand_dims(arr, axis=0)
    start = time.time()
    preds = model.predict(x, verbose=0)[0]
    infer_time = (time.time() - start) * 1000
    with open(class_json,'r') as f:
        class_indices = json.load(f)
    inv = {v:k for k,v in class_indices.items()}
    cls = inv[np.argmax(preds)]
    return img, cls, preds, inv, infer_time

def choose_image():
    file_path = filedialog.askopenfilename(title="Selecione uma imagem",
        filetypes=[("Imagens", "*.png *.jpg *.jpeg *.bmp *.webp *.tiff *.gif *.jfif *.avif *.heic *.heif *.*")]
    )
    if not file_path:
        return
    
    canvas_bg.pack_forget()
    root.configure(bg="#ffffff")
    content.configure(bg="#ffffff")

    for widget in content.winfo_children():
        if isinstance(widget, Button):
            continue  
        widget.configure(bg="#ffffff")

    img, pred_classe, probs, inv_map, infer_time = predict_image(model, file_path)
    img_resized = img.resize((280, 280))
    img_tk = ImageTk.PhotoImage(img_resized)
    image_label.config(image=img_tk, bg="#ffffff")
    image_label.image = img_tk

    result_label.config(text=f"♻️ Predição: {pred_classe}", fg="#2e7d32", bg="#ffffff")
    time_label.config(text=f"🕒 Tempo de Inferência: {infer_time:.1f} ms", fg="red", bg="#ffffff")

    show_probabilities(probs, inv_map)

def show_probabilities(probs, inv_map):
    for widget in bars_frame.winfo_children():
        widget.destroy()

    Label(bars_frame, text="📊 Probabilidades", font=("Arial", 13, "bold"), bg="#ffffff", fg="#2e7d32").pack(pady=(0,5))

    max_idx = np.argmax(probs)
    for i in range(len(probs)):
        classe = inv_map[i]
        valor = probs[i]
        row = Frame(bars_frame, bg="#ffffff")
        row.pack(fill="x", pady=3)
        Label(row, text=f"{classe}", font=("Arial", 11), bg="#ffffff").pack(side="left", padx=5)
        bar_width = int(valor * 200)
        bar = Canvas(row, width=200, height=18, bg="#dddddd", highlightthickness=0)
        bar.pack(side="left")
        cor = "#4CAF50" if i == max_idx else "#3498db"
        bar.create_rectangle(0, 0, bar_width, 18, fill=cor)
        Label(row, text=f"{valor*100:.1f}%", font=("Arial", 10), bg="#ffffff").pack(side="left", padx=5)

root = Tk()
root.title("Classificador de Reciclagem")
root.geometry("600x550")


canvas_bg = Canvas(root, width=600, height=550)
canvas_bg.pack(fill="both", expand=True)
for i in range(550):
    r = 220 - int(i/4)
    g = 255 - int(i/8)
    b = 220 - int(i/10)
    color = f"#{r:02x}{g:02x}{b:02x}"
    canvas_bg.create_line(0, i, 600, i, fill=color)

content = Frame(root, bg="#f8fff8")
content.place(relx=0.5, rely=0.5, anchor="center")

model = load_model()

Label(content, text="♻️ Classificador de Materiais Recicláveis ♻️", 
      font=("Arial", 16, "bold"), bg="#f8fff8", fg="#2e7d32").pack(pady=10)

image_label = Label(content, bg="#f8fff8")
image_label.pack(pady=10)

result_label = Label(content, text="📸 Selecione uma imagem", font=("Arial", 14), bg="#f8fff8", fg="#1b5e20")
result_label.pack(pady=5)

time_label = Label(content, text="", font=("Arial", 12), bg="#f8fff8", fg="#33691e")
time_label.pack(pady=2)

bars_frame = Frame(content, bg="#f8fff8")
bars_frame.pack(pady=10)


btn_select = Button(
    content,
    text="📂 Selecionar Imagem",
    command=choose_image,
    font=("Arial", 12, "bold"),
    bg="#43a047",          
    fg="white",
    activebackground="#2e7d32",
    activeforeground="white",
    padx=15,
    pady=5,
    relief="raised",
    bd=3
)
btn_select.pack(pady=15)

root.mainloop()
