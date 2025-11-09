from model import train
import os

if __name__ == "__main__":
    base = os.path.dirname(__file__) or '.'
    train_dir = os.path.join(base, "dataset", "train")
    val_dir = os.path.join(base, "dataset", "val")

    img_size = (128, 128)
    batch_size = 32
    epochs = 20
    save_path = "recycle_model.h5"

    model, history = train(train_dir, val_dir, use_filters=True,
                           img_size=img_size, batch_size=batch_size,
                           epochs=epochs, save_path=save_path)
    print("Treino finalizado. model salvo em:", save_path)

    