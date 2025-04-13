from ultralytics import YOLO

def main():
    # Load a model
    model = YOLO("yolo11n.pt")  # Ensure this path is correct

    # Train the model
    train_results = model.train(
        data="config.yaml",  # path to dataset YAML
        epochs=100,  # number of training epochs
        imgsz=640,  # training image size
        device="0",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    )

    # Evaluate model performance on the validation set
    metrics = model.val()

    # Perform object detection on an image
    results = model("C:\\Users\\LX\\Downloads\\frambuesa.jpg")
    results[0].show()

    # Export the model to ONNX format
    path = model.export(format="onnx")  # return path to exported model

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
