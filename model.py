import torch
from ultralytics import YOLO
import cv2
import multiprocessing

class Yolo(torch.nn.Module):
    def __init__(self, model_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_path = model_path
        self.model = self.prepare()

    def prepare(self):
        return YOLO(self.model_path)

    def predict(self, input, device):
        Yolo_model = self.model.to(device)
        Yolo_model.eval()

        imgsz = 640
        conf_thres = 0.5
        iou_thres = 0.7

        output = Yolo_model.predict(
            source = input,   # uses the 'test' key in data.yaml if present
            task = "detect",
            imgsz = imgsz,
            conf = conf_thres,
            iou = iou_thres,
            device = device,
            save = False,            # save annotated images to runs/predict
            save_txt = False        # skip saving YOLO-format txt preds
        )

        pred = output[0]
        img = cv2.imread(input)
        for box in pred.boxes.data:
            x1, y1, x2, y2, score, cls_id = box.tolist()
            cls_name = Yolo_model.names[int(cls_id)]
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img, f"{cls_name} {score:.2f}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return img, pred

if __name__ == "__main__":
    multiprocessing.freeze_support()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Yolo("runs/detect/yolo11n_ver4/weights/best.pt")
    output_img, prediction  = model.predict("hhh.jpg", device)

    cv2.imshow("YOLOv11 Prediction", output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()