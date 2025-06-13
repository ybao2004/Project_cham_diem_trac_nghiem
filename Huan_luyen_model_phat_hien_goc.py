import os
import shutil
import random
import xml.etree.ElementTree as ET
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt
import time
import cv2
from tqdm import tqdm  # Thêm thư viện tqdm để hiển thị tiến trình

def ReSize(img, ChieuCao):
    if ChieuCao is not None:
        target_height = ChieuCao
    else:
        target_height = 500
    original_height, original_width = img.shape[0], img.shape[1]
    if original_height == 0:
        print("Lỗi: Chiều cao ảnh gốc không hợp lệ (bằng 0).")
        return img
    scale = target_height / original_height
    new_width = int(original_width * scale)
    new_size = (new_width, target_height)
    interpolation_method = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized_img = cv2.resize(img, new_size, interpolation=interpolation_method)
    return resized_img

# Định nghĩa hàm chuẩn bị dữ liệu
def chuan_bi_du_lieu(input_dir, output_dir):
    """Chuẩn bị dữ liệu: chuyển XML sang YOLO format và chia train/val."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    images_dir = os.path.join(output_dir, 'images')
    labels_dir = os.path.join(output_dir, 'labels')
    for dir_path in [images_dir, labels_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    train_images = os.path.join(images_dir, 'train')
    val_images = os.path.join(images_dir, 'val')
    train_labels = os.path.join(labels_dir, 'train')
    val_labels = os.path.join(labels_dir, 'val')
    for dir_path in [train_images, val_images, train_labels, val_labels]:
        os.makedirs(dir_path, exist_ok=True)
    
    image_files = [f for f in os.listdir(input_dir) if f.endswith('.jpg')]
    random.shuffle(image_files)
    split_index = int(len(image_files) * 0.8)
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]
    
    def xu_ly_anh(image_file, split):
        src_path = os.path.join(input_dir, image_file)
        dst_path = os.path.join(output_dir, 'images', split, image_file)
        shutil.copy(src_path, dst_path)
        
        base_name = os.path.splitext(image_file)[0]
        xml_path = os.path.join(input_dir, base_name + '.xml')
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        
        label_path = os.path.join(output_dir, 'labels', split, base_name + '.txt')
        with open(label_path, 'w') as f:
            for obj in root.findall('object'):
                name = obj.find('name').text
                class_id = int(name) - 1
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                
                center_x = ((xmin + xmax) / 2) / width
                center_y = ((ymin + ymax) / 2) / height
                box_width = (xmax - xmin) / width
                box_height = (ymax - ymin) / height
                
                f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}\n")
    
    print("Đang chuẩn bị dữ liệu...")
    for image_file in train_files:
        xu_ly_anh(image_file, 'train')
    for image_file in val_files:
        xu_ly_anh(image_file, 'val')
    
    data_yaml_content = """
                        train: images/train
                        val: images/val
                        nc: 4
                        names: ['1', '2', '3', '4']
                        """
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        f.write(data_yaml_content)
    
    print("Đã chuẩn bị dữ liệu thành công.")

# Định nghĩa hàm huấn luyện mô hình
def huan_luyen_mo_hinh(output_dir):
    """Huấn luyện mô hình YOLOv8 và lưu thông tin huấn luyện."""
    print("Đang tải mô hình YOLOv8s...")
    model = YOLO('yolov8s.pt') # n, s, m, l, x --> Độ lớn mô hình
    
    start_time = time.time()
    
    print("Đang huấn luyện mô hình...")
    model.train(
        data=os.path.join(output_dir, 'data.yaml'),
        epochs=100,
        patience=50,
        batch=-1, # tự động
        device=0, # Sử dụng GPU - cần tải thư viện cho GPU. (Pytorch 12.1)
        imgsz=640,
        project=output_dir,
        name='train'
    )
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Đã huấn luyện mô hình thành công trong {total_time / 60:.2f} phút.")
    
    results_path = os.path.join(output_dir, 'train', 'results.csv')
    if os.path.exists(results_path):
        print(f"Kết quả huấn luyện được lưu tại: {results_path}")
        ve_bieu_do(results_path, output_dir)
    else:
        print("Không tìm thấy file results.csv")

# Định nghĩa hàm vẽ biểu đồ
def ve_bieu_do(results_path, output_dir):
    """Vẽ biểu đồ loss và mAP từ file results.csv."""
    df = pd.read_csv(results_path)
    
    plt.figure(figsize=(10, 5))
    plt.plot(df['epoch'], df['train/box_loss'], label='Train Box Loss')
    plt.plot(df['epoch'], df['train/cls_loss'], label='Train Class Loss')
    plt.plot(df['epoch'], df['val/box_loss'], label='Val Box Loss')
    plt.plot(df['epoch'], df['val/cls_loss'], label='Val Class Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
    plt.close()
    
    plt.figure(figsize=(10, 5))
    plt.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@50')
    plt.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@50-95')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.title('mAP over Epochs')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'map_plot.png'))
    plt.close()
    
    print(f"Biểu đồ đã được lưu tại {output_dir}")

def du_doan_anh(duong_dan_anh_thu, output_dir):
    # Kiểm tra xem mô hình đã tồn tại chưa
    model_path = os.path.join(output_dir, 'train', 'weights', 'best.pt')
    if not os.path.exists(model_path):
        print(f"Lỗi: Không tìm thấy mô hình tại {model_path}. Vui lòng huấn luyện mô hình trước.")
        return

    # Kiểm tra xem ảnh thử nghiệm có tồn tại không
    if not os.path.exists(duong_dan_anh_thu):
        print(f"Lỗi: Không tìm thấy ảnh thử nghiệm tại {duong_dan_anh_thu}")
        return

    # Tạo thư mục kết quả nếu chưa tồn tại
    ket_qua_dir = os.path.join(output_dir, 'ket_qua')
    os.makedirs(ket_qua_dir, exist_ok=True)
    output_path = os.path.join(ket_qua_dir, os.path.basename(duong_dan_anh_thu))

    # Thực hiện dự đoán
    try:
        model = YOLO(model_path)
        # Dự đoán và lưu kết quả (bao gồm cả ảnh đã vẽ)
        results = model.predict(duong_dan_anh_thu, save=False)

        # Lấy kết quả đầu tiên (vì chỉ có một ảnh)
        result = results[0]

        # In thông tin kết quả để kiểm tra
        boxes = result.boxes.xyxy  # Tọa độ bounding box
        classes = result.boxes.cls  # Nhãn lớp
        scores = result.boxes.conf  # Độ tin cậy
        print(f"Đã phát hiện {len(boxes)} đối tượng:")
        for box, cls, score in zip(boxes, classes, scores):
            print(f"Box: {box}, Class: {int(cls)}, Score: {score:.2f}")

        # Tính tọa độ tâm của các đối tượng và lưu vào list (đã làm tròn)
        toa_do = []
        for box in boxes:
            xmin, ymin, xmax, ymax = box.tolist()
            toa_do_tam = (round((xmin + xmax) / 2), round((ymin + ymax) / 2))  # Tọa độ tâm (x, y) đã làm tròn
            toa_do.append(toa_do_tam)

        # Hiển thị list tọa độ đã làm tròn
        print(f"Tọa độ tâm của các đối tượng đã phát hiện (đã làm tròn): {toa_do}")

        # Lấy ảnh đã được vẽ bounding box từ kết quả
        img_array = result.plot()  # Lấy ảnh với các bounding box đã vẽ
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)  # Chuyển sang RGB để hiển thị
        #cv2.imshow('n', img_array)
        #cv2.waitKey()

        # Lưu ảnh đã đánh dấu vào thư mục kết quả
        cv2.imwrite(output_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
        print(f"Ảnh đã đánh dấu được lưu tại: {output_path}")

        # Hiển thị ảnh
        plt.figure(figsize=(10, 10))
        plt.imshow(img_rgb)
        plt.title("Ảnh thử nghiệm với các điểm đã đánh dấu")
        plt.axis('off')
        plt.show()
        print("Ảnh đã được hiển thị trực tiếp.")

    except Exception as e:
        print(f"Lỗi khi dự đoán: {str(e)}")

def du_doan_anh_nhieu_anh_tu_dong(thu_muc_anh, output_dir, thoi_gian_hien_thi_ms=1000):
    mau_tam = (0, 0, 255)
    ban_kinh_tam = 5
    do_day_tam = -1
    mau_bbox= (0, 255, 0)
    do_day_bbox = 2
    mau_chu = (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale_chu = 0.7
    do_day_chu = 2

    # Kiểm tra xem mô hình đã tồn tại chưa
    model_path = os.path.join(output_dir, 'train', 'weights', 'best.pt')
    if not os.path.exists(model_path):
        print(f"Lỗi: Không tìm thấy mô hình tại {model_path}. Vui lòng huấn luyện mô hình trước.")
        return

    # Kiểm tra xem thư mục ảnh thử nghiệm có tồn tại không
    if not os.path.exists(thu_muc_anh):
        print(f"Lỗi: Không tìm thấy thư mục ảnh tại {thu_muc_anh}")
        return

    # Tạo thư mục kết quả nếu chưa tồn tại
    ket_qua_dir = os.path.join(output_dir, 'ket_qua')
    os.makedirs(ket_qua_dir, exist_ok=True)

    # Lấy danh sách tất cả các tệp trong thư mục
    all_files = os.listdir(thu_muc_anh)
    # Lọc ra chỉ các tệp ảnh
    anh_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    if not anh_files:
        print(f"Không tìm thấy tệp ảnh nào trong thư mục: {thu_muc_anh}")
        return

    # Tải mô hình
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {str(e)}")
        return

    print(f"Bắt đầu xử lý và hiển thị tự động {len(anh_files)} ảnh trong thư mục: {thu_muc_anh} (mỗi ảnh hiển thị trong {thoi_gian_hien_thi_ms / 1000} giây)")
    for ten_anh in tqdm(anh_files, desc = "Đang xử lý và hiển thị ảnh"):
        duong_dan_anh_thu = os.path.join(thu_muc_anh, ten_anh)
        output_path = os.path.join(ket_qua_dir, f"marked_full_{ten_anh}")

        try:
            # Thực hiện dự đoán
            results = model.predict(duong_dan_anh_thu, save=False, verbose=False)
            result = results[0]  # Lấy kết quả đầu tiên
            img_goc = cv2.imread(duong_dan_anh_thu) # Lấy ảnh gốc để vẽ lên
            if img_goc is None:
                print(f"Lỗi: Không thể đọc ảnh {ten_anh}")
                continue
            img_copy = img_goc.copy()  # Tạo bản sao để vẽ

            # Lấy thông tin bounding box, class và score
            boxes = result.boxes.xyxy
            classes = result.boxes.cls
            scores = result.boxes.conf

            print(f"\nĐã phát hiện {len(boxes)} đối tượng trong ảnh: {ten_anh}")
            toa_do_tam_tat_ca = []
            for box, cls, score in zip(boxes, classes, scores):
                xmin, ymin, xmax, ymax = [int(coord) for coord in box.tolist()]
                toa_do_tam = (round((xmin + xmax) / 2), round((ymin + ymax) / 2))
                toa_do_tam_tat_ca.append(toa_do_tam)
                score_text = f"{score:.2f}"
                print(f"  Box: ({xmin}, {ymin}, {xmax}, {ymax}), Class: {int(cls)}, Score: {score_text}, Tâm: {toa_do_tam}")

                # Vẽ bounding box
                cv2.rectangle(img_copy, (xmin, ymin), (xmax, ymax), mau_bbox, do_day_bbox)
                # Vẽ hình tròn tại tâm
                cv2.circle(img_copy, toa_do_tam, ban_kinh_tam, mau_tam, do_day_tam)
                # Vẽ score lên ảnh
                text_location = (xmin, ymin - 10 if ymin - 10 > 10 else ymin + 20) # Đảm bảo chữ không bị ra ngoài ảnh
                cv2.putText(img_copy, score_text, text_location, font, scale_chu, mau_chu, do_day_chu, cv2.LINE_AA)

            # Lưu ảnh đã đánh dấu (bao gồm tâm, bounding box và score)
            img_copy = ReSize(img_copy, 720)
            cv2.imwrite(output_path, img_copy)
            print(f"  Ảnh đã đánh dấu (với tâm, bounding box và score) được lưu tại: {output_path}")

            # Hiển thị ảnh bằng cv2.imshow
            #cv2.imshow(f"Ảnh: {ten_anh} (Đã đánh dấu)", img_copy)
            cv2.imshow(f"Ảnh trong thư mục", img_copy)
            cv2.waitKey(thoi_gian_hien_thi_ms)  # Chờ một khoảng thời gian
            #cv2.destroyAllWindows()

        except Exception as e:
            print(f"Lỗi khi dự đoán ảnh {ten_anh}: {str(e)}")


# Thực thi
if __name__ == "__main__":
    input_dir = r"C:\Users\..." # Thư mục chứa ảnh và file gán nhãn .xml
    output_dir = r"D:\ket_qua_huan_luyen" # Thư mục chứa kết quả huấn luyện
    thu_muc_chua_anh = r"D:\Thu_nghiem" # Thư mục chứa kết quả huấn luyện
    
    # Bỏ comment nếu cần chuẩn bị dữ liệu và huấn luyện lại
    chuan_bi_du_lieu(input_dir, output_dir)
    huan_luyen_mo_hinh(output_dir)
    
    # Dự đoán trên ảnh thử nghiệm
    #du_doan_anh(r"D:\Thu_nghiem"\hinh_1.jpg", output_dir)
    #du_doan_anh_nhieu_anh_tu_dong(thu_muc_chua_anh, output_dir, 200)
