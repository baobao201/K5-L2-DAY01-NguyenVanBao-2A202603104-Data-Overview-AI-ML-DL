# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy: 06/11/2026**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics: Python**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: class_id = 468, class_name = cab, rank = 1, score = 0.510915, taxonomy_name = ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
Record này mô tả cả bức ảnh chứ không phải một vật thể riêng lẻ. Vì đây là tác vụ Image Classification (phân loại ảnh).
3. Danh sách tên lớp (class list) do đâu mà có: Danh sách lớp không phải do model tự nghĩ ra. Danh sách lớp được định nghĩa sẵn trong taxonomy ImageNet-1K đi kèm với checkpoint yolo11n-cls.pt.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
Cần ghi đầy đủ cả ba trường vì:
- class_id: là mã định danh duy nhất của lớp. Tránh nhầm lẫn khi các lớp có tên gần giống nhau.
- class_name: Giúp con người hiểu được ý nghĩa của lớp.
- taxonomy_name: Cho biết lớp đó thuộc bộ phân loại nào.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
Guideline cần quy định rõ:
- Thế nào là đối tượng chính (primary subject).
- Quy tắc ưu tiên khi có nhiều vật thể xuất hiện.
- Cách xử lý các ảnh có nhiều chủ đề ngang nhau.
- Có chọn vật thể lớn nhất, nổi bật nhất hay quan trọng nhất về ngữ nghĩa hay không.
6. Vì sao không được lấy `score` của model làm ground truth?
Không được dùng score = 0.510915 làm ground truth vì: Score chỉ thể hiện độ tự tin của mô hình. Độ tự tin cao không đồng nghĩa với việc dự đoán đúng. Mô hình có thể dự đoán sai nhưng vẫn có confidence cao.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=person ; `score`=0.91 ; `bbox_xyxy`=[684, 94, 882, 643] ; `bbox_width`≈ 198 pixel ; `bbox_height`≈ 549 pixel
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
Bounding box của đối tượng person có:
- Góc trên trái tại khoảng (684, 94)
- Góc dưới phải tại khoảng (882, 643)
- Chiều rộng khoảng 198 pixel
- Chiều cao khoảng 549 pixel
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
- threshold = 0.20 → 17 predictions ['person', 'bowl', 'bowl', 'oven', 'oven', 'person', 'bowl', 'bowl', 'cup', 'cup', 'bowl', 'spoon', 'potted plant', 'spoon', 'dining table', 'spoon', 'bottle']
- threshold = 0.60 → 6 predictions ['person', 'bowl', 'bowl', 'oven', 'oven', 'person']
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
Khi hạ ngưỡng confidence, hệ thống giữ lại nhiều box hơn.
Reviewer phải:
- Kiểm tra các box có score thấp.
- Loại bỏ các box nhận diện sai (false positive).
- Kiểm tra box trùng lặp.
- Kiểm tra nhãn lớp có đúng không.
- Kiểm tra box có bao đúng đối tượng hay không.
Nói cách khác, ngưỡng thấp giúp giảm bỏ sót đối tượng nhưng làm tăng khối lượng kiểm duyệt của reviewer.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
Vẽ bounding box nhỏ nhất có thể nhưng vẫn bao phủ toàn bộ phần nhìn thấy được của đối tượng. Các cạnh của box phải sát biên ngoài của vật thể, không để quá nhiều khoảng trống nền bên trong box.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
Trong ảnh kitchen có thể thấy:
- Người ở góc trái ảnh chỉ xuất hiện một phần cơ thể.
- Đối tượng này bị cắt mép ảnh.
Mình sẽ đưa tình huống này vào guideline, không cần escalation.
Quy tắc: Nếu đối tượng bị cắt bởi mép ảnh nhưng vẫn nhìn thấy rõ và nhận diện được lớp đối tượng, vẫn gán nhãn và vẽ box bao quanh toàn bộ phần nhìn thấy được của đối tượng.
## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-001 ; `class_name`=person ; `score`=0.899318 ; số điểm polygon=`polygon_point_count`=348 ; 3–5 điểm đầu của `polygon_xy`=[446.0, 70.0]
[445.0, 71.0]
[444.0, 71.0]
[443.0, 72.0]
[442.0, 72.0]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
Bounding box chỉ cho biết vùng hình chữ nhật bao quanh đối tượng thông qua tọa độ: [x_min, y_min, x_max, y_max] Polygon cung cấp đường biên chi tiết của vật thể bằng nhiều điểm tọa độ liên tiếp. Polygon cho phép mô tả chính xác đầu, vai, cánh tay, tạp dề và chân của người đó. Vì vậy polygon thể hiện hình dạng thực của đối tượng tốt hơn box và giúp phân biệt đâu là đối tượng, đâu là nền.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
instance_id dùng để định danh một đối tượng cụ thể trong ảnh.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
Vẽ polygon bám sát đường biên nhìn thấy được của vật thể. Chỉ bao gồm các pixel thuộc đối tượng và không mở rộng sang nền hoặc vật thể khác. Với các phần bị che khuất, chỉ vẽ phần còn nhìn thấy được.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
Trong ảnh kitchen, đối tượng: person (kitchen-001) đứng trước khu vực bếp và bị chồng lấn với nhiều vật dụng phía sau như: nồi, chảo, dụng cụ treo tường
Tại vùng vai và hai bên cơ thể, ranh giới giữa người và các vật thể phía sau không hoàn toàn rõ ràng. Trường hợp này mình sẽ xử lý bằng guideline thay vì escalation.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
|---------|---------|---------|---------|---------|
| Phân loại ảnh | Một nhãn duy nhất cho toàn bộ ảnh (ví dụ: `cab`, `bus`, `street scene`). | Ảnh giao thông chứa nhiều đối tượng như xe buýt, ô tô và người đi bộ nên khó xác định lớp đại diện duy nhất. JSON cho thấy model dự đoán `cab` với `score = 0.510915`, nhưng đây chỉ là prediction. | Chọn một lớp duy nhất cho toàn bộ ảnh theo đúng guideline và taxonomy. | Kiểm tra lớp được chọn có phản ánh đúng nội dung chính của ảnh và có tuân thủ guideline hay không. |
| Phát hiện vật thể | Mỗi đối tượng có một nhãn lớp và một bounding box (`class + bbox`). | Đối tượng nhỏ, bị che khuất hoặc chồng lấn dễ bị bỏ sót. Bounding box có thể quá rộng hoặc quá hẹp. | Xác định từng đối tượng cần gán nhãn, vẽ bounding box và gán đúng class. | Kiểm tra có bỏ sót đối tượng không, class có đúng không và bounding box có bao phủ hợp lý vật thể hay không. |
| Instance segmentation | Mỗi đối tượng có một nhãn lớp và một polygon/mask (`class + mask`). | Biên vật thể phức tạp, bị che khuất hoặc lẫn với nền nên khó vẽ chính xác. | Vẽ polygon hoặc mask bám sát biên thực của từng đối tượng và gán đúng class. | Kiểm tra mask có bám sát vật thể, không ăn vào nền và không bỏ sót các phần quan trọng của đối tượng. |

### 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì?

- Chỉ sử dụng dữ liệu được cung cấp cho mục đích học tập trong bài lab, không chia sẻ, sao chép hoặc phát tán dữ liệu ra ngoài phạm vi được cho phép.

2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?

- Tôi sẽ dừng xử lý dữ liệu đó ngay lập tức và báo cho Lab Coach, giảng viên phụ trách hoặc BTC AI20K để được hướng dẫn xử lý phù hợp.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
