# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

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

1. Chép record hạng 1: `class_id`= 468 ; `class_name`="cab"; `rank`=1 ; `score`= 0.510915 ; `taxonomy_name`="ImageNet-1K"

2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
Record này nói về cả bức ảnh vì "task": "image_classification": gán nhãn tổng thể cho toàn bộ ảnh

3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
Mô hình không tự nghĩ ra tên lớp mà chỉ chọn ra các lớp có độ tự tin cao nhất từ "taxonomy_name": "ImageNet-1K" có sẵn

4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
Đồng bộ và tránh sai lệch dữ liệu
"class_id": tránh gõ sai tên lớp
"class_name": biết được tên danh mục vật thể "cab"
"taxonomy_name": xác định đúng lớp "cab" được dùng từ bộ dữ liệu nào

5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
vd trong ảnh giao thông cần chỉ rõ chọn vật thể có diện tích lớn nhất ở trung tâm ảnh, hoặc chọn nhãn bao quát nhất

6. Vì sao không được lấy `score` của model làm ground truth?
mô hình đoán sai với độ tin cậy cao nhất, nhận diện "cab" là cao nhất khi đây là ảnh của đoạn giao thông đông đúc với nhiều phương tiện khác

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`= person ; `score`=0.91 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`= 113.58 ; `bbox_height`= 279.68

2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
Góc trên-trái ( 𝑥𝑚𝑖𝑛 , 𝑦𝑚𝑖𝑛  ): Nằm tại tọa độ pixel ( 385.33 , 69.24 ) (385.33,69.24), nghĩa là cách mép trái của ảnh 385.33 px 385.33 px và cách mép trên của ảnh 69.24 px 69.24 px (gốc ( 0 , 0 ) (0,0) ở góc trên-cùng bên trái). Góc dưới-phải ( 𝑥𝑚𝑎𝑥 , 𝑦𝑚𝑎𝑥 ): Nằm tại tọa độ pixel ( 498.92 , 348.92 ) (498.92,348.92), cách mép trái 498.92 px 498.92 px và cách mép trên 348.92 px 348.92 px. Kích thước hộp: Chiều rộng là 113.58 px 113.58 px ( 498.92 − 385.33 498.92−385.33) và chiều cao là 279.68 px 279.68 px ( 348.92 − 69.24 348.92−69.24). Hộp này bao trọn vẹn phần nhìn thấy của người phụ nữ trong ảnh.

3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
ngưỡng 0.20: 17 vật thể
ngưỡng 0.60: 6 vật thể

4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
khi hạ ngưỡng số lượng vật thể tăng lên rất nhiều
người "reviewer" phải sàng lọc lại, xoá bỏ trùng lặp

5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
Hộp bao phải vẽ sát vào vật thể nhất có thể, không để thừa khoảng trống của nền, không cắt lẹm vào vật thể

6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
Chiếc bát ở góc dưới cùng bên trái (bowl 0.72) bị cắt mépn do rìa ảnh
nên đưa lên lab coach để đảm bảo thống nhất một quy chuẩn chung duy nhất cho toàn bộ dự án, tránh việc dữ liệu bị gán nhãn không đồng nhất.


## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`= kitchen-006 ; `class_name`= "bowl"; `score`= 0.528202 ; số điểm polygon=`polygon_point_count`= 84 ; 3–5 điểm đầu của `polygon_xy`= [
        29.0,
        303.0
      ],
      [
        28.0,
        304.0
      ],
      [
        26.0,
        304.0
      ],

2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
Polygon cung cấp thông tin chính xác về đường viền (contour) và ranh giới thực tế đến từng pixel của vật thể, giúp mô hình hiểu rõ hình dáng hình học thực sự thay vì chỉ biết giới hạn khung hình chữ nhật bao quanh như Bounding Box.

3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
instance_id dùng để phân biệt các cá thể (thực thể) riêng biệt thuộc cùng một lớp (class) trong cùng một bức ảnh.
• Nó KHÔNG PHẢI class ID: class_id là mã định danh cho loại danh mục. Ví dụ: Lớp person có class_id = 0. Nếu trong ảnh có 3 người, cả 3 người này đều dùng chung một mã class_id = 0. Ngược lại, instance_id sẽ cấp mã riêng cho từng người (ví dụ: người A là instance_id = 1, người B là instance_id = 2) để máy tính biết đây là hai cơ thể tách biệt chứ không phải một.
• Nó KHÔNG PHẢI tracking ID: tracking_id là mã định danh dùng để theo dõi một vật thể xuyên suốt qua nhiều khung hình (video). Nếu người A đi từ khung hình 1 sang khung hình 2, họ vẫn giữ nguyên một tracking_id. Trong khi đó, instance_id chỉ có giá trị độc lập trong phạm vi của một bức ảnh duy nhất. Sang một bức ảnh khác, hệ thống sẽ đánh số instance_id lại từ đầu, không quan tâm người đó có phải là người ở ảnh trước hay không.

4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
Người gán nhãn phải chấm các điểm polygon dọc theo đường viền ranh giới thực tế của vật thể. không được cắt lẹm vào thân vật thể (làm mất pixel của vật thể) và không được lấn ra ngoài nền (background) quá 1 pixel. Đối với các đoạn bo cong, phải tăng mật độ chấm điểm (khoảng cách giữa các điểm từ 2–3 pixel) để đảm bảo mặt nạ (mask) sau khi tạo ra mịn mậy và ôm khít hoàn toàn cấu trúc hình học của thực thể.

5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
vùng chiếc lò nướng (oven) bị che khuất một phần lớn phía sau lưng người
lựa chọn phương án sử dụng Guideline
Các dự án phân vùng thực thể tiêu chuẩn luôn có một mục gọi là "Quy tắc xử lý chồng lấp (Overlap/Occlusion Rule)" trong Guideline. Guideline thường sẽ quy định sẵn: "Khi vật thể A che khuất vật thể B, hãy vẽ polygon của vật thể A đầy đủ phần nhìn thấy, còn vật thể B chỉ vẽ phần hiển thị được (không tự suy đoán phần bị ẩn)". Nếu trường hợp này đã có quy tắc, annotator chỉ cần làm theo để đảm bảo tiến độ công việc và tính nhất quán.


## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh |1 nhãn (class_id, class_name) cho toàn bộ bức ảnh theo taxonomy cố định (vd: ImageNet-1K).  |Ảnh traffic có nhiều đối tượng (xe buýt, ô tô, người); mô hình dự đoán cab chỉ là một phần nhỏ, khó chọn 1 nhãn duy nhất đại diện cho cả ảnh.  |Đọc guideline để xác định chủ thể chính (dominant subject) theo quy tắc ưu tiên; gán đúng 1 nhãn trong taxonomy.  |Kiểm tra nhãn được gán có đúng theo quy tắc ưu tiên chủ thể trong guideline không; kiểm tra các ca ảnh đa chủ đề (ambiguous).  |
| Phát hiện vật thể |Tập hợp các hộp [x_min, y_min, x_max, y_max] kèm class_id cho từng vật thể nhìn thấy.  |Lò nướng và bát bị cắt mép ảnh (x≈0); ở ngưỡng 0.20 sinh ra nhiều box rác (spoon, potted plant) và box trùng lặp.  |Vẽ bounding box ôm sát từng vật thể theo danh mục taxonomy; xử lý vật bị che khuất/cắt mép theo ngưỡng quy định.  |Soát lỗi sót vật thể (False Negative), xóa bỏ hộp ảo/rác (False Positive), kiểm tra độ khít (tightness) của các cạnh hộp.  |
| Instance segmentation |Tập hợp các đa giác polygon_xy = [[x1, y1], [x2, y2], ...] kèm instance_id và class_id cho từng cá thể.  |Biên tiếp xúc giữa người và bàn bếp bị mờ; các vật thể nhỏ chồng lấn khó phân định chính xác ranh giới pixel.  |Dùng công cụ viền khít từng đối tượng; phân biệt ranh giới giữa các vật thể chạm nhau và gán mã instance ID riêng.  |Phóng to kiểm tra đường biên mask có bị ăn lẹm ra nền hoặc cắt phạm vào vật thể không; kiểm tra tính độc lập của từng instance.  |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
Chỉ sử dụng tập ảnh mẫu công khai COCO và checkpoint được cấp phép trong bài lab, tuyệt đối không tải lên dữ liệu cá nhân, thông tin định danh (họ tên, MSSV, email, số điện thoại) hoặc dữ liệu nội bộ lên notebook, báo cáo hay mã nguồn mở.

2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
Tôi sẽ dừng ngay việc xử lý và báo cáo ngay lập tức cho Lab Coach / Giảng viên phụ trách môn học để được hướng dẫn xử lý và cô lập dữ liệu theo quy trình an toàn.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
