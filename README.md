# BÁO CÁO ĐỒ ÁN
# ỨNG DỤNG CÁC THUẬT TOÁN TÌM KIẾM TRONG TRÍ TUỆ NHÂN TẠO VÀO BÀI TOÁN 8-PUZZLE

![8-puzzle](https://i.imgur.com/example-image.png) *(hình ảnh có thể được thay thế)*

## THÔNG TIN ĐỒ ÁN
- **Tên đồ án:** Ứng dụng các thuật toán tìm kiếm AI vào bài toán 8-puzzle
- **Môn học:** Trí tuệ nhân tạo
- **Học kỳ:** 2 - Năm học 2024-2025
-**Giảng viên hướng dẫn:** TS.Phan Thị Huyền Trang
**Môn học:** Trí Tuệ Nhân Tạo
**Lớp học phần:** ARIN330585_05

- **Sinh viên thực hiện:** Lê Thị Thảo
- **Mã số sinh viên:** 23110321

## MỤC LỤC
1. [Đặt vấn đề](#đặt-vấn-đề)
2. [Cơ sở lý thuyết](#cơ-sở-lý-thuyết)
   - [Bài toán 8-puzzle](#bài-toán-8-puzzle)
   - [Các thuật toán tìm kiếm](#các-thuật-toán-tìm-kiếm)
3. [Thiết kế và cài đặt](#thiết-kế-và-cài-đặt)
   - [Cấu trúc dự án](#cấu-trúc-dự-án)
   - [Công nghệ sử dụng](#công-nghệ-sử-dụng)
4. [Chi tiết thuật toán](#chi-tiết-thuật-toán)
   - [Thuật toán không có thông tin](#thuật-toán-không-có-thông-tin)
     - [Tìm Kiếm Theo Chiều Rộng (BFS)](#1-tìm-kiếm-theo-chiều-rộng-bfs)
     - [Tìm Kiếm Theo Chiều Sâu (DFS)](#2-tìm-kiếm-theo-chiều-sâu-dfs)
     - [Tìm Kiếm Chi Phí Đồng Nhất (UCS)](#3-tìm-kiếm-chi-phí-đồng-nhất-ucs)
     - [Tìm Kiếm Sâu Lặp (IDS)](#4-tìm-kiếm-sâu-lặp-ids)
   - [Thuật toán có thông tin](#thuật-toán-có-thông-tin)
     - [Tìm Kiếm Tham Lam](#1-tìm-kiếm-tham-lam-greedy-best-first-search)
     - [Tìm Kiếm A*](#2-tìm-kiếm-a-a-search)
     - [Tìm Kiếm A* Lặp Sâu](#3-tìm-kiếm-a-lặp-sâu-ida-search)
   - [Thuật toán tìm kiếm cục bộ](#thuật-toán-tìm-kiếm-cục-bộ)
     - [Thuật Toán Di Truyền](#1-thuật-toán-di-truyền-genetic-algorithm)
     - [Leo Đồi và Biến Thể](#2-leo-đồi-hill-climbing-và-biến-thể)
     - [Mô Phỏng Luyện Kim](#3-mô-phỏng-luyện-kim-simulated-annealing)
   - [Thuật toán CSP](#thuật-toán-csp)
     - [Quay Lui](#1-quay-lui-backtracking)
     - [Quay Lui với Kiểm Tra Trước](#2-quay-lui-với-kiểm-tra-trước-forward-checking)
     - [Xung Đột Tối Thiểu](#3-xung-đột-tối-thiểu-min-conflicts)
   - [Thuật toán học tăng cường](#thuật-toán-học-tăng-cường)
     - [Q-Learning](#1-q-learning)
     - [Mạng Q Sâu](#2-mạng-q-sâu-dqn)
     - [SARSA và Độ Dốc Chính Sách](#3-sarsa-và-độ-dốc-chính-sách-policy-gradient)
   - [Thuật toán cho môi trường phức tạp](#thuật-toán-cho-môi-trường-phức-tạp)
     - [Tìm Kiếm Cây AND-OR](#1-tìm-kiếm-cây-and-or)
     - [Tìm Kiếm Trong Môi Trường Quan Sát Một Phần](#2-tìm-kiếm-trong-môi-trường-quan-sát-một-phần)
     - [Tìm Kiếm Trạng Thái Niềm Tin](#3-tìm-kiếm-trạng-thái-niềm-tin-belief-state-search)
5. [Các hàm heuristic](#các-hàm-heuristic)
6. [Đánh giá và so sánh hiệu suất](#đánh-giá-và-so-sánh-hiệu-suất)
7. [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
   - [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
   - [Giao diện đồ họa](#giao-diện-đồ-họa)
8. [Kết luận và hướng phát triển](#kết-luận-và-hướng-phát-triển)
9. [Tài liệu tham khảo](#tài-liệu-tham-khảo)

## ĐẶT VẤN ĐỀ
Bài toán 8-puzzle là một bài toán kinh điển trong lĩnh vực trí tuệ nhân tạo, được sử dụng rộng rãi để minh họa và đánh giá hiệu quả của các thuật toán tìm kiếm. Đồ án này nhằm mục đích triển khai và so sánh hiệu suất của nhiều thuật toán tìm kiếm khác nhau khi ứng dụng vào bài toán 8-puzzle, từ các thuật toán cơ bản đến nâng cao, từ đó đánh giá ưu nhược điểm của từng phương pháp.

Mục tiêu của đồ án:
- Hiểu và cài đặt các thuật toán tìm kiếm trong AI
- So sánh hiệu suất của các thuật toán
- Xây dựng giao diện trực quan để minh họa quá trình giải puzzle
- Phân tích ảnh hưởng của các hàm heuristic đến hiệu quả tìm kiếm

## CƠ SỞ LÝ THUYẾT

### Bài toán 8-puzzle
Bài toán 8-puzzle bao gồm một lưới 3×3 với 8 ô đánh số từ 1 đến 8 và một ô trống. Mục tiêu là sắp xếp lại các ô từ trạng thái ban đầu cho trước đến trạng thái đích bằng cách trượt các ô vào ô trống.

**Trạng thái đích:**
```
1 2 3
4 5 6
7 8 _
```

Bài toán 8-puzzle có một số đặc điểm quan trọng:
- Không gian trạng thái: 9!/2 = 181,440 trạng thái hợp lệ
- Các trạng thái có thể đạt được phụ thuộc vào tính chẵn lẻ của số lần đảo vị trí
- Mỗi trạng thái có tối đa 4 trạng thái kế tiếp (tương ứng với 4 hướng di chuyển của ô trống)

### Các thuật toán tìm kiếm
Các thuật toán tìm kiếm được phân loại thành các nhóm chính:

1. **Thuật toán không có thông tin (Uninformed Search)**: Không sử dụng kiến thức về bài toán ngoài định nghĩa của nó.
2. **Thuật toán có thông tin (Informed Search)**: Sử dụng kiến thức đặc thù về bài toán để cải thiện hiệu suất tìm kiếm.
3. **Thuật toán tìm kiếm cục bộ (Local Search)**: Duy trì một hoặc một số trạng thái hiện tại thay vì duy trì cây tìm kiếm.
4. **Thuật toán thỏa mãn ràng buộc (CSP)**: Giải quyết bài toán bằng cách đáp ứng các ràng buộc giữa các biến.
5. **Thuật toán học tăng cường (Reinforcement Learning)**: Học từ tương tác với môi trường.
6. **Thuật toán cho môi trường phức tạp**: Xử lý các môi trường không xác định hoặc chỉ quan sát được một phần.

## THIẾT KẾ VÀ CÀI ĐẶT

### Cấu trúc dự án
Dự án được tổ chức thành các module chức năng:
- `Uniformed_Search.py`: Triển khai các thuật toán tìm kiếm không có thông tin
- `Informed_Search.py`: Triển khai các thuật toán tìm kiếm có thông tin
- `Local_Search.py`: Triển khai các thuật toán tìm kiếm cục bộ
- `CSPs.py`: Triển khai các thuật toán thỏa mãn ràng buộc
- `Reinforcement_Search.py`: Triển khai các thuật toán học tăng cường
- `Complex_Environments.py`: Triển khai các thuật toán cho môi trường phức tạp
- `Puzzle_GUI.py`: Triển khai giao diện đồ họa người dùng
- `main.py`: Điểm vào cho ứng dụng

### Công nghệ sử dụng
- **Ngôn ngữ lập trình**: Python 3.6+
- **Thư viện đồ họa**: Tkinter
- **Các thư viện khác**: Numpy, Matplotlib (cho biểu đồ so sánh hiệu suất)

## CHI TIẾT THUẬT TOÁN

### Thuật toán không có thông tin
#### 1. Tìm Kiếm Theo Chiều Rộng (BFS)
- **Nguyên lý**: Khám phá tất cả các nút ở độ sâu hiện tại trước khi chuyển sang các nút ở độ sâu tiếp theo
- **Ưu điểm**: Luôn tìm được đường đi ngắn nhất
- **Nhược điểm**: Tiêu tốn nhiều bộ nhớ

#### 2. Tìm Kiếm Theo Chiều Sâu (DFS)
- **Nguyên lý**: Khám phá càng sâu càng tốt theo từng nhánh trước khi quay lui
- **Ưu điểm**: Tiết kiệm bộ nhớ
- **Nhược điểm**: Có thể tìm được đường đi dài và không tối ưu

#### 3. Tìm Kiếm Chi Phí Đồng Nhất (UCS)
- **Nguyên lý**: Khám phá các nút theo thứ tự chi phí đường đi tăng dần
- **Ưu điểm**: Tìm được đường đi có chi phí thấp nhất
- **Nhược điểm**: Chậm hơn một số thuật toán khác

#### 4. Tìm Kiếm Sâu Lặp (IDS)
- **Nguyên lý**: Thực hiện nhiều lần DFS với giới hạn độ sâu tăng dần
- **Ưu điểm**: Tìm được đường đi ngắn nhất với bộ nhớ ít hơn BFS
- **Nhược điểm**: Phải duyệt lại nhiều nút ở mỗi vòng lặp

### Thuật toán có thông tin
#### 1. Tìm Kiếm Tham Lam (Greedy Best-First Search)
- **Nguyên lý**: Sử dụng hàm heuristic để ước tính khoảng cách đến đích, luôn mở rộng nút có vẻ gần đích nhất
- **Ưu điểm**: Rất nhanh so với các thuật toán khác
- **Nhược điểm**: Có thể tìm ra đường đi không tối ưu

#### 2. Tìm Kiếm A* (A* Search)
- **Nguyên lý**: Kết hợp chi phí đường đi (g(n)) và ước lượng heuristic (h(n)), mở rộng các nút có f(n) = g(n) + h(n) thấp nhất
- **Ưu điểm**: Kết hợp tốt nhất giữa tốc độ và tối ưu
- **Nhược điểm**: Cần nhiều bộ nhớ hơn một số thuật toán khác

#### 3. Tìm Kiếm A* Lặp Sâu (IDA* Search)
- **Nguyên lý**: Kết hợp tìm kiếm lặp sâu với tìm kiếm A*
- **Ưu điểm**: Sử dụng ít bộ nhớ hơn A* nhưng vẫn tìm được đường đi tối ưu
- **Nhược điểm**: Có thể chậm hơn A* do phải duyệt lại nhiều nút

### Thuật toán tìm kiếm cục bộ
#### 1. Thuật Toán Di Truyền (Genetic Algorithm)
- **Nguyên lý**: Mô phỏng quá trình tiến hóa tự nhiên, duy trì một quần thể các giải pháp và áp dụng các toán tử lai ghép, đột biến
- **Ưu điểm**: Có khả năng thoát khỏi cực tiểu cục bộ
- **Nhược điểm**: Không đảm bảo tìm được giải pháp tối ưu

#### 2. Leo Đồi (Hill Climbing) và Biến Thể
- **Nguyên lý**: Thuật toán tìm kiếm cục bộ đơn giản, chỉ di chuyển đến trạng thái tốt hơn
- **Biến thể**: Simple, Steepest Ascent, Stochastic Hill Climbing
- **Ưu điểm**: Đơn giản, ít tốn bộ nhớ
- **Nhược điểm**: Dễ bị mắc kẹt ở cực tiểu cục bộ

#### 3. Mô Phỏng Luyện Kim (Simulated Annealing)
- **Nguyên lý**: Cho phép chấp nhận các di chuyển xấu hơn với xác suất giảm dần
- **Ưu điểm**: Có thể tìm được giải pháp gần tối ưu cho các bài toán phức tạp
- **Nhược điểm**: Không đảm bảo tìm được giải pháp tối ưu

### Thuật toán CSP
#### 1. Quay Lui (Backtracking)
- **Nguyên lý**: Gán giá trị tuần tự cho các biến, quay lui khi gặp xung đột
- **Ưu điểm**: Đơn giản, dễ hiểu
- **Nhược điểm**: Có thể chậm với các bài toán phức tạp

#### 2. Quay Lui với Kiểm Tra Trước (Forward Checking)
- **Nguyên lý**: Cải tiến từ quay lui thông thường, loại bỏ sớm các giá trị không hợp lệ
- **Ưu điểm**: Hiệu quả hơn quay lui thông thường
- **Nhược điểm**: Vẫn có thể chậm với một số bài toán

#### 3. Xung Đột Tối Thiểu (Min-Conflicts)
- **Nguyên lý**: Bắt đầu với một giải pháp hoàn chỉnh (có thể có xung đột) và sửa dần các xung đột bằng cách thay đổi giá trị của các biến
- **Cách áp dụng cho 8-puzzle**:
  - **Mô hình dán nhãn**: Mỗi vị trí (0-8) trên bảng là một biến, mỗi số trên bảng (1-8 và ô trống) là một nhãn cần gán
  - Ràng buộc chính: Mỗi nhãn chỉ được gán cho đúng một vị trí, cấu hình phải đạt được từ trạng thái ban đầu
  - Hàm xung đột: Số ô không ở đúng vị trí đích hoặc số cặp vi phạm ràng buộc
  - Thuật toán lặp đi lặp lại việc chọn một vị trí có xung đột và đổi nhãn với vị trí khác để giảm tổng số xung đột
  - Không mô phỏng việc di chuyển ô trống như các thuật toán tìm kiếm đường đi, mà tập trung vào việc tối ưu gán nhãn
- **Ưu điểm**: Rất hiệu quả trên nhiều bài toán CSP thực tế, hội tụ nhanh với các bài toán lớn
- **Nhược điểm**: Có thể bị mắc kẹt ở cực tiểu cục bộ, yêu cầu kiểm tra tính khả thi của cấu hình

### Thuật toán học tăng cường
#### 1. Q-Learning
- **Nguyên lý**: Học từ tương tác với môi trường, xây dựng bảng Q-value để đánh giá các hành động
- **Ưu điểm**: Không cần mô hình môi trường, có khả năng hội tụ
- **Nhược điểm**: Chậm hội tụ với không gian trạng thái lớn

#### 2. Mạng Q Sâu (DQN)
- **Nguyên lý**: Kết hợp Q-Learning với mạng nơ-ron sâu
- **Ưu điểm**: Xử lý được các không gian trạng thái phức tạp
- **Nhược điểm**: Phức tạp để cài đặt và điều chỉnh

#### 3. SARSA và Độ Dốc Chính Sách (Policy Gradient)
- **Nguyên lý**: SARSA cập nhật dựa trên chính sách thực tế; Policy Gradient tối ưu hóa trực tiếp chính sách
- **Ưu điểm**: Phù hợp với các loại môi trường khác nhau
- **Nhược điểm**: Có thể chậm hội tụ hoặc không ổn định

### Thuật toán cho môi trường phức tạp
#### 1. Tìm Kiếm Cây AND-OR
- **Nguyên lý**: Xử lý môi trường không xác định, tìm kiếm kế hoạch mạnh mẽ
- **Ưu điểm**: Xử lý được sự không chắc chắn trong môi trường
- **Nhược điểm**: Phức tạp và tốn nhiều bộ nhớ

#### 2. Tìm Kiếm Trong Môi Trường Quan Sát Một Phần
- **Nguyên lý**: Xử lý trường hợp không thể quan sát toàn bộ trạng thái
- **Ưu điểm**: Mô phỏng thực tế khi thông tin không đầy đủ
- **Nhược điểm**: Khó tìm được giải pháp tối ưu

#### 3. Tìm Kiếm Trạng Thái Niềm Tin (Belief State Search)
- **Nguyên lý**: Duy trì đồng thời nhiều trạng thái niềm tin, tìm giải pháp chung
- **Ưu điểm**: Xử lý hiệu quả sự không chắc chắn về trạng thái
- **Nhược điểm**: Tốn nhiều tài nguyên tính toán

## CÁC HÀM HEURISTIC

### 1. Khoảng Cách Manhattan
- **Mô tả**: Tổng khoảng cách ngang và dọc từ mỗi ô đến vị trí đích của nó
- **Tính chất**: Admissible (không bao giờ ước lượng cao)
- **Ưu điểm**: Hiệu suất tốt cho bài toán 8-puzzle, cân bằng giữa tính đơn giản và tính thông tin

### 2. Ô Sai Vị Trí (Misplaced Tiles)
- **Mô tả**: Đếm số ô không nằm đúng vị trí đích
- **Tính chất**: Admissible, đơn giản nhưng ít thông tin hơn Khoảng cách Manhattan
- **Ưu điểm**: Dễ tính toán

### 3. Xung Đột Tuyến Tính (Linear Conflict)
- **Mô tả**: Mở rộng từ Khoảng cách Manhattan, thêm phạt cho các ô nằm đúng hàng/cột nhưng sai thứ tự
- **Tính chất**: Admissible, cung cấp nhiều thông tin hơn Manhattan
- **Ưu điểm**: Thường dẫn đến giải pháp nhanh hơn do ước lượng gần hơn với chi phí thực

## ĐÁNH GIÁ VÀ SO SÁNH HIỆU SUẤT

### Thuật toán không có thông tin
- **BFS**: Thường nhanh đối với 8-puzzle, đảm bảo giải pháp tối ưu, nhưng sử dụng nhiều bộ nhớ
- **DFS**: Hiệu quả về bộ nhớ nhưng có thể tìm được giải pháp không tối ưu
- **UCS**: Giải pháp tối ưu, nhưng có thể khám phá nhiều nút không cần thiết
- **IDS**: Kết hợp ưu điểm của BFS và DFS, dù có chi phí từ việc khám phá lặp lại

### Thuật toán có thông tin
- **Greedy**: Rất nhanh, nhưng không đảm bảo tối ưu
- **A***: Cân bằng tốt giữa tốc độ và tối ưu
- **IDA***: Tiết kiệm bộ nhớ hơn A* nhưng vẫn tối ưu

### Thuật toán tìm kiếm cục bộ và các loại khác
- **Hill Climbing**: Nhanh nhưng dễ bị mắc kẹt ở cực tiểu cục bộ
- **Simulated Annealing**: Thoát khỏi cực tiểu cục bộ tốt hơn
- **Genetic Algorithm**: Tìm kiếm rộng nhưng có thể chậm hội tụ
- **CSP và RL**: Cung cấp các phương pháp tiếp cận khác, phù hợp cho các biến thể phức tạp hơn của bài toán

## HƯỚNG DẪN SỬ DỤNG

### Yêu cầu hệ thống
- Python 3.6 trở lên
- Tkinter (cho giao diện đồ họa)
- Các thư viện phụ thuộc: [liệt kê các thư viện]


#### Thuật toán không có thông tin:
```
# Chạy một thuật toán không có thông tin cụ thể
python main.py --console --uninformed --bfs
python main.py --console --uninformed --dfs
python main.py --console --uninformed --ucs
python main.py --console --uninformed --ids

# So sánh tất cả các thuật toán không có thông tin
python main.py --console --uninformed --all
```

#### Thuật toán có thông tin:
```
# Chạy một thuật toán có thông tin cụ thể với một heuristic
python main.py --console --informed --astar --manhattan
python main.py --console --informed --greedy --misplaced
python main.py --console --informed --idastar --linear

# So sánh tất cả các thuật toán có thông tin với một heuristic
python main.py --console --informed --all --manhattan
```

*(Các lệnh tương tự cho các loại thuật toán khác)*

### Giao diện đồ họa
Khởi động giao diện đồ họa:
```
python main.py
```
hoặc
```
python main.py --gui
```

#### Tính năng giao diện đồ họa:
- Chọn nhóm thuật toán và thuật toán cụ thể
- Chọn hàm heuristic (với thuật toán có thông tin)
- Trộn (Shuffle) bảng puzzle ngẫu nhiên
- Giải (Solve) trạng thái hiện tại của puzzle
- Trực quan hóa các bước giải
- Xem thống kê hiệu suất

#### Tính năng đặc biệt:
- Mô phỏng môi trường không xác định (với AND-OR Tree)
- Hiển thị một phần thông tin (với Partially Observable Search)
- Hiển thị nhiều trạng thái niềm tin đồng thời (với Belief State Search)

## KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

Đồ án đã triển khai và so sánh một loạt các thuật toán tìm kiếm AI khi áp dụng vào bài toán 8-puzzle. Các kết quả cho thấy:

1. Thuật toán A* với heuristic Linear Conflict thường đạt hiệu suất tốt nhất về tỉ lệ tối ưu/tốc độ
2. Các thuật toán tìm kiếm cục bộ như Simulated Annealing có thể là lựa chọn tốt cho các trạng thái phức tạp
3. Các thuật toán học tăng cường cần thời gian huấn luyện nhưng có thể hoạt động tốt sau khi được huấn luyện

Các hướng phát triển trong tương lai:
- Mở rộng lên các biến thể puzzle lớn hơn (15-puzzle, 24-puzzle)
- Tích hợp thêm các kỹ thuật học máy hiện đại
- Tối ưu hóa hiệu suất cho các không gian trạng thái lớn
- Phát triển các hàm heuristic mới, hiệu quả hơn

## TÀI LIỆU THAM KHẢO

1. Russell, S. J., & Norvig, P. (2020). Artificial Intelligence: A Modern Approach (4th Edition). Pearson.
2. Korf, R. E. (1985). Depth-first iterative-deepening: An optimal admissible tree search. Artificial Intelligence, 27(1), 97-109.
3. Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). A formal basis for the heuristic determination of minimum cost paths. IEEE Transactions on Systems Science and Cybernetics, 4(2), 100-107.
4. [Thêm các tài liệu tham khảo khác] 