# BÁO CÁO KIỂM TOÁN, HIỆU ĐÍNH VÀ XÁC MINH TOÁN TÀI CHÍNH
## Phiên bản: JAX NTBN v1 AuditFix (`experiments/jax_ntbn_v1_auditfix`)

**Thực hiện:** Hội đồng Tác tử Cấp cao Sovereign Epistemic Gate (`@gng5` — Lõi mô hình Gemini 4 Argon)  
**Tập thể tiếp nhận:** @agy1 và Nhóm Kỹ nghệ Lượng hóa  
**Thời gian niêm phong:** 13/11/2026  
**Thư mục lưu trữ:** [`experiments/jax_ntbn_v1_auditfix/`](file:///c:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_v1_auditfix)

---

## 1. VẤN ĐỀ PHÁT HIỆN, VỊ TRÍ TRONG MÃ VÀ NGUYÊN NHÂN

Qua rà soát toán học và truy nguyên luồng dữ liệu thực tế, Hội đồng `@gng5` đã bóc tách được 3 vấn đề kỹ thuật cốt lõi trong phiên bản NTBN v1 sơ khởi:

### Vấn đề 1: Biến dạng hành động do hàm `softclip` làm phá vỡ tính chất dải không giao dịch
* **Vị trí trong mã:** `experiments/jax_ntbn_v1/model_ntbn.py`, dòng 191 (`rollout_ntbn`):
  ```python
  delta_target, a_target = ntbn_action(delta, bl, bu)
  a = softclip(a_target + idelta * first, lb, ub)
  delta = delta + a
  ```
* **Nguyên nhân toán học:**  
  Trong mã gốc Buehler et al., `softclip(a, lb, ub)` là một phép xấp xỉ trơn của hàm kẹp dựa trên `tfp.bijectors.SoftClip`:
  $$r = \frac{a - \text{lb}}{\text{ub} - \text{lb}}, \quad r' = 1 - \frac{\text{softplus}(1 - \text{softplus}(r))}{\text{softplus}(1)}, \quad \text{output} = r' \cdot (\text{ub} - \text{lb}) + \text{lb}$$
  Khi kiểm tra với giới hạn chuẩn $\text{lb} = -5.0, \text{ub} = 5.0$:
  Tại $a = 0.0$ (khi vị thế hiện tại đã nằm trong dải, không cần giao dịch), $r = 0.5$. Công thức trên cho ra:
  $$\text{softclip}(0.0, -5.0, 5.0) = -0.377395 \neq 0.0$$
* **Hậu quả tài chính:**  
  Ngay cả khi nhà đầu tư đã ở vị thế tối ưu trong dải $[b_l, b_u]$, `softclip` vẫn ép bán khống $-0.3774$ cổ phiếu, làm vị thế bị đẩy văng ra ngoài dải và kích hoạt chi phí giao dịch giả mạo tại **mọi bước thời gian**. Điều này vi phạm trắng trợn định nghĩa về "vùng không giao dịch".

### Vấn đề 2: Lệch vị thế bước đầu do tham số `idelta`
* **Vị trí trong mã:** `experiments/jax_ntbn_v1/model_ntbn.py`, dòng 182 & 191:
  ```python
  idelta = p["idelta"][None, :] if a_cfg.init_delta else zeros_i
  a = softclip(a_target + idelta * first, lb, ub)
  ```
* **Nguyên nhân kỹ thuật:**  
  Trong mô hình semi-recurrent gốc của Buehler, mạng nơ-ron học lượng gia hành động $\Delta \delta$, do đó tác giả thêm tham số `idelta` để học vị thế cơ sở ban đầu.  
  Trong NTBN, mạng dự báo trực tiếp dải $[b_l(0), b_u(0)]$. Tại bước $t=0$, vị thế khởi tạo là $\delta_0 = 0$, lệnh giao dịch cần thiết để đưa vị thế vào dải là:
  $$a_0 = \text{clamp}(0, b_l(0), b_u(0)) - 0$$
  Nếu cộng thêm `idelta` ngẫu nhiên: $a_0 = \text{clamp} + \text{idelta}$, vị thế sau giao dịch bước 1 trở thành $\delta_1 = \text{clamp} + \text{idelta}$, hoàn toàn lệch khỏi dải mục tiêu.

### Vấn đề 3: Nguy cơ hiểu lầm đơn vị chi phí do cách diễn đạt báo cáo
* **Vị trí đối chiếu:** `deephedging-jax/deephedging/world.py` (dòng 151) và `model.py` (dòng 193).
* **Bản chất thực tế:**  
  Trong `world.py`: `cost_dS = spot[:, :nSteps] * c.cost_s`. Mảng `market["cost"]` **đã chứa sẵn giá tài sản $S_t$**!  
  Do đó, trong vòng lặp rollout: `cost = cost + sum(|a| * c)` thì biến `c` chính là $S_t \times c_s$. Tích số $|a_t| \times c$ đã là chi phí dollar $|a_t| \times S_t \times c_s$.  
  Nếu trong báo cáo viết lý thuyết $C = c \cdot S \cdot |a|$ mà không giải thích rằng `market["cost"]` đã bao gồm $S_t$, người đọc có thể nhầm tưởng mã thiếu $S_t$ hoặc nhân thêm $S_t$ lần thứ hai dẫn đến sai số bậc hai $S_t^2$.

---

## 2. THAY ĐỔI ĐÃ THỰC HIỆN VÀ LÝ DO

Trong thư mục [`experiments/jax_ntbn_v1_auditfix/`](file:///c:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_v1_auditfix), các thay đổi sau đã được áp dụng:

1. **Chuẩn hóa Chính sách Hành động trong `model_ntbn.py`:**
   - Bổ sung trường `action_clipping: str = "none"` vào `AgentConfigNTBN`.
     - Chế độ `"none"` (mặc định): Triển khai Pure NTBN, $a = \text{clamp}(\delta, b_l, b_u) - \delta$. Tuyệt đối không méo số khi $a=0$.
     - Chế độ `"hard"`: Giới hạn vật lý bằng `jnp.clip(a, lb, ub)`. Giữ nguyên $a$ chính xác khi nằm trong giới hạn thanh khoản, chỉ can thiệp khi lệnh vượt quá giới hạn thị trường.
     - Chế độ `"soft"`: Giữ lại hàm `softclip` chỉ để phục vụ kiểm thử và phân tích đối chứng (ablation).
   - Đặt `init_delta: bool = False` mặc định cho NTBN, đảm bảo bước đầu tiên tuân thủ nghiêm ngặt quy tắc kẹp từ $\delta_0 = 0$ vào $[b_l(0), b_u(0)]$.
2. **Làm rõ Hệ thống Kế toán và Đơn vị Chi phí:**
   - Ghi chú minh bạch trong mã nguồn: `m["cost"]` là chi phí hiệu dụng theo đơn vị tiền tệ trên mỗi cổ phiếu ($S_t \times c_s$).
   - Giữ nguyên công thức `cost = cost + jnp.sum(jnp.abs(a) * c, axis=1)` vì đây là công thức chính xác 100% khớp với định dạng dữ liệu của JAX gốc.
3. **Bộ Kiểm thử Độc lập Chuyên sâu:**
   - Xây dựng 27 test cases bao quát mọi khía cạnh: tính chất dải, giới hạn hành động, đơn vị chi phí theo giá biến thiên theo thời gian, tính tương đương của các hàm tiện ích và tính bất biến của kho gốc.

---

## 3. BẢNG ĐỐI CHIẾU THẬN TRỌNG VỚI BÀI BÁO IMAKI ET AL. (2021)

| Thành phần / Công thức | Bài báo Imaki et al. (2021) | Hiện thực trong AuditFix | Phân loại Thiết kế | Bằng chứng & Kiểm tra Thực nghiệm |
| :--- | :--- | :--- | :--- | :--- |
| **Đặc trưng ngoại sinh $I_t$** | $I_t \in \mathbb{R}^d$ không chứa $\delta_t$ (Eq. 12) | Loại bỏ `"delta"` và `"action"` khỏi input | Hiện thực trực tiếp quy tắc NTBN | Kiểm tra số chiều đầu vào trong `init_params_ntbn` |
| **Dự báo dải $[b_l, b_u]$** | $(b_l, b_u) = \mathcal{N}_\theta(I_t)$ | $b_{\text{center}} \pm \text{softplus}(w)$ | Lựa chọn thiết kế riêng của mã (đảm bảo $b_l \le b_u$) | Bài báo gốc dùng LeakyReLU cộng/trừ vào BS Delta |
| **Hàm kẹp vị thế** | $\text{clamp}(\delta, b_l, b_u)$ (Eq. 13) | `ntbn_clamp(delta, bl, bu)` | Hiện thực trực tiếp quy tắc NTBN | `test_action_policy.py` (Cases 1, 2, 3) |
| **Biên đảo ngược $b_l > b_u$** | Trung điểm $(b_l + b_u)/2$ (Mục 4.1) | `jnp.where(bl <= bu, clamped, 0.5 * (bl + bu))` | Hiện thực trực tiếp đề xuất bài báo | `test_action_policy.py` (Case 7) |
| **Không dùng `idelta`** | $\delta_0 = 0$, $a_0 = \text{clamp}(0, b_l, b_u)$ | `init_delta = False` (mặc định) | Hiện thực trực tiếp quy tắc NTBN | `test_action_policy.py` (Case 4) |
| **Không dùng `softclip`** | $a = \Delta \delta$ trực tiếp | `action_clipping = "none"` | Hiện thực trực tiếp quy tắc NTBN | `test_action_policy.py` (Case 5) |
| **Giả định Chi phí dương $c > 0$** | Bắt buộc để tồn tại dải (Proposition 4.1) | Rà soát 14 configs, chỉ định rõ Cfg 06 & `trainer.ipynb` | Giả định lý thuyết | `configs_ntbn.py` kiểm toán chi tiết |
| **Giả định Tính trơn Payoff** | Payoff phải khả vi bậc 2 (Proposition 4.1) | Cảnh báo không áp dụng lý thuyết cho Barrier/Digital | Giả định lý thuyết | Phân loại riêng trong bảng kiểm toán |
| **Số công cụ phòng hộ** | Chứng minh cho $n_{\text{inst}} = 1$ (Mục 4.2 & Trang 14) | Hỗ trợ dải phân tách độc lập cho $n_{\text{inst}} > 1$ | Mở rộng kỹ thuật (chưa tối ưu đa giác lồi) | `test_action_policy.py` (Case 6) |

---

## 4. KẾT QUẢ TỪNG BÀI KIỂM THỬ (BAO GỒM CẢ CA LỖI ĐÃ SỬA)

Toàn bộ 27 test cases thuộc 6 module trong `experiments/jax_ntbn_v1_auditfix/tests/` đã chạy thành công:

1. **`test_action_policy.py` (7/7 PASS):**
   - Case 1 (Trong dải): $\delta \in [b_l, b_u] \implies a \equiv 0$, vị thế giữ nguyên.
   - Case 2 (Dưới dải): $\delta < b_l \implies a = b_l - \delta > 0$, vị thế nhảy về $b_l$.
   - Case 3 (Trên dải): $\delta > b_u \implies a = b_u - \delta < 0$, vị thế nhảy về $b_u$.
   - Case 4 (Bước đầu & `idelta`): Chứng minh `init_delta=False` đưa vị thế vào dải, trong khi `init_delta=True` làm lệch vị thế ra ngoài dải.
   - Case 5 (Giới hạn hành động): Chứng minh `softclip(0)` sinh ra $-0.3774$ (sai lệch), trong khi `none` và `hard` bảo toàn $a=0$.
   - Case 6 (Nhiều công cụ): Kiểm tra 2 công cụ hoạt động độc lập với dải riêng biệt.
   - Case 7 (Biên đảo ngược): Trả về chính xác trung điểm $(b_l + b_u)/2$.
2. **`test_cost_units_and_invariants.py` (3/3 PASS):**
   - Giá thay đổi theo thời gian: Xác nhận `market["cost"]` biến thiên và khớp $S_t \times c_s$ với độ lệch bằng 0.
   - `market["hedges"]` khớp chính xác $S_T - S_t$.
   - Bất biến kế toán: $\text{gains} \equiv \text{payoff} + \text{pnl} - \text{cost}$ đạt độ chính xác $10^{-5}$ trong đơn vị dollar.
3. **`test_jax_parity_shared_components.py` (3/3 PASS):**
   - 6 hàm hữu dụng khớp mã JAX gốc tới $10^{-7}$.
   - Chuyển đổi dữ liệu `as_batch` khớp 100%.
   - **Ghi nhận lỗi kiểm thử ban đầu & Cách khắc phục:**  
     Trong lần chạy đầu tiên, bài test `test_original_repo_unmodified` gọi `git status` từ thư mục cha và bị lỗi do `c:\Users\Hi Windows 11 Home\Desktop\KET_NOI_USER04` không phải root git, mà bản thân `deephedging-jax/.git` mới là git repository độc lập.  
     Đã hiệu chỉnh lệnh kiểm tra nhắm vào `deephedging-jax/deephedging/` và xác nhận: **Các tệp nguồn lõi của repo gốc không có bất kỳ sửa đổi nào** (`clean`).
4. **`test_ntbn_clamp.py` (6/6 PASS):** Kiểm tra đầy đủ các nhánh kẹp giải tích.
5. **`test_ntbn_gradient.py` (5/5 PASS):** Đạo hàm subgradient giải tích chính xác, không phát sinh `NaN` hay `Inf`.
6. **`test_ntbn_checkpoint.py` (2/2 PASS):** Băm SHA-256 tất định, serialization/deserialization bảo tồn trọng số.
7. **`test_smoke_pipeline.py` (1/1 PASS):** Chạy mượt mà 1 epoch trên Black-Scholes ($c_s = 0.0002$) trong **0.52 giây** trên CPU.

---

## 5. NHỮNG ĐIỂM CHƯA XÁC MINH VÀ MỨC ĐỘ TIN CẬY

* **Mức độ tin cậy về Mã nguồn & Toán học:** **RẤT CAO (Epistemic Level 5).**  
  Mã nguồn đã được kiểm chứng bằng 27 bài kiểm tra độc lập, triệt tiêu mọi biến dạng phi vật lý từ `softclip` và `idelta`, xác minh tường minh đơn vị chi phí.
* **Những điểm lý thuyết còn mở:**
  1. *Hình học vùng không giao dịch đa tài sản:* Với $n_{\text{inst}} \ge 2$, bài báo Imaki et al. thừa nhận đây là bài toán mở phức tạp. Mã hiện tại dùng dải phân tách hộp $[b_l^{(1)}, b_u^{(1)}] \times [b_l^{(2)}, b_u^{(2)}]$, đây là giải pháp gần đúng khả thi nhưng chưa được chứng minh là tối ưu toàn cục.
  2. *Quyền chọn có điều kiện rào cản (Barrier/Digital):* Do payoff không trơn, tính chất khả vi bậc hai của hàm mục tiêu không thỏa mãn. Dải kẹp vẫn hoạt động về mặt cơ học mạng nơ-ron nhưng không còn bảo đảm tính tối ưu Bellman lý thuyết.

---

## 6. KẾT LUẬN & ĐỀ XUẤT BƯỚC TIẾP THEO

### Kết luận:
Mã nguồn `experiments/jax_ntbn_v1_auditfix/` **ĐÃ HOÀN TOÀN ĐỦ ĐIỀU KIỆN KỸ THUẬT VÀ TOÁN HỌC** để bước vào giai đoạn thực nghiệm so sánh nhỏ.

### Đề xuất Benchmark Nhỏ (Tiến hành ở lượt sau khi được phê duyệt):
- **Cấu hình đề xuất:** Mô hình Black-Scholes 1-tài sản cơ sở, quyền chọn Châu Âu ATM Call, có chi phí giao dịch dương thực tế $c_s = 0.0002$ (2 bps) và $c_s = 0.001$ (10 bps), dựa trên cấu hình chuẩn của notebook `trainer.ipynb`.
- **Phương pháp so sánh:**
  - Nhánh 1: Vanilla Deep Hedging JAX gốc (`action-dependent` feedforward).
  - Nhánh 2: JAX NTBN AuditFix (`action-independent` pure NTBN).
- **Quy mô:** 3 seeds, ngân sách 100 - 200 epochs để kiểm chứng tốc độ hội tụ và độ mượt của dải phòng hộ.
- **Cam kết:** Không tự ý chạy benchmark lớn hay nộp job Slurm trong lượt hiện tại.
