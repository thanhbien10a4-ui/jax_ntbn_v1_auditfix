# JAX NTBN V1 AUDITFIX: RÀ SOÁT, SỬA ĐỔI VÀ KIỂM ĐỊNH TOÁN TÀI CHÍNH

Thư mục này chứa phiên bản sửa đổi kỹ thuật (`jax_ntbn_v1_auditfix`) được tách biệt hoàn toàn với `experiments/jax_ntbn_v1/` và kho gốc `deephedging-jax/`.

---

## 1. Mục Đích & Nội Dung Sửa Đổi Kỹ Thuật

Qua quá trình rà soát độc lập, Hội đồng `@gng5` đã phát hiện và xử lý triệt để 2 vấn đề lớn làm sai lệch chính sách NTBN trong phiên bản sơ khởi:

1. **Khắc phục biến dạng hành động từ `softclip`:**
   - Trong repo gốc Buehler et al., `softclip(a, lb, ub)` là hàm bão hòa trơn của TensorFlow Probability.
   - Khi hành động mục tiêu $a = 0$ (vị thế đã nằm trong dải, không cần giao dịch), `softclip(0, -5, 5)` cho ra giá trị **$-0.3774$** (bán khống 0.3774 cổ phiếu), làm vị thế văng ra khỏi dải và phát sinh chi phí giả tại mọi bước.
   - **Giải pháp:** Cung cấp cấu hình `action_clipping`:
     - `"none"` (mặc định cho Pure NTBN): Hành động $a = \text{clamp}(\delta, b_l, b_u) - \delta$ tuyệt đối không bị méo. Khi $\delta \in [b_l, b_u]$, $a \equiv 0$ chính xác.
     - `"hard"`: Giới hạn vật lý $\text{clip}(a, lb, ub)$, không gây méo khi hành động nằm trong ngưỡng giao dịch tối đa.
     - `"soft"`: Giữ lại duy nhất cho mục đích chẩn đoán/ablation đối chứng.

2. **Khắc phục độ lệch vị thế ban đầu từ `idelta`:**
   - Trong repo gốc, `idelta` là tham số học khởi tạo hành động bước đầu của Buehler.
   - Nếu áp dụng vào NTBN: $a_0 = (\text{clamp}(0, b_l, b_u) - 0) + \text{idelta}$, làm vị thế bước đầu $\delta_1 = \text{clamp} + \text{idelta}$ bị đẩy ra ngoài dải $[b_l, b_u]$.
   - **Giải pháp:** Đặt `init_delta: bool = False` mặc định trong `AgentConfigNTBN`. Nhà đầu tư bắt đầu với vị thế $\delta_0 = 0$, lệnh giao dịch đầu tiên đưa vị thế thẳng vào dải $[b_l(0), b_u(0)]$.

3. **Xác minh công thức và đơn vị chi phí:**
   - Xác nhận `market["cost"]` trong `world.py` đã chứa thành phần giá: $\text{cost}_{t} = S_t \times c_s$.
   - Công thức trong rollout: `cost = cost + sum(|a| * m["cost"])` là **đúng chuẩn dollar**: $\sum |a_t| \times S_t \times c_s$.
   - Tuyệt đối không nhân thêm giá $S_t$ một lần nữa (tránh lỗi $S_t^2$).

---

## 2. Bảng Đối Chiếu Thận Trọng Với Bài Báo Imaki et al. (2021)

| Thành phần | Công thức Bài báo (Imaki 2021) | Hiện thực trong AuditFix | Phân loại | Giả định Lý thuyết |
| :--- | :--- | :--- | :--- | :--- |
| **Đầu vào mạng** | $I_t$ ngoại sinh (giá, $T-t$, $\sigma$) | `price`, `time_left`, `payoff_features` (loại bỏ `delta`, `action`) | Hiện thực trực tiếp | Không phụ thuộc hành động quá khứ |
| **Dự báo dải** | $(b_l(t), b_u(t)) = \mathcal{N}_\theta(I_t)$ | `b_center - w, b_center + w` với $w = \text{softplus}(\dots)$ | Lựa chọn thiết kế mã (đảm bảo $b_l \le b_u$) | Bài báo dùng LeakyReLU cộng/trừ vào BS Delta |
| **Hàm kẹp vị thế** | $\delta_{t+1} = \text{clamp}(\delta_t, b_l, b_u)$ (Eq. 13) | `ntbn_clamp(delta, bl, bu)` | Hiện thực trực tiếp | Cần chi phí $c > 0$ để tối ưu |
| **Biên đảo ngược** | Trung điểm $(b_l + b_u)/2$ khi $b_l > b_u$ (Sec 4.1) | `jnp.where(bl <= bu, clamped, 0.5 * (bl + bu))` | Hiện thực trực tiếp | Tránh mất ổn định số |
| **Hành động giao dịch** | $a_t = \delta_{t+1} - \delta_t$ | `a = delta_target - delta_prev` | Hiện thực trực tiếp | Không dùng `idelta` hay `softclip` |
| **Phí giao dịch** | $C_t = c \cdot S_t \cdot \|a_t\|$ | `sum(|a| * market["cost"])` | Khớp 100% repo JAX | Đơn vị dollar nhất quán với PnL |

---

## 3. Cấu Trúc Tệp Tin & Mã Băm SHA-256

```
3b94fae5a6a43ebbda92b4c3555a34ddc398b131c002e44d85889af37844f1a2  __init__.py
140dfe7dc3f6f9902d4b20d1067f8a477b0e325d8ef655dedda7ce42e62dbb0c  configs_ntbn.py
f3d812ccb5a552c74db0e15e91299fe1ccd0d7a954b0fe02d58d5ea0e7988346  model_ntbn.py
426f4c3bef8a5694c5ebf9513abe7bb74256acb2b44535bc28bff4649258a17a  ntbn_core.py
ee0e5b5b682bdf9e26e2d3bfe0b045dd4e8793064749febf04ae8fafa4dea472  train_ntbn.py
791a9b084d0e28cf5dc0b1ce9fb0dc44d39c9bd8e5563ac17f72a6843b6ba751  tests/test_action_policy.py
4b5857eff74872cbb70a7ef278cc1906c17d4f086ed4f1d9b938e4655c530d08  tests/test_cost_units_and_invariants.py
0215a4ad4f691750c07c68f6537b3ec3bccbbe4ec4c9704172cd5914828e5173  tests/test_jax_parity_shared_components.py
a994df1866180aea01d03db273212e67f1d3bd3de555d854bdd6195ada928601  tests/test_ntbn_checkpoint.py
1d6688880fe5b73d9975e3bd1ab6ebc7773c4df4326b50eff0877e9af945c491  tests/test_ntbn_clamp.py
3805f43ccf219fcadd658849e11ef5c1616ec898ee8bab314a06fcd3745e462b  tests/test_ntbn_gradient.py
2e3ebe0174aa12afb533f469e1ea54d97a029429acb234a159c51ac1c9e175ca  tests/test_smoke_pipeline.py
```

---

## 4. Hướng Dẫn Chạy Toàn Bộ Kiểm Thử

```bash
python -m unittest discover -s experiments/jax_ntbn_v1_auditfix/tests -p "test_*.py" -v
```

Kết quả: **27/27 bài kiểm tra đạt tuyệt đối (OK)** trong ~6.15 giây.
