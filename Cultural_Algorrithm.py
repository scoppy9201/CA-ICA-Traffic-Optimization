import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
from scipy.spatial.distance import cdist

# --- Tham số mô phỏng ---
SO_DAN_CU = 30                # Số điểm dân cư
SO_TRAM = 5                   # Số trạm trung chuyển
KICH_THUOC_QUAN_THE = 50      # Kích thước quần thể
SO_THE_HE = 100               # Số vòng lặp (thế hệ)
GIOI_HAN_X = (0, 100)         # Biên tọa độ X
GIOI_HAN_Y = (0, 100)         # Biên tọa độ Y

# --- Tạo dữ liệu dân cư ---
np.random.seed(42)
dan_cu = np.random.uniform(0, 100, size=(SO_DAN_CU, 2))      # Vị trí dân cư ngẫu nhiên
mat_do = np.random.randint(1, 10, size=SO_DAN_CU)            # Mật độ dân cư tại mỗi điểm

# --- Hàm đánh giá (fitness): Tổng khoảng cách dân cư đến trạm gần nhất ---
def ham_danh_gia(giai_phap, dan_cu, mat_do):
    khoang_cach = cdist(dan_cu, giai_phap)                   # Ma trận khoảng cách
    kc_min = np.min(khoang_cach, axis=1)                     # Khoảng cách nhỏ nhất đến trạm
    return np.sum(mat_do * kc_min)                           # Tổng khoảng cách có trọng số

# --- Khởi tạo quần thể ---
def khoi_tao_quan_the():
    return [np.random.uniform([GIOI_HAN_X[0], GIOI_HAN_Y[0]], [GIOI_HAN_X[1], GIOI_HAN_Y[1]], size=(SO_TRAM, 2)) for _ in range(KICH_THUOC_QUAN_THE)]

# --- Chọn cá thể tốt nhất ---
def chon_tot_nhat(quan_the, dan_cu, mat_do):
    danh_gia = [ham_danh_gia(ca_the, dan_cu, mat_do) for ca_the in quan_the]
    chi_so_tot = np.argmin(danh_gia)
    return quan_the[chi_so_tot], danh_gia[chi_so_tot]

# --- Tạo cá thể con (đột biến) ---
def dot_bien(ca_the, tot_nhat, pham_vi):
    con = ca_the.copy()
    for i in range(SO_TRAM):
        if np.random.rand() < 0.5:
            con[i] = tot_nhat[i] + np.random.uniform(-1, 1, 2) * pham_vi
    return con

# --- Thuật toán chính (Cultural Algorithm) ---
def thuat_toan_van_hoa():
    quan_the = khoi_tao_quan_the()
    tot_nhat_toan_cuc, diem_tot_nhat = chon_tot_nhat(quan_the, dan_cu, mat_do)
    pham_vi = 10
    lich_su = []
    cac_giai_phap_tot = []

    for the_he in range(SO_THE_HE):
        quan_the_moi = [dot_bien(ca_the, tot_nhat_toan_cuc, pham_vi) for ca_the in quan_the]
        tot_nhat_the_he, diem = chon_tot_nhat(quan_the_moi, dan_cu, mat_do)
        if diem < diem_tot_nhat:
            tot_nhat_toan_cuc = tot_nhat_the_he
            diem_tot_nhat = diem
        lich_su.append(diem_tot_nhat)
        cac_giai_phap_tot.append(tot_nhat_toan_cuc.copy())
        quan_the = quan_the_moi

    return tot_nhat_toan_cuc, lich_su, cac_giai_phap_tot

# --- Chạy thuật toán ---
giai_phap_tot_nhat, lich_su, cac_giai_phap_tot = thuat_toan_van_hoa()

# --- Biểu diễn động quá trình hội tụ ---
def ve_dong_qua_trinh(dan_cu, cac_giai_phap_tot, lich_su):
    fig, ax = plt.subplots(figsize=(8, 6))

    def cap_nhat(frame):
        ax.clear()
        ax.scatter(dan_cu[:, 0], dan_cu[:, 1], c='blue', s=30, label='Dân cư', marker='o')
        ax.scatter(cac_giai_phap_tot[frame][:, 0], cac_giai_phap_tot[frame][:, 1], c='red', marker='X', s=100, label='Trạm')
        for p in dan_cu:
            tram_gan = cac_giai_phap_tot[frame][np.argmin(np.linalg.norm(cac_giai_phap_tot[frame] - p, axis=1))]
            ax.plot([p[0], tram_gan[0]], [p[1], tram_gan[1]], 'gray', linewidth=0.5)
        ax.set_title(f"Tối ưu vị trí trạm - Thế hệ {frame} | Fitness: {lich_su[frame]:.2f}")
        ax.set_xlim(GIOI_HAN_X)
        ax.set_ylim(GIOI_HAN_Y)
        ax.legend()
        ax.grid(True)

    ani = FuncAnimation(fig, cap_nhat, frames=len(cac_giai_phap_tot), interval=200, repeat=False)
    plt.show()

# --- Vẽ biểu đồ phân bố khoảng cách ---
def bieu_do_khoang_cach(dan_cu, tram):
    khoang_cach = cdist(dan_cu, tram)
    kc_min = np.min(khoang_cach, axis=1)
    plt.figure(figsize=(8, 5))
    sns.histplot(kc_min, bins=10, kde=True, color='teal')
    plt.title("Phân bố khoảng cách dân cư đến trạm gần nhất", fontsize=13)
    plt.xlabel("Khoảng cách")
    plt.ylabel("Số dân cư")
    plt.tight_layout()
    plt.show()

# --- Vẽ biểu đồ hội tụ ---
def bieu_do_hoi_tu(lich_su):
    sns.set(style="darkgrid")
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=range(len(lich_su)), y=lich_su, color='orange', linewidth=2.5)
    plt.title("Biểu đồ hội tụ Cultural Algorithm", fontsize=14)
    plt.xlabel("Thế hệ")
    plt.ylabel("Fitness tốt nhất")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- Gọi các hàm vẽ ---
ve_dong_qua_trinh(dan_cu, cac_giai_phap_tot, lich_su)
bieu_do_khoang_cach(dan_cu, giai_phap_tot_nhat)
bieu_do_hoi_tu(lich_su)