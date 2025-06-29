import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
from scipy.spatial.distance import cdist
import matplotlib.patches as patches
from matplotlib.ticker import MaxNLocator
# Tham số mô phỏng thuật toán 
So_dan_cu = 30                # Số điểm dân cư
So_tram = 5                   # Số trạm trung chuyển
Kich_thuoc_quan_the = 50      # Kích thước quần thể
So_the_he = 100               # Số vòng lặp (thế hệ)
Bien_do_x = (0, 100)         # Biên tọa độ X
Bien_do_y = (0, 100)         # Biên tọa độ Y
Pham_vi_Dot_bien = 10          # Phạm vi đột biến

# Sinh dữ liệu dân cư ngẫu nhiên 
np.random.seed(42)
dan_cu = np.random.uniform(0, 100, size=(So_dan_cu, 2))      # Vị trí dân cư ngẫu nhiên
mat_do = np.random.randint(1, 10, size=So_dan_cu)            # Mật độ dân cư tại mỗi điểm

# Hàm đánh giá (fitness) - tổng khoảng cách dân cư đến trạm gần nhất
def ham_danh_gia(giai_phap, dan_cu, mat_do):
    khoang_cach = cdist(dan_cu, giai_phap)                   # Ma trận khoảng cách
    kc_min = np.min(khoang_cach, axis=1)                     # Khoảng cách nhỏ nhất đến trạm
    return np.sum(mat_do * kc_min)                           # Tổng khoảng cách có trọng số

# Khởi tạo quân thể (1) 
def khoi_tao_quan_the():
    return [np.random.uniform([Bien_do_x[0], Bien_do_y[0]], 
                              [Bien_do_x[1], Bien_do_y[1]], 
                              size=(So_tram, 2)) for _ in range(Kich_thuoc_quan_the)]

# Chọn cá thể tốt nhất (3)
def chon_tot_nhat(quan_the, dan_cu, mat_do):
    danh_gia = [ham_danh_gia(ca_the, dan_cu, mat_do) for ca_the in quan_the]
    chi_so_tot = np.argmin(danh_gia)
    return quan_the[chi_so_tot], danh_gia[chi_so_tot]

# Đột biến có hướng (belief space tác động lên thế hệ mới)
def dot_bien(ca_the, tot_nhat, pham_vi):
    con = ca_the.copy()
    for i in range(So_tram):
        if np.random.rand() < 0.5:
            con[i] = tot_nhat[i] + np.random.uniform(-1, 1, 2) * pham_vi
    return con

# Thực thi thuật toán (Cultural Algorithm) 
def thuat_toan_van_hoa():
    quan_the = khoi_tao_quan_the() #(1)
    tot_nhat_toan_cuc, diem_tot_nhat = chon_tot_nhat(quan_the, dan_cu, mat_do) #(3)
    lich_su = [] #luu trữ lịch sử fitness
    cac_giai_phap_tot = [] # lưu trữ các giải pháp tốt nhất qua các thế hệ

    for the_he in range(So_the_he):
        # B4: sinh thế hệ mới với đột biến có hướng 
        quan_the_moi = [dot_bien(ca_the, tot_nhat_toan_cuc, Pham_vi_Dot_bien) for ca_the in quan_the]
        #b2+3: đánh giá và cập nhật 
        tot_nhat_the_he, diem = chon_tot_nhat(quan_the_moi, dan_cu, mat_do)
        if diem < diem_tot_nhat:
            tot_nhat_toan_cuc = tot_nhat_the_he
            diem_tot_nhat = diem
        lich_su.append(diem_tot_nhat)
        cac_giai_phap_tot.append(tot_nhat_toan_cuc.copy())
        quan_the = quan_the_moi

    return tot_nhat_toan_cuc, lich_su, cac_giai_phap_tot

# Gọi lại thuật toán để thực thi 
giai_phap_tot_nhat, lich_su, cac_giai_phap_tot = thuat_toan_van_hoa()

# Biểu diễn động quá trình hội tụ 
def ve_dong_qua_trinh(dan_cu, cac_giai_phap_tot, lich_su):
    fig, (ax_map, ax_fitness) = plt.subplots(2, 1, figsize=(9, 7),
                                             gridspec_kw={'height_ratios': [2, 1]})
    def cap_nhat(frame):
        ax_map.clear()
        ax_fitness.clear()
        tram_hien_tai = cac_giai_phap_tot[frame]

        for x in range(0, 100, 10):
            ax_map.axvline(x, color="#f0f0f0", linewidth=0.6)
        for y in range(0, 100, 10):
            ax_map.axhline(y, color="#f0f0f0", linewidth=0.6)
        ax_map.set_facecolor("#fcfcfc")

        for idx, (x, y) in enumerate(dan_cu):
            ax_map.scatter(x, y, c='white', s=80, zorder=2, edgecolors='gray', linewidth=1.2) 
            ax_map.scatter(x, y, c='royalblue', s=55, zorder=3, edgecolors='white', linewidth=0.7) 
            ax_map.text(x + 1.0, y + 1.0, f"{idx+1}", fontsize=7, color='navy', zorder=4)

        for idx, (x, y) in enumerate(tram_hien_tai):
            ax_map.scatter(x, y, marker='s', s=200, c='black', alpha=0.15, zorder=2) 
            ax_map.scatter(x, y, marker='s', s=150, c='crimson', edgecolors='black',
                           linewidths=1.5, zorder=5)  
            ax_map.text(x, y + 3.5, f"Trạm {idx+1}", fontsize=8.5,
                        ha='center', color='black', fontweight='bold', zorder=6)

        for p in dan_cu:
            tram_gan = tram_hien_tai[np.argmin(np.linalg.norm(tram_hien_tai - p, axis=1))]
            kc = np.linalg.norm(tram_gan - p)
            ax_map.plot([p[0], tram_gan[0]], [p[1], tram_gan[1]],
                        'gray', linewidth=0.4, alpha=0.7, zorder=1)
            mx, my = (p[0] + tram_gan[0]) / 2, (p[1] + tram_gan[1]) / 2
            ax_map.text(mx, my, f"{kc:.1f}", fontsize=6,
                        color='forestgreen', ha='center', zorder=3)

        ax_map.set_xlim(Bien_do_x)
        ax_map.set_ylim(Bien_do_y)
        ax_map.set_title(f" Vị trí trạm – Thế hệ {frame} |  Fitness: {lich_su[frame]:.2f}",
                         fontsize=11, fontweight='bold')
        ax_map.axis('off')

        # Biểu đồ fitness
        ax_fitness.plot(range(frame + 1), lich_su[:frame + 1], color='darkorange', linewidth=2.0)
        ax_fitness.set_title(" Biểu đồ Fitness theo thời gian", fontsize=10)
        ax_fitness.set_xlabel("Thế hệ")
        ax_fitness.set_ylabel("Fitness")
        ax_fitness.grid(True, linestyle='--', alpha=0.5)

    ani = FuncAnimation(fig, cap_nhat, frames=len(cac_giai_phap_tot), interval=300, repeat=False)
    plt.tight_layout()
    plt.show()

# Biểu đồ phân bố khoảng cách
def bieu_do_khoang_cach(dan_cu, tram):

    # Tính khoảng cách gần nhất từ mỗi dân cư đến trạm
    khoang_cach = cdist(dan_cu, tram)
    kc_min = np.min(khoang_cach, axis=1)

    kc_tb = np.mean(kc_min)
    kc_min_val = np.min(kc_min)
    kc_max_val = np.max(kc_min)

    plt.figure(figsize=(9, 5.5))
    sns.set(style="whitegrid")
    ax = sns.histplot(kc_min, bins=12, kde=True, color="#3399cc", edgecolor="white", linewidth=1.2)

    # đường tb
    plt.axvline(kc_tb, color='red', linestyle='--', linewidth=1.5, label=f'TB: {kc_tb:.2f}')
    plt.axvline(kc_min_val, color='green', linestyle=':', linewidth=1.2, label=f'Min: {kc_min_val:.2f}')
    plt.axvline(kc_max_val, color='purple', linestyle=':', linewidth=1.2, label=f'Max: {kc_max_val:.2f}')
    plt.title(" Phân bố khoảng cách dân cư đến trạm gần nhất", fontsize=14, fontweight='bold')
    plt.xlabel("Khoảng cách", fontsize=12)
    plt.ylabel("Số lượng dân cư", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))  
    plt.legend(loc='upper right', fontsize=9, frameon=True, facecolor='white', edgecolor='gray')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# Biêu đồ hội tụ của thuật toán CA
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

# Gọi các hàm để thực thi 
ve_dong_qua_trinh(dan_cu, cac_giai_phap_tot, lich_su)
bieu_do_khoang_cach(dan_cu, giai_phap_tot_nhat)
bieu_do_hoi_tu(lich_su)
