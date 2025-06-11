import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation

# --- Thông số bài toán ---
so_tuyen = 10
tong_so_xe = 100
so_de_quoc = 10
so_thuoc_dia = 40
so_vong_lap = 200
np.random.seed(42)

# --- Dữ liệu nhu cầu và vị trí dân cư ---
nhu_cau = np.random.uniform(0.5, 1.5, so_tuyen)
dan_so = np.random.rand(50, 2) * 100  # vị trí dân cư

# --- Hàm đánh giá ---
def ham_danh_gia(phan_bo, alpha=0.7, beta=0.3):
    nhu_cau_moi_xe = nhu_cau / (phan_bo + 1e-6)
    qua_tai = np.maximum(0, nhu_cau - phan_bo)
    tong_thoi_gian_cho = np.sum(nhu_cau_moi_xe)
    tong_qua_tai = np.sum(qua_tai)
    return alpha * tong_thoi_gian_cho + beta * tong_qua_tai

# --- Tạo quốc gia ---
def tao_quoc_gia():
    return np.random.multinomial(tong_so_xe, np.ones(so_tuyen) / so_tuyen)

# --- Di chuyển thuộc địa ---
def di_chuyen(empire, colony):
    hieu = empire - colony
    buoc = np.round(hieu * np.random.uniform(0.1, 0.3)).astype(int)
    thuoc_dia_moi = colony + buoc
    du_thua = thuoc_dia_moi.sum() - tong_so_xe
    if du_thua != 0:
        chi_so = np.random.choice(so_tuyen, abs(du_thua), replace=True)
        for i in chi_so:
            thuoc_dia_moi[i] -= np.sign(du_thua)
    return np.clip(thuoc_dia_moi, 0, tong_so_xe)

# --- ICA với lưu lịch sử ---
de_quoc = [tao_quoc_gia() for _ in range(so_de_quoc)]
thuoc_dia = [tao_quoc_gia() for _ in range(so_thuoc_dia)]
lich_su_phan_bo = []
gia_tri_hoi_tu = []

for vong in range(so_vong_lap):
    tat_ca_quoc_gia = de_quoc + thuoc_dia
    gia_tri_fitness = np.array([ham_danh_gia(qg) for qg in tat_ca_quoc_gia])
    chi_so_sap_xep = np.argsort(gia_tri_fitness)
    tat_ca_quoc_gia = [tat_ca_quoc_gia[i] for i in chi_so_sap_xep]
    de_quoc = tat_ca_quoc_gia[:so_de_quoc]
    thuoc_dia = tat_ca_quoc_gia[so_de_quoc:]

    for i in range(len(thuoc_dia)):
        dq = de_quoc[i % so_de_quoc]
        thuoc_dia[i] = di_chuyen(dq, thuoc_dia[i])

    phan_bo = de_quoc[0].copy()
    vi_tri_tuyen = np.random.rand(so_tuyen, 2) * 100
    lich_su_phan_bo.append((phan_bo, vi_tri_tuyen))
    gia_tri_hoi_tu.append(ham_danh_gia(phan_bo))

# --- Biểu đồ động quá trình phân bổ ---
fig, ax = plt.subplots(figsize=(10, 8))

def cap_nhat(frame):
    ax.clear()
    phan_bo, vi_tri_tuyen = lich_su_phan_bo[frame]

    # Vẽ dân cư
    ax.scatter(dan_so[:, 0], dan_so[:, 1], c='green', s=60, marker='^', label='Dân cư')

    # Vẽ tuyến
    for i in range(so_tuyen):
        ax.scatter(vi_tri_tuyen[i, 0], vi_tri_tuyen[i, 1], 
                   c='blue', s=50 + phan_bo[i] * 5, marker='s', label='Xe buýt' if i == 0 else "")
        ax.text(vi_tri_tuyen[i, 0]+0.5, vi_tri_tuyen[i, 1]+0.5, f"{phan_bo[i]}", fontsize=8)

    # Vẽ kết nối: Mỗi dân cư nối đến 1 tuyến gần nhất
    for ds in dan_so:
        dists = np.linalg.norm(vi_tri_tuyen - ds, axis=1)
        gan_nhat = np.argmin(dists)
        tuyen_pos = vi_tri_tuyen[gan_nhat]
        ax.plot([ds[0], tuyen_pos[0]], [ds[1], tuyen_pos[1]], color='gray', alpha=0.6)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_title(f"Phân bổ xe buýt - Thế hệ {frame + 1}")
    ax.legend(loc="upper right")
    ax.grid(True)


ani = FuncAnimation(fig, cap_nhat, frames=len(lich_su_phan_bo), interval=500, repeat=False)
plt.show()

# --- Biểu đồ phân bổ tốt nhất ---
phan_bo_tot_nhat, _ = lich_su_phan_bo[-1]
plt.figure(figsize=(10, 5))
plt.bar(range(1, so_tuyen + 1), phan_bo_tot_nhat)
plt.xlabel("Tuyến xe buýt")
plt.ylabel("Số xe được phân bổ")
plt.title("Phân bổ số xe tối ưu theo ICA (thế hệ cuối)")
plt.grid(True)
plt.show()

# --- Heatmap so sánh nhu cầu và phân bổ ---
plt.figure(figsize=(8, 5))
du_lieu_heatmap = np.vstack([nhu_cau, phan_bo_tot_nhat / tong_so_xe]).T
sns.heatmap(du_lieu_heatmap, annot=True, cmap="YlGnBu", xticklabels=["Nhu cầu", "Tỷ lệ xe"])
plt.title("So sánh nhu cầu và tỷ lệ xe phân bổ")
plt.yticks(np.arange(0.5, so_tuyen + 0.5), [f"Tuyến {i+1}" for i in range(so_tuyen)], rotation=0)
plt.show()

# --- Biểu đồ hội tụ ---
plt.figure(figsize=(8, 4))
plt.plot(gia_tri_hoi_tu, marker='o')
plt.title("Biểu đồ hội tụ của ICA")
plt.xlabel("Thế hệ")
plt.ylabel("Giá trị hàm đánh giá (fitness)")
plt.grid(True)
plt.show()
