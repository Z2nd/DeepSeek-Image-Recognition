# backend/vision/color_clustering.py
import numpy as np
from skimage.color import rgb2lab, lab2rgb, deltaE_ciede2000
from sklearn.cluster import MiniBatchKMeans

# 你可以把这个色表扩展到 30~40 个（如 CSS/XKCD），这里只放一个干净的基础表
# 值为 CIELAB（D65）。为了简洁，下面用大致值；生产中建议用更完整表。
NAMED_COLORS_LAB = {
    "red":      np.array([53.2, 80.1, 67.2]),
    "orange":   np.array([70.0, 48.0, 69.0]),
    "yellow":   np.array([97.6, -15.8, 93.4]),
    "green":    np.array([87.7, -79.2,  80.9]),
    "cyan":     np.array([91.1, -48.1, -14.1]),
    "blue":     np.array([32.3, 79.2, -107.9]),
    "purple":   np.array([60.3, 84.0, -60.0]),
    "pink":     np.array([75.0, 59.0, -1.0]),
    "brown":    np.array([37.0, 20.0, 20.0]),
}

NEUTRAL_NAMES = {
    "white": {"L_min": 85},        # L* >= 85 且 C*ab 小 → white
    "black": {"L_max": 30},        # L* <= 30 且 C*ab 小 → black
    "gray":  {"L_min": 30, "L_max": 85},  # 其余近中性 → gray
}

def _to_lab(image_rgb_uint8: np.ndarray) -> np.ndarray:
    """uint8 sRGB -> float Lab (L in [0,100], a/b ~ [-128, 127])"""
    img = image_rgb_uint8.astype(np.float32) / 255.0
    lab = rgb2lab(img)  # skimage 走 sRGB gamma + D65
    return lab

def _flatten_roi(lab: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    h, w, _ = lab.shape
    if mask is not None:
        mask = (mask > 0).reshape(h, w)
        pts = lab[mask]
    else:
        pts = lab.reshape(-1, 3)
    return pts

def _split_neutral_and_color(lab_pts: np.ndarray, chroma_thresh: float = 8.0):
    a = lab_pts[:, 1]
    b = lab_pts[:, 2]
    chroma = np.sqrt(a*a + b*b)
    is_neutral = chroma < chroma_thresh
    return lab_pts[is_neutral], lab_pts[~is_neutral], is_neutral

def _name_neutral(L_vals: np.ndarray) -> str:
    L = float(np.median(L_vals))
    if L >= NEUTRAL_NAMES["white"]["L_min"]:
        return "white"
    if L <= NEUTRAL_NAMES["black"]["L_max"]:
        return "black"
    return "gray"

def _best_k_by_silhouette(X: np.ndarray, k_min: int = 3, k_max: int = 6) -> int:
    # 极小样本/单色避免 silhouette 失败
    n = len(X)
    if n < 2000:
        return max(2, min(k_max, k_min))
    from sklearn.metrics import silhouette_score
    best_k, best_s = k_min, -1
    for k in range(k_min, k_max + 1):
        try:
            km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=4096)
            labels = km.fit_predict(X)
            s = silhouette_score(X, labels, metric="euclidean")
            if s > best_s:
                best_s, best_k = s, k
        except Exception:
            continue
    return best_k

def _name_by_ciede2000(center_lab: np.ndarray) -> str:
    # 用 CIEDE2000 到基础色表做最近邻命名
    center = center_lab.reshape(1, 1, 3)
    best_name, best_d = None, 1e9
    for name, ref in NAMED_COLORS_LAB.items():
        d = float(deltaE_ciede2000(center, ref.reshape(1,1,3)))
        if d < best_d:
            best_d, best_name = d, name
    return best_name

def dominant_colors(
    image_rgb_uint8,
    mask=None,
    k_range=(3,6),
    sample_limit=30000,
    neutral_chroma_thresh=8.0,
    highlight_L_max=98.0,
    shadow_L_min=5.0,
    chroma_gamma=1.2,       # >1 提升高饱和的权重
    min_color_prop=0.08,    # 彩色像素至少占 8% 才报彩色主色
    named_colors_lab=None,  # 同前文的 NAMED_COLORS_LAB
):
    img = image_rgb_uint8.astype(np.float32) / 255.0
    lab = rgb2lab(img)

    if mask is not None:
        m = (mask > 0).reshape(lab.shape[:2])
    else:
        m = np.ones(lab.shape[:2], bool)

    L = lab[...,0][m]
    a = lab[...,1][m]
    b = lab[...,2][m]
    pts = np.stack([L,a,b], axis=1)

    # 采样
    if len(pts) > sample_limit:
        idx = np.random.RandomState(42).choice(len(pts), sample_limit, replace=False)
        pts = pts[idx]

    # 近中性判定
    chroma = np.sqrt(pts[:,1]**2 + pts[:,2]**2)
    is_extreme = (pts[:,0] > highlight_L_max) | (pts[:,0] < shadow_L_min)
    is_neutral = (chroma < neutral_chroma_thresh) | is_extreme

    neutrals = pts[is_neutral]
    colors   = pts[~is_neutral]
    chroma_c = chroma[~is_neutral]

    # —— 中性汇总
    out = []
    if len(neutrals) > 0:
        Lmed = float(np.median(neutrals[:,0]))
        if Lmed >= 85: n="white"
        elif Lmed <= 30: n="black"
        else: n="gray"
        out.append(("neutral", n, np.median(neutrals, axis=0), len(neutrals)))

    # —— 彩色只在 a*b* 平面聚类，并按色度加权
    if len(colors) > 0:
        # 彩色像素比例判定
        color_prop = len(colors) / max(1, len(pts))
        if color_prop >= min_color_prop:
            ab = colors[:,1:3]  # 只用 a*, b*
            # 选择 K（避免超小样本抖动）
            k_min,k_max = k_range
            k = max(2, k_min) if len(ab) < 2000 else \
                max(k_min, min(k_max, k_min+1))
            km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=4096)
            # 简易加权：重复采样近似权重（足够好用）
            w = (np.clip(chroma_c, 1e-3, None))**chroma_gamma
            w = (w / w.mean()).clip(0.2, 5.0)  # 防爆
            reps = np.clip(np.round(w).astype(int), 1, 5)
            ab_w = np.repeat(ab, reps, axis=0)
            km.fit(ab_w)
            centers_ab = km.cluster_centers_
            # 恢复 L*：用每簇成员的中位 L*
            labels = km.predict(ab)
            for cid in range(k):
                members = colors[labels==cid]
                if len(members)==0: continue
                Lc = np.median(members[:,0])
                center_lab = np.array([Lc, centers_ab[cid,0], centers_ab[cid,1]])
                # 命名（CIEDE2000 最近邻）
                best, best_d = None, 1e9
                for name, ref in (named_colors_lab or NAMED_COLORS_LAB).items():
                    d = float(deltaE_ciede2000(center_lab.reshape(1,1,3), ref.reshape(1,1,3)))
                    if d < best_d: best, best_d = name, d
                out.append(("color", best, center_lab, len(members)))

    # —— 组装
    total = sum(n for *_, n in out) or 1
    out = sorted(out, key=lambda x: -x[3])
    res = []
    for kind, name, lab_c, cnt in out:
        rgb = (np.clip(lab2rgb(lab_c.reshape(1,1,3)),0,1)[0,0]*255).round().astype(np.uint8).tolist()
        res.append({"name":name,"kind":kind,"lab":lab_c.round(2).tolist(),"rgb":rgb,"proportion":round(cnt/total,4)})
    return res
