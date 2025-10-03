# backend/vision/color_clustering.py
import numpy as np
from skimage.color import rgb2lab, lab2rgb, deltaE_ciede2000
from sklearn.cluster import MiniBatchKMeans

# Base named colors in CIELAB (D65), can be expanded to 30~40 colors (e.g., CSS/XKCD)
# Values are approximate; production code may use a more complete reference table.
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
# Neutral color thresholds based on L* and chroma
NEUTRAL_NAMES = {
    "white": {"L_min": 85},        # L* >= 85 → white if chroma is small
    "black": {"L_max": 30},        # L* <= 30 → black if chroma is small
    "gray":  {"L_min": 30, "L_max": 85},  # Otherwise near-neutral → gray
}

def _to_lab(image_rgb_uint8: np.ndarray) -> np.ndarray:
    """
    Converts a uint8 RGB image to float CIELAB format.

    Args:
        image_rgb_uint8 (np.ndarray): RGB image with uint8 values [0,255].

    Returns:
        np.ndarray: Image in Lab color space with L in [0,100], a/b ~ [-128,127].
    """
    img = image_rgb_uint8.astype(np.float32) / 255.0
    lab = rgb2lab(img)  # skimage 走 sRGB gamma + D65
    return lab

def _flatten_roi(lab: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """
    Flattens the Lab image to a list of pixels, optionally using a mask.

    Args:
        lab (np.ndarray): Lab image of shape (H, W, 3).
        mask (np.ndarray or None): Boolean mask of shape (H, W), or None.

    Returns:
        np.ndarray: Flattened array of Lab pixels selected by the mask.
    """
    h, w, _ = lab.shape
    if mask is not None:
        mask = (mask > 0).reshape(h, w)
        pts = lab[mask]
    else:
        pts = lab.reshape(-1, 3)
    return pts

def _split_neutral_and_color(lab_pts: np.ndarray, chroma_thresh: float = 8.0):
    """
    Splits Lab pixels into neutral and colored points based on chroma.

    Args:
        lab_pts (np.ndarray): Lab pixels of shape (N, 3).
        chroma_thresh (float): Threshold below which pixels are considered neutral.

    Returns:
        tuple: (neutral_pts, color_pts, is_neutral_mask)
    """
    a = lab_pts[:, 1]
    b = lab_pts[:, 2]
    chroma = np.sqrt(a*a + b*b)
    is_neutral = chroma < chroma_thresh
    return lab_pts[is_neutral], lab_pts[~is_neutral], is_neutral

def _name_neutral(L_vals: np.ndarray) -> str:
    """
    Assigns a neutral color name based on the median L* value.

    Args:
        L_vals (np.ndarray): L* values of neutral pixels.

    Returns:
        str: One of 'white', 'black', or 'gray'.
    """
    L = float(np.median(L_vals))
    if L >= NEUTRAL_NAMES["white"]["L_min"]:
        return "white"
    if L <= NEUTRAL_NAMES["black"]["L_max"]:
        return "black"
    return "gray"

def _best_k_by_silhouette(X: np.ndarray, k_min: int = 3, k_max: int = 6) -> int:
    """
    Determines the optimal number of clusters using silhouette score.

    Args:
        X (np.ndarray): Feature array of shape (N, 2) or (N, 3).
        k_min (int): Minimum number of clusters.
        k_max (int): Maximum number of clusters.

    Returns:
        int: Best number of clusters.
    """
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
    """
    Assigns a color name to a Lab color by finding the nearest reference color
    using CIEDE2000 distance.

    Args:
        center_lab (np.ndarray): Lab color array of shape (3,).

    Returns:
        str: Closest named color.
    """
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
    chroma_gamma=1.2,   
    min_color_prop=0.08,  
    named_colors_lab=None, 
):
    """
    Extracts dominant colors from an RGB image (or masked region).

    Args:
        image_rgb_uint8 (np.ndarray): RGB image with uint8 values [0,255].
        mask (np.ndarray or None): Optional mask to select pixels.
        k_range (tuple): Minimum and maximum cluster numbers for colored pixels.
        sample_limit (int): Maximum number of pixels to sample.
        neutral_chroma_thresh (float): Chroma threshold for neutral classification.
        highlight_L_max (float): L* above this is considered highlight.
        shadow_L_min (float): L* below this is considered shadow.
        chroma_gamma (float): Exponent to emphasize high-chroma pixels.
        min_color_prop (float): Minimum proportion of color pixels to report.
        named_colors_lab (dict or None): Optional reference Lab color dictionary.

    Returns:
        list[dict]: Each dict contains keys:
            'name' (str): color name
            'kind' (str): 'color' or 'neutral'
            'lab' (list): Lab value
            'rgb' (list): RGB value [0-255]
            'proportion' (float): proportion of pixels in this color
    """
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

    # Random sampling to limit large images
    if len(pts) > sample_limit:
        idx = np.random.RandomState(42).choice(len(pts), sample_limit, replace=False)
        pts = pts[idx]

    # Classify neutral vs colored pixels
    chroma = np.sqrt(pts[:,1]**2 + pts[:,2]**2)
    is_extreme = (pts[:,0] > highlight_L_max) | (pts[:,0] < shadow_L_min)
    is_neutral = (chroma < neutral_chroma_thresh) | is_extreme

    neutrals = pts[is_neutral]
    colors   = pts[~is_neutral]
    chroma_c = chroma[~is_neutral]

    # Collect neutral colors
    out = []
    if len(neutrals) > 0:
        Lmed = float(np.median(neutrals[:,0]))
        if Lmed >= 85: n="white"
        elif Lmed <= 30: n="black"
        else: n="gray"
        out.append(("neutral", n, np.median(neutrals, axis=0), len(neutrals)))

    # Process colored pixels
    if len(colors) > 0:
        color_prop = len(colors) / max(1, len(pts))
        if color_prop >= min_color_prop:
            ab = colors[:,1:3]
            # Select K for clustering (avoid small sample jitter)
            k_min,k_max = k_range
            k = max(2, k_min) if len(ab) < 2000 else \
                max(k_min, min(k_max, k_min+1))
            km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=4096)
            # Weight high-chroma pixels more
            w = (np.clip(chroma_c, 1e-3, None))**chroma_gamma
            w = (w / w.mean()).clip(0.2, 5.0)
            reps = np.clip(np.round(w).astype(int), 1, 5)
            ab_w = np.repeat(ab, reps, axis=0)
            km.fit(ab_w)
            centers_ab = km.cluster_centers_
            labels = km.predict(ab)
            for cid in range(k):
                members = colors[labels==cid]
                if len(members)==0: continue
                Lc = np.median(members[:,0])
                center_lab = np.array([Lc, centers_ab[cid,0], centers_ab[cid,1]])
                # Assign color name using CIEDE2000 nearest neighbor
                best, best_d = None, 1e9
                for name, ref in (named_colors_lab or NAMED_COLORS_LAB).items():
                    d = float(deltaE_ciede2000(center_lab.reshape(1,1,3), ref.reshape(1,1,3)))
                    if d < best_d: best, best_d = name, d
                out.append(("color", best, center_lab, len(members)))

    # Assemble final results
    total = sum(n for *_, n in out) or 1
    out = sorted(out, key=lambda x: -x[3])
    res = []
    for kind, name, lab_c, cnt in out:
        rgb = (np.clip(lab2rgb(lab_c.reshape(1,1,3)),0,1)[0,0]*255).round().astype(np.uint8).tolist()
        res.append({"name":name,"kind":kind,"lab":lab_c.round(2).tolist(),"rgb":rgb,"proportion":round(cnt/total,4)})
    return res
