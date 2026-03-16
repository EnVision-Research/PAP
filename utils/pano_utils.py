import cv2
import numpy as np
import os
from PIL import Image as _PILImage, ImageDraw as _ImageDraw, ImageFont as _ImageFont

# Module-level caches for precomputed remap maps.
# Keys: (pano_h, pano_w, fov_h, fov_w) / (h, w, grid_index)
_fov_restore_maps = {}
_pano_restore_maps = {}


def _load_font_with_fallback(font_path, font_size):
    """
    Load a TrueType font without relying on the process CWD.
    Priority:
    1) user-provided absolute path
    2) user-provided relative path (as-is, module dir, module/fonts dir)
    3) common bundled/system font candidates
    4) Pillow default bitmap font
    """
    module_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = []

    if font_path:
        if os.path.isabs(font_path):
            candidates.append(font_path)
        else:
            candidates.extend([
                font_path,
                os.path.join(module_dir, font_path),
                os.path.join(module_dir, "fonts", font_path),
            ])

    candidates.extend([
        os.path.join(module_dir, "Comic.ttf"),
        os.path.join(module_dir, "fonts", "Comic.ttf"),
    ])

    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if os.path.exists(candidate):
            try:
                return _ImageFont.truetype(candidate, font_size)
            except Exception:
                continue

    # Try a common name that Pillow may resolve via fontconfig.
    # try:
    #     return _ImageFont.truetype("DejaVuSans.ttf", font_size)
    # except Exception:
    #     return _ImageFont.load_default()


def process_panorama(img, grid_index, sub_grid_index=None, grid_cols=4, grid_rows=3, visualize=True, output_dir="output"):
    """
    旋转全景图使指定格子居中；若提供 sub_grid_index 则进一步使子格居中。
    输入: img (np.ndarray), grid_index (int), sub_grid_index (int|None) 子格索引,
          grid_cols (int), grid_rows (int) 网格行列数。
    输出: clean_img (np.ndarray) 居中后的全景图。
    """
    if img is None: return None
    h, w = img.shape[:2]
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    # Calculate main grid position
    r_idx, c_idx = (grid_index - 1) // grid_cols, (grid_index - 1) % grid_cols
    
    # Dynamic steps based on grid dimensions
    lat_step = np.pi / grid_rows
    lon_step = (2.0 * np.pi) / grid_cols
    
    # Main grid center offsets
    t_lat = ((grid_rows - 1) / 2.0 - r_idx) * lat_step
    t_lon = (c_idx - (grid_cols - 1) / 2.0) * lon_step

    if sub_grid_index is not None:
        sub_r, sub_c = (sub_grid_index - 1) // grid_cols, (sub_grid_index - 1) % grid_cols
        # Offset in equirectangular space (sub-grid assumed to have same dimensions as main grid)
        t_lat += ((grid_rows - 1) / 2.0 - sub_r) * (lat_step / grid_rows)
        t_lon += (sub_c - (grid_cols - 1) / 2.0) * (lon_step / grid_cols)

    R = np.array([[np.cos(t_lon), -np.sin(t_lon), 0], [np.sin(t_lon), np.cos(t_lon), 0], [0, 0, 1]]) @ \
        np.array([[np.cos(t_lat), 0, -np.sin(t_lat)], [0, 1, 0], [np.sin(t_lat), 0, np.cos(t_lat)]])

    u, v = np.meshgrid(np.arange(w), np.arange(h))
    lon_out, lat_out = (u / float(w) - 0.5) * 2 * np.pi, -(v / float(h) - 0.5) * np.pi
    v_out = np.stack((np.cos(lat_out)*np.cos(lon_out), np.cos(lat_out)*np.sin(lon_out), np.sin(lat_out)), axis=-1).reshape(-1, 3)
    v_in = v_out @ R.T
    
    map_x = ((np.arctan2(v_in[:, 1], v_in[:, 0]) / (2 * np.pi) + 0.5) * w).reshape(h, w).astype(np.float32)
    map_y = ((0.5 - np.arcsin(np.clip(v_in[:, 2], -1.0, 1.0)) / np.pi) * h).reshape(h, w).astype(np.float32)

    clean_img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    #cv2.imwrite(os.path.join(output_dir, f"clean_rotated_grid_{grid_index}.jpg"), clean_img)

    if visualize:
        palette = np.array([
            [255,100,100],[100,255,100],[100,100,255],[255,255,100],
            [255,100,255],[100,255,255],[150,50,0],[0,150,50],
            [50,0,150],[200,150,50],[50,200,150],[150,50,200],
            [255,128,0],[0,255,128],[128,0,255],[255,0,128],
            [128,255,0],[0,128,255],[200,100,100],[100,200,100],
            [100,100,200],[200,200,100],[200,100,200],[100,200,200],
            [50,50,50],[150,150,150],[250,250,250],[0,0,0],
            [128,128,128],[64,64,64],[192,192,192],[128,0,0]
        ], dtype=np.uint8)
        gr, gc = (map_y // (h / grid_rows)).clip(0, grid_rows-1).astype(int), (map_x // (w / grid_cols)).clip(0, grid_cols-1).astype(int)
        idx = (gr * grid_cols + gc).flatten()
        vis_mask = palette[idx % len(palette)].reshape(h, w, 3)
        #cv2.imwrite(os.path.join(output_dir, f"visual_rotated_grid_{grid_index}.jpg"), cv2.addWeighted(clean_img, 0.7, vis_mask, 0.3, 0))
    return clean_img

def restore_panorama(img, grid_index, sub_grid_index=None, grid_cols=4, grid_rows=3, output_path=None, output_dir="output"):
    """
    逆过程：从旋转后的图恢复全景图。
    输入: img (np.ndarray), grid_index (int), sub_grid_index (int|None) 若 process 时用了子格则需一致。
          grid_cols (int), grid_rows (int) 网格行列数。
    输出: restored (np.ndarray) 恢复的全景图。
    Remap maps are cached by (h, w, grid_index, sub_grid_index, grid_cols, grid_rows) to avoid recomputing trig across calls.
    """
    if img is None: return None
    h, w = img.shape[:2]

    cache_key = (h, w, grid_index, sub_grid_index, grid_cols, grid_rows)
    if cache_key not in _pano_restore_maps:
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        
        # Calculate main grid position
        r_idx, c_idx = (grid_index - 1) // grid_cols, (grid_index - 1) % grid_cols
        
        # Dynamic steps based on grid dimensions
        lat_step = np.pi / grid_rows
        lon_step = (2.0 * np.pi) / grid_cols
        
        # Main grid center offsets
        t_lat = ((grid_rows - 1) / 2.0 - r_idx) * lat_step
        t_lon = (c_idx - (grid_cols - 1) / 2.0) * lon_step

        if sub_grid_index is not None:
            sub_r, sub_c = (sub_grid_index - 1) // grid_cols, (sub_grid_index - 1) % grid_cols
            # Offset in equirectangular space
            t_lat += ((grid_rows - 1) / 2.0 - sub_r) * (lat_step / grid_rows)
            t_lon += (sub_c - (grid_cols - 1) / 2.0) * (lon_step / grid_cols)

        R = np.array([[np.cos(t_lon), -np.sin(t_lon), 0], [np.sin(t_lon), np.cos(t_lon), 0], [0, 0, 1]]) @ \
            np.array([[np.cos(t_lat), 0, -np.sin(t_lat)], [0, 1, 0], [np.sin(t_lat), 0, np.cos(t_lat)]])

        yy, xx = np.indices((h, w))
        lon, lat = (xx / w - 0.5) * 2 * np.pi, (0.5 - yy / h) * np.pi
        v_in = np.stack((np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)), axis=-1).reshape(-1, 3)
        v_out = v_in @ R

        map_x = ((np.arctan2(v_out[:, 1], v_out[:, 0]) / (2 * np.pi) + 0.5) * w).reshape(h, w).astype(np.float32)
        map_y = ((0.5 - np.arcsin(np.clip(v_out[:, 2], -1.0, 1.0)) / np.pi) * h).reshape(h, w).astype(np.float32)
        _pano_restore_maps[cache_key] = (map_x, map_y)

    map_x, map_y = _pano_restore_maps[cache_key]
    restored = cv2.remap(img, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP)
    #cv2.imwrite(output_path or os.path.join(output_dir, f"restored_grid_{grid_index}.jpg"), restored)
    return restored

def extract_fov(img, output_size=(1200, 1200), fov_deg=120, output_path=None, output_dir="output"):
    """
    裁剪中心 FoV 图像（正方形视野，横竖一致）。
    输入: img (np.ndarray), output_size (tuple) 输出尺寸, fov_deg (float) 视野角度（默认 120°）。
    输出: out (np.ndarray) 裁剪后的 FoV 图像。
    """
    if img is None: return None
    h_src, w_src = img.shape[:2]
    hr = np.radians(fov_deg / 2.0)
    uu, vv = np.meshgrid(np.arange(output_size[0]), np.arange(output_size[1]))
    px, py = (2.0 * uu / (output_size[0]-1) - 1.0) * np.tan(hr), (1.0 - 2.0 * vv / (output_size[1]-1)) * np.tan(hr)
    lon, lat = np.arctan(px), np.arctan(py * np.cos(np.arctan(px)))
    
    map_x, map_y = ((lon / (2 * np.pi) + 0.5) * w_src).astype(np.float32), ((0.5 - lat / np.pi) * h_src).astype(np.float32)
    out = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    #cv2.imwrite(output_path or os.path.join(output_dir or ".", "fov_120x120.jpg"), out)
    return out

def restore_fov_to_panorama(fov_img, pano_w, pano_h, fov_deg=120, background_pano=None, output_path=None, output_dir="output"):
    """
    将 FoV 图贴回全景图。
    输入: fov_img (np.ndarray), pano_w/h (int) 全景尺寸, fov_deg (float) 视野角度（默认 120°）, background_pano (np.ndarray) 可选背景。
    输出: pano (np.ndarray) 合成后的全景图。
    Remap maps are cached by (pano_h, pano_w, fov_h, fov_w, fov_deg) to avoid recomputing trig across calls.
    """
    if fov_img is None: return None
    h_out, w_out = fov_img.shape[:2]

    cache_key = (pano_h, pano_w, h_out, w_out, fov_deg)
    if cache_key not in _fov_restore_maps:
        hr = np.radians(fov_deg / 2.0)
        yy, xx = np.indices((pano_h, pano_w))
        lon, lat = (xx / pano_w - 0.5) * 2 * np.pi, (0.5 - yy / pano_h) * np.pi
        in_fov = (np.abs(lon) <= hr) & (np.abs(lat) <= hr)

        u = (np.tan(lon) / np.tan(hr) * 0.5 + 0.5) * (w_out - 1)
        v = (0.5 - np.tan(lat) / np.cos(lon) / np.tan(hr) * 0.5) * (h_out - 1)
        mx = np.zeros((pano_h, pano_w), dtype=np.float32)
        my = np.zeros((pano_h, pano_w), dtype=np.float32)
        mx[in_fov] = u[in_fov].astype(np.float32)
        my[in_fov] = v[in_fov].astype(np.float32)
        _fov_restore_maps[cache_key] = (in_fov, mx, my)

    in_fov, mx, my = _fov_restore_maps[cache_key]
    pano = background_pano.copy() if background_pano is not None else np.zeros((pano_h, pano_w, 3), dtype=np.uint8)
    fov_region = cv2.remap(fov_img, mx, my, cv2.INTER_NEAREST)
    pano[in_fov] = fov_region[in_fov]
    #cv2.imwrite(output_path or os.path.join(output_dir or ".", "panorama_restored_fov.jpg"), pano)
    return pano

def draw_fov_contour(img_or_path, fov_deg=120, output_path=None, output_dir="output", color=(204,153,102), thickness=4):
    """
    在全景图上绘制 FoV 轮廓线（正方形视野，横竖一致）。
    输入: img_or_path (str/np.ndarray), fov_deg (float) 视野角度（默认 120°）, color (tuple) 颜色, thickness (int) 粗细。
    输出: out (np.ndarray) 带轮廓的全景图。
    """
    img = cv2.imread(img_or_path) if isinstance(img_or_path, str) else img_or_path
    if img is None: return None
    h, w = img.shape[:2]
    out, hr = img.copy(), np.radians(fov_deg / 2.0)
    
    lons = np.linspace(-hr, hr, 128)
    for sign in [1, -1]:
        lats = sign * np.arctan(np.tan(hr) * np.cos(lons))
        pts = np.column_stack([((lons/(2*np.pi)+0.5)*w), ((0.5-lats/np.pi)*h)]).astype(np.int32)
        cv2.polylines(out, [pts], False, color, thickness, lineType=cv2.LINE_AA)
    
    lats_side = np.linspace(np.arctan(-np.tan(hr)*np.cos(hr)), np.arctan(np.tan(hr)*np.cos(hr)), 64)
    for lon_side in [-hr, hr]:
        pts = np.column_stack([((lon_side/(2*np.pi)+0.5)*w)*np.ones_like(lats_side), ((0.5-lats_side/np.pi)*h)]).astype(np.int32)
        cv2.polylines(out, [pts], False, color, thickness, lineType=cv2.LINE_AA)
    
    #cv2.imwrite(output_path or os.path.join(output_dir or ".", "fov_contour.jpg"), out)
    return out

def draw_grid(img, grid_cols=4, grid_rows=3, line_color="white", line_thickness=5,
              text_color="black", font_path="Comic.ttf", font_size=None, stroke_width=None):
    """
    在图像上叠加格子线和编号（用于 VLM Visual Prompt）。
    内部使用 PIL 渲染，保证与原始版本视觉一致。

    输入: img — PIL Image 或 np.ndarray (BGR)
    输出: 与输入相同类型
    """
    if img is None: return None

    # stroke_width = 2
    if stroke_width is None:
        stroke_width = font_size // 25

    is_numpy = isinstance(img, np.ndarray)
    if is_numpy:
        pil_img = _PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        pil_img = img.copy()

    draw = _ImageDraw.Draw(pil_img)
    w, h = pil_img.size
    cell_h, cell_w = h // grid_rows, w // grid_cols

    # 绘制格子边界线
    for i in range(1, grid_cols):
        x = i * cell_w
        draw.line([(x, 0), (x, h)], fill=line_color, width=line_thickness)
    for i in range(1, grid_rows):
        y = i * cell_h
        draw.line([(0, y), (w, y)], fill=line_color, width=line_thickness)

    # 加载字体（与 CWD 解耦）
    fs = font_size if font_size is not None else cell_h // 3
    font = _load_font_with_fallback(font_path, fs)

    # 绘制格子编号（中心对齐，带描边）
    for i in range(grid_rows * grid_cols):
        cx = (i % grid_cols) * cell_w + cell_w // 2
        cy = (i // grid_cols) * cell_h + cell_h // 2
        draw.text((cx, cy), str(i + 1), fill=text_color, font=font,
                  anchor="mm", stroke_width=stroke_width)

    if is_numpy:
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return pil_img

def draw_grid_color(img, grid_cols=4, grid_rows=3, alpha=100, text_color="black", font_path="Comic.ttf", font_size=None, stroke_width=2):
    """
    在图像上叠加半透明的彩色格子和编号（用于 VLM Visual Prompt）。
    内部使用 PIL 渲染。

    输入: img — PIL Image 或 np.ndarray (BGR)
    输出: 与输入相同类型
    """
    if img is None: return None

    is_numpy = isinstance(img, np.ndarray)
    if is_numpy:
        pil_img = _PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        pil_img = img.copy()

    overlay = _PILImage.new('RGBA', pil_img.size, (0, 0, 0, 0))
    draw = _ImageDraw.Draw(overlay)
    
    w, h = pil_img.size
    cell_h, cell_w = h / grid_rows, w / grid_cols

    palette = [
        (255,100,100), (100,255,100), (100,100,255), (255,255,100),
        (255,100,255), (100,255,255), (150,50,0), (0,150,50),
        (50,0,150), (200,150,50), (50,200,150), (150,50,200),
        (255,128,0), (0,255,128), (128,0,255), (255,0,128),
        (128,255,0), (0,128,255), (200,100,100), (100,200,100),
        (100,100,200), (200,200,100), (200,100,200), (100,200,200),
        (50,50,50), (150,150,150), (250,250,250), (0,0,0),
        (128,128,128), (64,64,64), (192,192,192), (128,0,0)
    ]

    fs = font_size if font_size is not None else int(cell_h // 3)
    font = _load_font_with_fallback(font_path, fs)

    for i in range(grid_rows * grid_cols):
        r, c = i // grid_cols, i % grid_cols
        shape = [c * cell_w, r * cell_h, (c + 1) * cell_w, (r + 1) * cell_h]
        color = palette[i % len(palette)] + (alpha,)
        draw.rectangle(shape, fill=color)

        cx = c * cell_w + cell_w / 2
        cy = r * cell_h + cell_h / 2
        draw.text((cx, cy), str(i + 1), fill=text_color, font=font,
                  anchor="mm", stroke_width=stroke_width)

    pil_img = pil_img.convert('RGBA')
    out = _PILImage.alpha_composite(pil_img, overlay).convert('RGB')

    if is_numpy:
        return cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)
    return out


def draw_highlight(img, grid_index, grid_cols=4, grid_rows=3, color=(255, 255, 150, 100)):
    """
    在指定的格子位置绘制半透明底色。
    color: (R, G, B, Alpha)，默认淡黄色半透明。
    """
    from PIL import Image, ImageDraw
    # 转换格式
    is_numpy = isinstance(img, np.ndarray)
    if is_numpy:
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        pil_img = img.copy()
    
    # 创建透明层
    overlay = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    w, h = pil_img.size
    cell_w, cell_h = w / grid_cols, h / grid_rows
    r, c = (grid_index - 1) // grid_cols, (grid_index - 1) % grid_cols
    
    # 计算矩形坐标
    shape = [c * cell_w, r * cell_h, (c + 1) * cell_w, (r + 1) * cell_h]
    draw.rectangle(shape, fill=color)
    
    # 复合图像
    pil_img = pil_img.convert('RGBA')
    out = Image.alpha_composite(pil_img, overlay).convert('RGB')
    
    if is_numpy:
        return cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)
    return out
