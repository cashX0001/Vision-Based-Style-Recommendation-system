from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from PIL import Image
import io
import numpy as np
from sklearn.cluster import KMeans
import cv2
import torch
from torchvision import transforms
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights,
)

app = FastAPI(title="Vision Fashion Helper")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalysisResponse(BaseModel):
    clothing_detected: bool
    dominant_colors: List[str]
    top_colors: List[str]
    bottom_colors: List[str]
    pattern_overall: str
    style_summary: str
    style_tips: List[str]
    note: str


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_deeplab_model = None


def get_deeplab_model():
    global _deeplab_model
    if _deeplab_model is None:
        weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
        _deeplab_model = deeplabv3_resnet50(weights=weights).to(DEVICE)
        _deeplab_model.eval()
    return _deeplab_model


@app.get("/api/health")
def health():
    return {"status": "ok"}


def pil_to_cv2(img: Image.Image) -> np.ndarray:
    """Convert PIL RGB -> OpenCV BGR."""
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def map_rgb_to_color_name(rgb: np.ndarray) -> str:
    """Map an RGB color to a basic name using nearest distance."""
    basic_colors = {
        "black": np.array([0, 0, 0]),
        "white": np.array([255, 255, 255]),
        "gray": np.array([128, 128, 128]),
        "red": np.array([200, 30, 30]),
        "orange": np.array([230, 120, 40]),
        "yellow": np.array([240, 230, 50]),
        "green": np.array([50, 160, 60]),
        "blue": np.array([50, 80, 200]),
        "navy": np.array([10, 20, 80]),
        "purple": np.array([130, 60, 180]),
        "pink": np.array([230, 150, 190]),
        "brown": np.array([120, 70, 40]),
        "beige": np.array([220, 200, 160]),
        "olive": np.array([120, 130, 60]),
    }

    rgb = np.array(rgb, dtype=float)
    best_name = None
    best_dist = 1e9
    for name, ref in basic_colors.items():
        dist = np.linalg.norm(rgb - ref)
        if dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name or "unknown"


def extract_dominant_colors_masked(
    bgr_img: np.ndarray, mask: np.ndarray, k: int = 3
) -> List[str]:
    """Extract dominant colors only from pixels where mask==True."""
    if mask is None or not mask.any():
        return []

    pixels = bgr_img[mask]  
    if pixels.size == 0:
        return []

    # Convert to RGB
    rgb_pixels = cv2.cvtColor(
        pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2RGB
    ).reshape(-1, 3)
    num_pixels = rgb_pixels.shape[0]
    k = min(k, num_pixels)
    if k <= 0:
        return []

    kmeans = KMeans(n_clusters=k, n_init=5, random_state=42)
    kmeans.fit(rgb_pixels)
    centers = kmeans.cluster_centers_

    names = [map_rgb_to_color_name(c) for c in centers]

    unique = []
    for n in names:
        if n not in unique:
            unique.append(n)

    return unique[:3]


def guess_pattern_masked(bgr_img: np.ndarray, mask: np.ndarray) -> str:
    """Guess pattern from pixels inside mask using edge density."""
    if mask is None or not mask.any():
        return "unknown"

    masked = bgr_img.copy()
    masked[~mask] = 0

    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 50, 150)

    edge_values = edges[mask]
    if edge_values.size == 0:
        return "unknown"

    edge_value = edge_values.mean()

    if edge_value < 5:
        return "solid"
    elif edge_value < 20:
        return "simple pattern (maybe stripes or checks)"
    else:
        return "busy pattern (print/graphic)"


def get_person_mask(image: Image.Image) -> np.ndarray:
    """Run DeepLab to get 'person' mask. Returns boolean mask (H, W)."""
    model = get_deeplab_model()
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    img_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(img_tensor)["out"][0] 

    person_class_index = 15  
    mask = out.argmax(0).cpu().numpy() == person_class_index
    return mask


def make_skin_mask(bgr_image: np.ndarray, person_mask: np.ndarray) -> np.ndarray:
    """Basic skin mask inside the person region using YCrCb."""
    if person_mask is None or not person_mask.any():
        return np.zeros(bgr_image.shape[:2], dtype=bool)

    masked_bgr = bgr_image.copy()
    masked_bgr[~person_mask] = 0

    ycrcb = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2YCrCb)
    _, Cr, Cb = cv2.split(ycrcb)

    # Basic skin range
    skin_mask = (Cr > 135) & (Cr < 180) & (Cb > 85) & (Cb < 135)
    return skin_mask


def generate_style_feedback(
    dominant_colors: List[str],
    top_colors: List[str],
    bottom_colors: List[str],
    pattern: str,
) -> (str, List[str]):
    """
    Simple rule-based style summary and tips based on colors & pattern.
    """
    tips: List[str] = []
    summary = "Casual look."

    neutrals = {"black", "white", "gray", "beige", "brown", "navy", "olive"}
    brights = {"red", "orange", "yellow", "green", "blue", "purple", "pink"}

    color_set = set(dominant_colors)

    if not dominant_colors:
        summary = "Low-contrast or very soft outfit."
        tips.append(
            "Try combining one darker piece with one lighter piece to add a bit more contrast."
        )
        tips.append(
            "If you want the clothes to stand out more, choose one clear accent color for shoes or accessories."
        )
        return summary, tips

    # Summary by palette
    if color_set.issubset(neutrals):
        summary = "Minimal, neutral-toned outfit."
        tips.append(
            "You already have a clean neutral base. If you want more interest, add one small accent color (watch, bag, shoes)."
        )
    elif color_set & brights:
        summary = "Outfit with a pop of color."
        tips.append(
            "You have some nice color energy. Keep other pieces more neutral so the colorful item stands out."
        )
    else:
        summary = "Soft, muted color palette."
        tips.append(
            "Muted tones are easy to wear. You can add a subtle pattern or textured piece to keep it from feeling too flat."
        )

    # Pattern-based tips
    if "busy" in pattern:
        tips.append(
            "Your clothes already have a busy pattern, so keep other items simple and solid to avoid visual overload."
        )
    elif "simple pattern" in pattern:
        tips.append(
            "Simple patterns are versatile. You can pair them with either solid colors or very subtle patterns."
        )
    elif "solid" in pattern:
        tips.append(
            "Solid pieces are great for layering. Consider adding one patterned scarf, shirt, or accessory for interest."
        )

    # Top vs bottom relationship
    if top_colors and bottom_colors:
        top_main = top_colors[0]
        bottom_main = bottom_colors[0]

        if top_main in neutrals and bottom_main in neutrals:
            tips.append(
                "Since both top and bottom are neutral, you can experiment with bolder shoes or a colored jacket."
            )
        elif (
            top_main in brights and bottom_main in neutrals
        ) or (bottom_main in brights and top_main in neutrals):
            tips.append(
                "You’re using one colorful piece with a neutral one, which is a solid combo. Just keep accessories simple and cohesive."
            )
        elif top_main == bottom_main:
            tips.append(
                "Your top and bottom are similar in color, giving a monochrome vibe. Add a contrasting belt or shoes to break it up if you’d like."
            )

    if not tips:
        tips.append(
            "You can play with contrast (light vs dark, color vs neutral) and one or two accessories to make the outfit feel more intentional."
        )

    return summary, tips


@app.post("/api/analyze-outfit", response_model=AnalysisResponse)
async def analyze_outfit(file: UploadFile = File(...)):
    # Accept any image/*
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Please upload an image file.",
        )

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not read image file. Make sure it's a valid image.",
        )

    # Convert to OpenCV BGR
    bgr = pil_to_cv2(img)
    h, w, _ = bgr.shape

    # 1) Segment person
    person_mask = get_person_mask(img)
    person_pixels = int(np.count_nonzero(person_mask))

    # If no person at all -> treat as generic center-region analysis
    if person_pixels == 0:
        center_mask = np.zeros((h, w), dtype=bool)
        h_crop = int(h * 0.6)
        w_crop = int(w * 0.6)
        y1 = (h - h_crop) // 2
        x1 = (w - w_crop) // 2
        y2 = y1 + h_crop
        x2 = x1 + w_crop
        center_mask[y1:y2, x1:x2] = True

        dominant_colors = extract_dominant_colors_masked(bgr, center_mask, k=3)
        pattern_overall = guess_pattern_masked(bgr, center_mask)
        style_summary, style_tips = generate_style_feedback(
            dominant_colors, [], [], pattern_overall
        )
        note = (
            "No clear person detected. Colors and pattern were estimated from a central region of the image."
        )

        return AnalysisResponse(
            clothing_detected=bool(dominant_colors),
            dominant_colors=dominant_colors,
            top_colors=[],
            bottom_colors=[],
            pattern_overall=pattern_overall,
            style_summary=style_summary,
            style_tips=style_tips,
            note=note,
        )

    # 2) Build skin & clothing masks
    skin_mask = make_skin_mask(bgr, person_mask)
    clothing_mask = person_mask & (~skin_mask)

    clothing_pixels = int(np.count_nonzero(clothing_mask))
    skin_pixels = int(np.count_nonzero(skin_mask))

    clothing_ratio = clothing_pixels / float(person_pixels) if person_pixels > 0 else 0.0
    skin_ratio = skin_pixels / float(person_pixels) if person_pixels > 0 else 0.0

    # 3) STRONG "no clothing" detection:
    #    - clothing area is very small
    #    AND
    #    - skin dominates the person region
    #    OR
    #    - clothing pixels absolutely tiny 
    MIN_CLOTHING_PIXELS = 500  

    if (
        clothing_pixels < MIN_CLOTHING_PIXELS
        or (clothing_ratio < 0.15 and skin_ratio > 0.4)
    ):
        # We treat this as "no clear visible clothing" instead of guessing random colors
        return AnalysisResponse(
            clothing_detected=False,
            dominant_colors=[],
            top_colors=[]
            ,
            bottom_colors=[],
            pattern_overall="no clear clothing detected",
            style_summary="I couldn’t see clear clothing in this photo.",
            style_tips=[
                "Make sure your outfit is clearly visible in the frame (top and bottom).",
                "Avoid very close-up body shots if you want outfit advice.",
                "Try wearing visible garments (top and/or bottom) and keep them in the camera’s view.",
            ],
            note=(
                "Person segmentation found very few non-skin pixels compared to the whole body, "
                "so I treated this as a photo without clearly visible clothing instead of guessing random outfit colors."
            ),
        )

    # 4) Normal clothing analysis
    fallback_used = False
    if not clothing_mask.any():
        fallback_used = True
        clothing_mask = np.zeros((h, w), dtype=bool)
        h_crop = int(h * 0.6)
        w_crop = int(w * 0.6)
        y1 = (h - h_crop) // 2
        x1 = (w - w_crop) // 2
        y2 = y1 + h_crop
        x2 = x1 + w_crop
        clothing_mask[y1:y2, x1:x2] = True

    dominant_colors = extract_dominant_colors_masked(bgr, clothing_mask, k=3)
    pattern_overall = guess_pattern_masked(bgr, clothing_mask)

    # Split into top and bottom halves of clothing mask
    ys, xs = np.where(clothing_mask)
    top_colors: List[str] = []
    bottom_colors: List[str] = []

    if ys.size > 0:
        y_min, y_max = ys.min(), ys.max()
        y_mid = (y_min + y_max) // 2

        top_mask = np.zeros_like(clothing_mask)
        top_mask[y_min:y_mid, :] = clothing_mask[y_min:y_mid, :]

        bottom_mask = np.zeros_like(clothing_mask)
        bottom_mask[y_mid : y_max + 1, :] = clothing_mask[y_mid : y_max + 1, :]

        top_colors = extract_dominant_colors_masked(bgr, top_mask, k=2)
        bottom_colors = extract_dominant_colors_masked(bgr, bottom_mask, k=2)

    style_summary, style_tips = generate_style_feedback(
        dominant_colors, top_colors, bottom_colors, pattern_overall
    )

    if fallback_used:
        note = (
            "Person segmentation was uncertain for clothes, so a center region was used as a fallback. "
            "For best results, stand centered in the photo against a simpler background."
        )
    else:
        note = (
            "Colors and pattern were estimated from the person region with a heuristic skin removal, "
            "so the focus is mostly on clothing rather than background or hair."
        )

    return AnalysisResponse(
        clothing_detected=bool(dominant_colors),
        dominant_colors=dominant_colors,
        top_colors=top_colors,
        bottom_colors=bottom_colors,
        pattern_overall=pattern_overall,
        style_summary=style_summary,
        style_tips=style_tips,
        note=note,
    )
