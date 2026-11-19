# Kaldrishti — Image Deepfake Detection Engine

**Kaldrishti** checks a single image to see if it is a **real photo** or a **fake / edited image** (like an AI-generated face, a face swap, or a photo edited in Photoshop). Instead of relying on just one model, it runs **four different checks** in parallel and combines them into a clear score plus a visual map showing where changes may be.

---

## How Kaldrishti Works (Simple Flow)

1. **Take the Image:** You give Kaldrishti an image (for example, comparing a real photo against a fake photo).
2. **Prepare Two Views:**
   * It keeps the **full original image** so background lighting and camera details stay intact.
   * It finds any face on the image and aligns it into a clean `256x256` face view.
3. **Run 4 Checks at Once:**
   * **Check 1 — Global & Lighting Look (`DINOv2 / DINOv3` + `CLIP-ViT` + `Effort`):** Checks if the whole picture makes physical and visual sense.
   * **Check 2 — Blending Edges (`SBI` + `EfficientNet-B4` + `XceptionNet`):** Looks closely around the face, jawline, and hair for face-swap edges.
   * **Check 3 — Hidden Frequency Patterns (`2D-FFT` + `SRM` + `F3-Net`):** Looks at frequency patterns to spot repeating grid marks from AI image generation or upsampling.
   * **Check 4 — Camera Noise & Manual Edits (`TruFor` + `Noiseprint++` + `CAT-Net`):** Checks camera sensor noise and compression clues for manual edits and builds a **heatmap**.
4. **Combine into One Final Score:** Kaldrishti combines all four checks into one score from `0%` to `100%` fake likelihood and labels the result **Likely Real**, **Alert**, or **Likely Fake**.

---

## Which Model Is Used Where, Why, and the Math Behind It

### Preprocessing — Face Finding & Alignment
* **Models / Algorithms:** **RetinaFace** + **68-Point Facial Landmarks** + **Affine Alignment**.
* **Where Used:** Before running face-focused checks, on the input image.
* **Why Used:** Faces can be tilted, small, or off-center. Aligning them to a standard `256x256` box helps the model compare the same facial spots every time, while keeping the full image preserves background lighting and camera noise.
* **Math Concept (2D Affine Transform):** Moves each point $(x, y)$ to a new spot $(x', y')$ using rotation, scaling, and shifting without bending straight lines into curves:
  $$\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} m_{11} & m_{12} \\ m_{21} & m_{22} \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} + \begin{bmatrix} t_x \\ t_y \end{bmatrix}$$

---

### Check 1 — Global & Lighting Look
* **Models / Algorithms:** **DINOv2 / DINOv3** + **CLIP-ViT** + **Effort** (SVD subspace adaptation).
* **Where Used:** **Stream A** on the overall image.
* **Why Used:** Full AI images (from diffusion or generative models) may not have a cut-and-paste seam, so you have to look at the whole scene—like whether reflections in the eyes match shadows on the face, or if hands and objects look physically wrong. **DINOv2** holds up well even when images are compressed, and **Effort** helps the model handle newer AI generators without forgetting what it already knows.
* **Math Concept (Self-Attention & SVD Subspace):**
  * **Self-Attention** compares every image patch against other patches to check global consistency:
    $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
  * **Singular Value Decomposition (SVD)** breaks a weight matrix $W_0 = U\Sigma V^T$ into ordered directions so Kaldrishti can freeze core visual knowledge and train only a small forgery adjustment $\Delta W$.

---

### Check 2 — Blending Edges
* **Models / Algorithms:** **SBI (Self-Blended Images)** + **EfficientNet-B4** + **XceptionNet**.
* **Where Used:** **Stream B** on the aligned face region.
* **Why Used:** Face-swaps paste one face onto another head. Even when lighting looks okay from afar, the border around the jawline, eyes, or hair can show faint edge differences or skin-texture mismatches. Training with **SBI** teaches the network to spot synthetic blending boundaries.
* **Math Concept (Convolution & Depthwise Separable Filtering):** Slides small filter weights $W_k$ across image patches $X_{i,j}$ to pick up local edge and texture changes:
  $$F_{i,j}^c = \sigma \left( \sum_{k} W_k * X_{i+j} + b_c \right)$$
  **XceptionNet** splits spatial filtering and channel mixing into separate steps so subtle texture signals are not drowned out by color channels.

---

### Check 3 — Hidden Frequency Patterns
* **Models / Algorithms:** **2D-FFT (Fast Fourier Transform)** + **SRM (Spatial Rich Models)** + **F3-Net**.
* **Where Used:** **Stream C** to inspect frequency patterns across the image.
* **Why Used:** AI generators and upsamplers often leave behind faint, repeating grid-like patterns that are hard to see in normal pixel view. **SRM** high-pass filters strip away smooth colors to expose fine noise, and frequency analysis highlights unusual spikes.
* **Math Concept (2D Discrete Fourier Transform):** Converts pixel brightness $x(h, w)$ from space into frequency waves $X(u, v)$ so repeating patterns show up as clear peaks:
  $$X(u, v) = \sum_{h=0}^{H-1} \sum_{w=0}^{W-1} x(h, w) e^{-j 2\pi \left(\frac{uh}{H} + \frac{vw}{W}\right)}$$
  Real camera photos usually follow a smooth drop-off at higher frequencies (roughly $1/f^\alpha$), whereas AI upsampling can create extra spikes.

---

### Check 4 — Camera Noise & Manual Edits
* **Models / Algorithms:** **TruFor** + **Noiseprint++** + **CAT-Net**.
* **Where Used:** **Stream D** to check camera sensor noise, compression blocks, and pixel-level edit maps.
* **Why Used:** A fraudster might not use AI at all—they might use a photo editor to paste a face, clone part of a background, or edit text on an ID. **Noiseprint++** looks at camera sensor noise patterns, **CAT-Net** checks JPEG compression blocks, and **TruFor** creates a **heatmap** $H_{\text{map}}$ showing where edits likely happened.
* **Math Concept (Noise Residual & Cross-Modal Attention):** Subtracting a denoised version $\mathcal{D}_{\theta}(I_{\text{in}})$ from the image leaves fine residual details:
  $$N_{\text{res}} = I_{\text{in}} - \mathcal{D}_{\theta}(I_{\text{in}})$$
  Comparing this residual with the image view helps highlight localized tampered regions.

---

### Final Step — Combining the Four Scores
* **Model / Algorithm:** **Temperature-Scaled Logistic Regression**.
* **Where Used:** **Stage 3 (Fusion)** at the end of the pipeline.
* **Why Used:** Raw model scores can be overly extreme (right next to `0` or `1`). Temperature scaling smooths and calibrates the combined score so thresholds like `15%` and `75%` behave reliably.
* **Math Concept (Sigmoid with Temperature $T$):** Turns weighted check scores $s_{\text{sem}}, s_{\text{bnd}}, s_{\text{freq}}, s_{\text{edit}}$ into a smooth probability between `0` and `1`:
  $$P(\text{Fake} \mid I_{\text{in}}) = \frac{1}{1 + \exp\left( -\frac{w_0 + w_1 s_{\text{sem}} + w_2 s_{\text{bnd}} + w_3 s_{\text{freq}} + w_4 s_{\text{edit}}}{T} \right)}$$

---

## Real vs. Fake Comparison Example

| Image Type | Global Look | Blending Edges | Frequency Check | Sensor / Edit Check | Final Fake Score | Result |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Real Photo** | Low (`0.04`) | Low (`0.06`) | Low (`0.08`) | Low (`0.03`) | **5%** | **Likely Real** |
| **Heavily Compressed Image** | Medium (`0.28`) | Medium (`0.41`) | Medium (`0.52`) | Medium (`0.30`) | **39%** | **Alert** (needs review) |
| **AI Fake / Edited Image** | High (`0.89`) | High (`0.94`) | High (`0.91`) | High (`0.85`) | **92%** | **Likely Fake** (+ heatmap) |

---

## Where It Can Be Used

* **KYC & Identity Checks:** Spotting fake selfies or edited ID photos.
* **News & Media Verification:** Checking if an image shared online is real or AI-made.
* **Insurance & Claims:** Checking whether damage or claim photos were edited.
* **Content Moderation:** Flagging fake or manipulated images on apps and platforms.

---

## Why Use Kaldrishti Instead of Just One Model?

* **One model can miss things:** A face-swap detector might miss a full AI image, and an AI image detector might miss a simple Photoshop cut-and-paste. Using four checks together covers more types of fakes.
* **Handles newer AI tools better:** Using foundation vision models (`DINOv2` / `CLIP-ViT`) with subspace adaptation (`Effort`) helps it spot newer AI generators better than older single models.
* **Shows where edits happened:** Alongside a score, it can produce a **heatmap** so a person can visually see what looks off.
* **Simple trade-off:** Running four checks takes more computer power and time than running one small model, so it is best used when accuracy matters more than ultra-fast basic filtering.

---

## Folder Structure

```text
kaldrishti/
├── README.md
├── requirements.txt
├── .env.example
├── configs/
│   ├── pipeline.yaml
│   ├── stream_a_semantic.yaml
│   ├── stream_b_boundary.yaml
│   ├── stream_c_frequency.yaml
│   └── stream_d_sensor_edit.yaml
├── weights/
├── data/
│   └── samples/
├── src/
│   └── kaldrishti/
│       ├── preprocessing/
│       │   ├── metadata.py
│       │   ├── face_detector.py
│       │   └── affine_align.py
│       ├── streams/
│       │   ├── base.py
│       │   ├── stream_a_semantic.py
│       │   ├── stream_b_boundary.py
│       │   ├── stream_c_frequency.py
│       │   └── stream_d_sensor_edit.py
│       ├── fusion/
│       │   ├── calibrator.py
│       │   ├── heatmap.py
│       │   └── decision_engine.py
│       ├── pipeline/
│       │   ├── engine.py
│       │   └── comparator.py
│       └── api/
│           ├── app.py
│           └── routes/
│               └── detect.py
├── scripts/
│   ├── run_compare.py
│   └── download_weights.py
└── tests/
```

---

## Quick Start

```bash
# 1. Create and turn on a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install packages
pip install -r requirements.txt

# 3. Copy example settings
cp .env.example .env

# 4. Compare a real image and a fake image
python scripts/run_compare.py \
  --real data/samples/real.jpg \
  --fake data/samples/fake.jpg
```

---

## Score Guide

* **Below `15%` (`0.00` to `0.15`):** **Likely Real** — No strong signs of AI generation or tampering.
* **`15%` to `74%` (`0.15` to `0.75`):** **Alert** — Unclear or compressed image; worth a closer look or second check.
* **`75%` and above (`0.75` to `1.00`):** **Likely Fake** — Strong signs of manipulation; check the heatmap for highlighted areas.
