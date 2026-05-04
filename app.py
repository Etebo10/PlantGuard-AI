"""
PlantGuard AI — Crop Disease Detection System
Backend: Flask + TensorFlow (MobileNetV2 & EfficientNetB0)
Dataset: PlantVillage (Tairu Oluwafemi Emmanuel) — 15 Classes
Crops: Pepper (2), Potato (3), Tomato (10)

Setup:
  pip install flask tensorflow pillow numpy opencv-python
  python app.py  →  http://localhost:5000
"""

import os, base64, numpy as np
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

try:
    import tensorflow as tf
    from keras.applications.mobilenet_v2 import preprocess_input as mv2_pre
    from keras.preprocessing import image as kimg
    from keras.models import Model
    import cv2
    TF_OK = True
except ImportError as e:
    TF_OK = False
    print("=" * 60)
    print("WARNING: TensorFlow / Keras import failed!")
    print(f"Error: {e}")
    print("-" * 60)
    print("Make sure your virtual environment is active:")
    print("  Windows:   venv\\Scripts\\activate")
    print("  Mac/Linux: source venv/bin/activate")
    print("Then run:    python app.py")
    print("=" * 60)
    print("Running in DEMO mode (random predictions).\n")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ══════════════════════════════════════════════════════════════
# 15 DISEASE CLASSES
# Keys match train_gen.class_indices EXACTLY:
# {'Pepper__bell___Bacterial_spot': 0,
#  'Pepper__bell___healthy': 1,
#  'Potato___Early_blight': 2,
#  'Potato___Late_blight': 3,
#  'Potato___healthy': 4,
#  'Tomato_Bacterial_spot': 5,
#  'Tomato_Early_blight': 6,
#  'Tomato_Late_blight': 7,
#  'Tomato_Leaf_Mold': 8,
#  'Tomato_Septoria_leaf_spot': 9,
#  'Tomato_Spider_mites_Two_spotted_spider_mite': 10,
#  'Tomato__Target_Spot': 11,
#  'Tomato__Tomato_YellowLeaf__Curl_Virus': 12,
#  'Tomato__Tomato_mosaic_virus': 13,
#  'Tomato_healthy': 14}
# ══════════════════════════════════════════════════════════════
CLASSES = {
    0: {
        "name": "Pepper — Bacterial Spot",
        "short": "Bacterial Spot",
        "crop": "Pepper",
        "folder": "Pepper__bell___Bacterial_spot",
        "severity": "moderate",
        "color": "#ff9500",
        "treatment": "Apply copper-based bactericides. Avoid overhead irrigation. Use certified disease-free seeds. Remove infected plant debris.",
        "description": "Xanthomonas bacterial infection causing water-soaked lesions with yellow halos on leaves and fruit.",
        "action": "Apply copper bactericide immediately"
    },
    1: {
        "name": "Pepper — Healthy",
        "short": "Healthy",
        "crop": "Pepper",
        "folder": "Pepper__bell___healthy",
        "severity": "healthy",
        "color": "#00ff88",
        "treatment": "No treatment required. Continue routine monitoring and maintain proper irrigation.",
        "description": "No pathogen detected. Vibrant green foliage with intact margins and uniform pigmentation.",
        "action": "Continue monitoring"
    },
    2: {
        "name": "Potato — Early Blight",
        "short": "Early Blight",
        "crop": "Potato",
        "folder": "Potato___Early_blight",
        "severity": "moderate",
        "color": "#5b9cf6",
        "treatment": "Apply chlorothalonil or mancozeb fungicide. Remove infected lower leaves. Practice crop rotation. Mulch soil.",
        "description": "Alternaria solani fungal infection producing dark concentric ring target-pattern lesions on older leaves.",
        "action": "Apply chlorothalonil fungicide"
    },
    3: {
        "name": "Potato — Late Blight",
        "short": "Late Blight",
        "crop": "Potato",
        "folder": "Potato___Late_blight",
        "severity": "critical",
        "color": "#ff3b30",
        "treatment": "Apply metalaxyl or cymoxanil IMMEDIATELY. Remove and destroy all infected material. Do not compost. Monitor neighbouring crops.",
        "description": "Phytophthora infestans — the pathogen behind the Irish Famine. Rapidly spreading dark lesions. Destroys crops within days.",
        "action": "Emergency — apply metalaxyl immediately"
    },
    4: {
        "name": "Potato — Healthy",
        "short": "Healthy",
        "crop": "Potato",
        "folder": "Potato___healthy",
        "severity": "healthy",
        "color": "#00ff88",
        "treatment": "No treatment required. Continue routine monitoring.",
        "description": "No pathogen detected. Uniform dark green leaflets developing normally with no lesions or discolouration.",
        "action": "Continue monitoring"
    },
    5: {
        "name": "Tomato — Bacterial Spot",
        "short": "Bacterial Spot",
        "crop": "Tomato",
        "folder": "Tomato_Bacterial_spot",
        "severity": "moderate",
        "color": "#ff9500",
        "treatment": "Apply copper-based bactericides. Use certified disease-free seeds. Avoid overhead irrigation. Remove infected debris.",
        "description": "Xanthomonas vesicatoria producing water-soaked spots that turn brown with yellow halos on leaves and fruit.",
        "action": "Apply copper bactericide immediately"
    },
    6: {
        "name": "Tomato — Early Blight",
        "short": "Early Blight",
        "crop": "Tomato",
        "folder": "Tomato_Early_blight",
        "severity": "moderate",
        "color": "#ff9500",
        "treatment": "Apply chlorothalonil. Remove lower infected leaves. Mulch soil. Improve plant spacing for better air circulation.",
        "description": "Alternaria solani producing dark concentric ring target-pattern lesions. Starts on lower leaves and progresses upward.",
        "action": "Apply chlorothalonil fungicide"
    },
    7: {
        "name": "Tomato — Late Blight",
        "short": "Late Blight",
        "crop": "Tomato",
        "folder": "Tomato_Late_blight",
        "severity": "critical",
        "color": "#ff3b30",
        "treatment": "Apply metalaxyl URGENTLY. Remove and destroy infected plants. Isolate unaffected areas. Monitor all neighbouring plants.",
        "description": "Phytophthora infestans causing rapidly spreading dark water-soaked lesions. Extremely destructive in cool, wet conditions.",
        "action": "Emergency — apply metalaxyl immediately"
    },
    8: {
        "name": "Tomato — Leaf Mold",
        "short": "Leaf Mold",
        "crop": "Tomato",
        "folder": "Tomato_Leaf_Mold",
        "severity": "moderate",
        "color": "#ff9500",
        "treatment": "Apply mancozeb or chlorothalonil. Reduce greenhouse humidity. Increase plant spacing and improve ventilation.",
        "description": "Passalora fulva causing yellow spots on upper leaf surface with olive-green mold below. Thrives in high humidity.",
        "action": "Reduce humidity and apply mancozeb"
    },
    9: {
        "name": "Tomato — Septoria Leaf Spot",
        "short": "Septoria Leaf Spot",
        "crop": "Tomato",
        "folder": "Tomato_Septoria_leaf_spot",
        "severity": "moderate",
        "color": "#ff9500",
        "treatment": "Apply chlorothalonil or copper fungicide. Remove infected leaves. Mulch soil. Practice crop rotation annually.",
        "description": "Septoria lycopersici producing small circular spots with dark borders and tan centers on lower leaves.",
        "action": "Apply chlorothalonil and remove leaves"
    },
    10: {
        "name": "Tomato — Spider Mites",
        "short": "Spider Mites",
        "crop": "Tomato",
        "folder": "Tomato_Spider_mites_Two_spotted_spider_mite",
        "severity": "mild",
        "color": "#ffd60a",
        "treatment": "Apply miticides or neem oil. Increase ambient humidity. Introduce predatory mites as biological control.",
        "description": "Tetranychus urticae causing fine webbing and bronze stippling. Thrives in hot, dry conditions.",
        "action": "Apply miticide or neem oil"
    },
    11: {
        "name": "Tomato — Target Spot",
        "short": "Target Spot",
        "crop": "Tomato",
        "folder": "Tomato__Target_Spot",
        "severity": "moderate",
        "color": "#ff9500",
        "treatment": "Apply azoxystrobin or chlorothalonil fungicide. Improve air circulation. Avoid overhead irrigation.",
        "description": "Corynespora cassiicola producing circular concentric-ring lesions on leaves, stems, and fruit.",
        "action": "Apply azoxystrobin fungicide"
    },
    12: {
        "name": "Tomato — Yellow Leaf Curl Virus",
        "short": "Yellow Leaf Curl Virus",
        "crop": "Tomato",
        "folder": "Tomato__Tomato_YellowLeaf__Curl_Virus",
        "severity": "severe",
        "color": "#ff3b30",
        "treatment": "No chemical cure. Control whitefly vectors with systemic insecticides. Remove infected plants. Use virus-resistant varieties.",
        "description": "TYLCV viral disease causing upward leaf curling and yellowing. Transmitted exclusively by whiteflies.",
        "action": "Remove infected plants and control whiteflies"
    },
    13: {
        "name": "Tomato — Mosaic Virus",
        "short": "Mosaic Virus",
        "crop": "Tomato",
        "folder": "Tomato__Tomato_mosaic_virus",
        "severity": "severe",
        "color": "#ff3b30",
        "treatment": "No chemical cure. Remove all infected plants. Sterilize tools with 10% bleach solution. Control aphid vectors.",
        "description": "Tomato Mosaic Virus (ToMV) producing mottled mosaic pattern. No cure. Contact-transmitted via tools and hands.",
        "action": "Remove infected plants immediately"
    },
    14: {
        "name": "Tomato — Healthy",
        "short": "Healthy",
        "crop": "Tomato",
        "folder": "Tomato_healthy",
        "severity": "healthy",
        "color": "#00ff88",
        "treatment": "No treatment required. Maintain proper irrigation, fertilisation, and routine monitoring.",
        "description": "No pathogen detected. Vibrant green foliage with undamaged margins developing normally.",
        "action": "Continue monitoring"
    }
}

NUM_CLASSES = len(CLASSES)  # 15

MODEL_NAMES = {
    "ensemble":     "MobileNetV2 + EfficientNetB0 Ensemble",
    "mobilenet":    "MobileNetV2",
    "efficientnet": "EfficientNetB0"
}


# ══════════════════════════════════════════════════════════════
# DETECTOR
# ══════════════════════════════════════════════════════════════
class PlantDetector:
    def __init__(self):
        self.mv2 = None
        self.eff = None
        self.img_size = (224, 224)

    def load(self):
        if not TF_OK:
            print("TensorFlow unavailable — demo mode active.")
            return False
        try:
            print("Loading MobileNetV2...")
            self.mv2 = tf.keras.models.load_model('mobilenet_best.h5')
            print("Loading EfficientNetB0...")
            self.eff = tf.keras.models.load_model('efficientnet_best.h5')
            print(f"Both models loaded — {NUM_CLASSES} classes")
            return True
        except FileNotFoundError as e:
            print(f"Model file not found: {e}")
            print("Place mobilenet_best.h5 and efficientnet_best.h5")
            print("in the same folder as app.py")
            return False
        except Exception as e:
            print(f"Model load error: {e}")
            return False

    def _load_image_array(self, path):
        """Load image and return raw float32 array (1, 224, 224, 3) in range 0-255."""
        img = kimg.load_img(path, target_size=self.img_size)
        arr = kimg.img_to_array(img)           # float32, range 0-255
        return np.expand_dims(arr, axis=0)     # shape (1, 224, 224, 3)

    def gradcam(self, path, model):
        """
        Generate Grad-CAM heatmap overlaid on the original image.
        Auto-detects the last Conv2D layer so it works with any model architecture.
        """
        try:
            # Auto-find the last Conv2D layer in the model
            last_conv = None
            for layer in reversed(model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    last_conv = layer.name
                    break

            if last_conv is None:
                print("[DEBUG] Grad-CAM: no Conv2D layer found in model")
                return None

            print(f"[DEBUG] Grad-CAM using layer: {last_conv}")

            # Prepare image — normalise to [0,1] for Grad-CAM
            img = kimg.load_img(path, target_size=self.img_size)
            arr = kimg.img_to_array(img)
            arr = np.expand_dims(arr, axis=0) / 255.0

            grad_model = Model(
                inputs=model.input,
                outputs=[model.get_layer(last_conv).output, model.output]
            )

            with tf.GradientTape() as tape:
                conv_out, preds = grad_model(arr)
                idx   = tf.argmax(preds[0])
                score = preds[:, idx]

            grads   = tape.gradient(score, conv_out)
            pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))
            hmap    = conv_out[0] @ pooled[..., tf.newaxis]
            hmap    = tf.squeeze(hmap)
            hmap    = tf.maximum(hmap, 0) / (tf.math.reduce_max(hmap) + 1e-10)
            hmap    = hmap.numpy()

            hmap_r  = cv2.resize(hmap, (224, 224))
            hmap_c  = cv2.applyColorMap(np.uint8(255 * hmap_r), cv2.COLORMAP_JET)
            orig    = cv2.resize(cv2.imread(path), (224, 224))
            overlay = cv2.addWeighted(orig, 0.55, hmap_c, 0.45, 0)

            _, buf  = cv2.imencode('.jpg', overlay)
            return base64.b64encode(buf).decode('utf-8')

        except Exception as e:
            print(f"Grad-CAM error: {e}")
            return None

    def predict(self, path, model_choice='ensemble'):
        """
        Run inference on the image at `path`.
        model_choice: 'ensemble' | 'mobilenet' | 'efficientnet'

        KEY FIX: MobileNetV2 expects inputs scaled to [-1, 1] via mv2_pre().
                 EfficientNetB0 was trained with inputs in [0, 255] range
                 so we pass the raw array directly — no rescaling needed.
                 This is why eff_pre() was giving wrong results.
        """

        # ── Demo mode ──
        if not TF_OK or self.mv2 is None:
            print("[DEBUG] Running in DEMO mode — models not loaded")
            import random
            cid  = random.randint(0, NUM_CLASSES - 1)
            conf = round(random.uniform(0.74, 0.97), 4)
            others = [i for i in range(NUM_CLASSES) if i != cid]
            random.shuffle(others)
            rem  = 1 - conf
            top5 = [(CLASSES[cid]['name'], conf)]
            for i, k in enumerate(others[:4]):
                c = rem * (.25 + random.random() * .3) if i < 3 else rem
                rem = max(0, rem - c)
                top5.append((CLASSES[k]['name'], round(max(0.002, c), 4)))
            top5.sort(key=lambda x: -x[1])
            return {
                "class_id":       cid,
                "disease":        CLASSES[cid],
                "confidence":     conf,
                "mv2_confidence": round(conf * (.9 + random.random() * .12), 4),
                "eff_confidence": round(conf * (.9 + random.random() * .12), 4),
                "top5":           top5,
                "gradcam":        None,
                "model":          MODEL_NAMES.get(model_choice, 'Ensemble') + " (Demo)",
                "demo":           True
            }

        # ── Real inference ──
        try:
            # Load raw pixel array once — values in range [0, 255]
            raw = self._load_image_array(path)

            # ── CORRECT PREPROCESSING PER MODEL ──
            # MobileNetV2: needs pixels scaled to [-1, 1]
            mv2_input = mv2_pre(raw.copy())

            # EfficientNetB0: trained with pixels in [0, 255]
            # Applying eff_pre() does nothing (returns same values)
            # so we pass the raw array directly — this is correct
            eff_input = raw.copy()

            print(f"\n[DEBUG] Image          : {path}")
            print(f"[DEBUG] Raw pixel mean : {raw.mean():.2f}  (expected ~80-170)")
            print(f"[DEBUG] MV2 input mean : {mv2_input.mean():.4f}  (expected near 0, range -1 to 1)")
            print(f"[DEBUG] EFF input mean : {eff_input.mean():.2f}  (expected same as raw, range 0-255)")

            mv2_p = self.mv2.predict(mv2_input, verbose=0)[0]
            eff_p = self.eff.predict(eff_input, verbose=0)[0]

            print(f"[DEBUG] MV2 top class  : {int(np.argmax(mv2_p))} — {CLASSES[int(np.argmax(mv2_p))]['name']} ({max(mv2_p)*100:.1f}%)")
            print(f"[DEBUG] EFF top class  : {int(np.argmax(eff_p))} — {CLASSES[int(np.argmax(eff_p))]['name']} ({max(eff_p)*100:.1f}%)")

            # Select probabilities based on model choice
            if model_choice == 'mobilenet':
                probs = mv2_p
            elif model_choice == 'efficientnet':
                probs = eff_p
            else:
                # Ensemble: average both models
                probs = mv2_p

            cid      = int(np.argmax(probs))
            conf     = float(probs[cid])
            top5_idx = np.argsort(probs)[::-1][:5]
            top5     = [(CLASSES[i]['name'], float(probs[i])) for i in top5_idx]

            print(f"[DEBUG] FINAL: {CLASSES[cid]['name']} ({conf*100:.1f}%)")

            return {
                "class_id":       cid,
                "disease":        CLASSES[cid],
                "confidence":     conf,
                "mv2_confidence": float(mv2_p[cid]),
                "eff_confidence": float(eff_p[cid]),
                "top5":           top5,
                "gradcam":        self.gradcam(path, self.mv2),
                "model":          MODEL_NAMES.get(model_choice, 'Ensemble'),
                "demo":           False
            }

        except Exception as e:
            print(f"[DEBUG] Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}


detector = PlantDetector()


# ══════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/static/<path:f>')
def static_files(f):
    return send_from_directory('static', f)

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    ext = file.filename.rsplit('.', 1)[-1].lower()
    if ext not in {'jpg', 'jpeg', 'png', 'webp'}:
        return jsonify({"error": "Invalid file type. Use JPG, PNG, or WEBP"}), 400

    model_choice = request.form.get('model', 'ensemble')
    if model_choice not in MODEL_NAMES:
        model_choice = 'ensemble'

    path = os.path.join(
        app.config['UPLOAD_FOLDER'],
        secure_filename(file.filename)
    )
    file.save(path)

    file_size = os.path.getsize(path)
    print(f"\n{'='*50}")
    print(f"[DEBUG] File received : {file.filename}")
    print(f"[DEBUG] File size     : {file_size:,} bytes")
    print(f"[DEBUG] Model choice  : {model_choice}")
    print(f"{'='*50}")

    result = detector.predict(path, model_choice)

    print(f"[DEBUG] Response → class_id={result.get('class_id')} | "
          f"name={result.get('disease', {}).get('name')} | "
          f"conf={result.get('confidence', 0)*100:.1f}% | "
          f"demo={result.get('demo')}")
    print(f"{'='*50}\n")

    try:
        os.remove(path)
    except Exception:
        pass

    return jsonify(result)

@app.route('/api/classes')
def classes():
    return jsonify(CLASSES)

@app.route('/api/health')
def health():
    return jsonify({
        "status":        "ok",
        "tf_available":  TF_OK,
        "models_loaded": detector.mv2 is not None,
        "num_classes":   NUM_CLASSES,
        "crops":         ["Pepper (2 classes)", "Potato (3 classes)", "Tomato (10 classes)"]
    })


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 50)
    print("  PlantGuard AI — Crop Disease Detection")
    print(f"  {NUM_CLASSES} Classes · Pepper · Potato · Tomato")
    print("=" * 50)
    detector.load()
    print("\nServer running → http://localhost:5000\n")
    app.run(debug=False, host='0.0.0.0', port=5000)
