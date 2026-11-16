import base64
import gc
import os
import cv2
import h5py
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image
from tf_keras.models import load_model

st.set_page_config(page_title="VestIA - Detector de Uniformes", page_icon="👁️")

st.title("👁️ VestIA")
st.subheader("Reconhecimento Inteligente de Uniformes Escolares")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "keras_model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "labels.txt")
CONFIANCA_MINIMA = 0.85


@st.cache_resource
def carregar_modelos():
    mp_pose = mp.solutions.pose
    detector = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5,
    )
    with h5py.File(MODEL_PATH, "r") as f:
        modelo_keras = load_model(f, compile=False)
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        nomes = [line.strip() for line in f.readlines()]
    return detector, modelo_keras, nomes


detector_pose, model, class_names = carregar_modelos()


def exibir_imagem_garantida(imagem_bgr, legenda=""):
    """Converte a imagem para Base64 e exibe via HTML direto (nunca quebra no Streamlit Cloud)."""
    sucesso, buffer = cv2.imencode(".jpg", imagem_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if sucesso:
        img_b64 = base64.b64encode(buffer).decode("utf-8")
        html = f"""
        <div style="text-align: center; margin: 15px 0;">
            <img src="data:image/jpeg;base64,{img_b64}" style="max-width: 100%; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.4);" />
            <p style="color: #a0a0a0; font-size: 0.9rem; margin-top: 8px;">{legenda}</p>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)


def classificar_roupa(recorte_bgr):
    if recorte_bgr.size == 0:
        return "Sem uniforme", 0.0
    h, w, _ = recorte_bgr.shape
    if h <= 10 or w <= 10:
        return "Muito distante", 0.0

    maior_lado = max(h, w)
    quadrado = np.zeros((maior_lado, maior_lado, 3), dtype=np.uint8)
    offset_y = (maior_lado - h) // 2
    offset_x = (maior_lado - w) // 2
    quadrado[offset_y : offset_y + h, offset_x : offset_x + w] = recorte_bgr

    img_redimensionada = cv2.resize(
        quadrado, (224, 224), interpolation=cv2.INTER_AREA
    )
    img_rgb = cv2.cvtColor(img_redimensionada, cv2.COLOR_BGR2RGB)

    image_array = np.asarray(img_rgb, dtype=np.float32).reshape(1, 224, 224, 3)
    normalized_image_array = (image_array / 127.5) - 1

    prediction = model.predict(normalized_image_array, verbose=0)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = float(prediction[0][index])

    nome_limpo = class_name[2:] if len(class_name) > 2 else class_name
    if "com" in nome_limpo.lower() and confidence_score < CONFIANCA_MINIMA:
        return "Sem uniforme", confidence_score

    return nome_limpo, confidence_score


def detectar_e_classificar(frame):
    h_orig, w_orig = frame.shape[:2]
    if w_orig > 800:
        escala = 800 / w_orig
        frame = cv2.resize(frame, (800, int(h_orig * escala)))

    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultados = detector_pose.process(frame_rgb)

    if not resultados.pose_landmarks:
        return frame, "Nenhuma pessoa detectada", 0.0

    lm = resultados.pose_landmarks.landmark
    ombro_esq, ombro_dir = lm[11], lm[12]
    quadril_esq, quadril_dir = lm[23], lm[24]

    if min(ombro_esq.visibility, ombro_dir.visibility) < 0.4:
        return frame, "Nenhuma pessoa detectada", 0.0

    x_ombros = [ombro_esq.x * w, ombro_dir.x * w]
    y_ombros = [ombro_esq.y * h, ombro_dir.y * h]

    largura_ombros = abs(x_ombros[0] - x_ombros[1])
    margem_x = int(largura_ombros * 0.25)
    margem_topo = int(largura_ombros * 0.15)

    rx1 = max(0, int(min(x_ombros) - margem_x))
    rx2 = min(w, int(max(x_ombros) + margem_x))
    ry1 = max(0, int(min(y_ombros) - margem_topo))

    if min(quadril_esq.visibility, quadril_dir.visibility) > 0.4:
        ry2 = min(h, int(max(quadril_esq.y * h, quadril_dir.y * h)))
    else:
        ry2 = min(h, int(ry1 + largura_ombros * 1.35))

    recorte_roupa = frame[ry1:ry2, rx1:rx2]
    classe, confianca = classificar_roupa(recorte_roupa)
    texto_resultado = f"{classe} ({confianca * 100:.1f}%)"

    cor = (0, 255, 0) if classe.lower().startswith("com") else (0, 165, 255)
    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), cor, 3)

    (largura_texto, altura_texto), _ = cv2.getTextSize(
        texto_resultado, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
    )
    cv2.rectangle(
        frame,
        (rx1, max(0, ry1 - altura_texto - 15)),
        (rx1 + largura_texto + 10, ry1),
        cor,
        -1,
    )
    cv2.putText(
        frame,
        texto_resultado,
        (rx1 + 5, max(20, ry1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
    )

    gc.collect()
    return frame, classe, confianca


modo = st.radio(
    "Escolha como testar:",
    ["📁 Enviar uma foto", "📷 Tirar foto pela câmera"],
    horizontal=True,
)
