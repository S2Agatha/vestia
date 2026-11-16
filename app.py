import gc
import os
import cv2
import h5py
import numpy as np
import streamlit as st
import torch
from PIL import Image
from tf_keras.models import load_model
from ultralytics import YOLO

st.set_page_config(page_title="VestIA - Detector de Uniformes", page_icon="👁️")

st.title("👁️ VestIA")
st.subheader("Reconhecimento Inteligente de Uniformes Escolares")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "keras_model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "labels.txt")
YOLO_PATH = os.path.join(BASE_DIR, "yolov8n-pose.pt")
CONFIANCA_MINIMA = 0.85


@st.cache_resource
def carregar_modelos():
    detector = YOLO(YOLO_PATH)
    with h5py.File(MODEL_PATH, "r") as f:
        modelo_keras = load_model(f, compile=False)
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        nomes = [line.strip() for line in f.readlines()]
    return detector, modelo_keras, nomes


detector_pose, model, class_names = carregar_modelos()


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

    img_redimensionada = cv2.resize(quadrado, (224, 224), interpolation=cv2.INTER_AREA)
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


def obter_caixa_roupa(keypoints_xy, keypoints_conf, caixa_pessoa, img_w, img_h):
    ombro_esq, ombro_dir = keypoints_xy[5], keypoints_xy[6]
    quadril_esq, quadril_dir = keypoints_xy[11], keypoints_xy[12]

    conf_ombros = min(keypoints_conf[5], keypoints_conf[6])
    conf_quadris = min(keypoints_conf[11], keypoints_conf[12])

    if conf_ombros > 0.5:
        x_ombros = [ombro_esq[0], ombro_dir[0]]
        y_ombros = [ombro_esq[1], ombro_dir[1]]
        largura_ombros = abs(ombro_esq[0] - ombro_dir[0])
        margem_x = int(largura_ombros * 0.25)
        margem_topo = int(largura_ombros * 0.15)

        x1 = int(min(x_ombros) - margem_x)
        x2 = int(max(x_ombros) + margem_x)
        y1 = int(min(y_ombros) - margem_topo)
        y2 = int(max(quadril_esq[1], quadril_dir[1])) if conf_quadris > 0.5 else int(y1 + largura_ombros * 1.35)

        return max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)

    px1, py1, px2, py2 = caixa_pessoa
    altura_p = py2 - py1
    return max(0, px1), max(0, int(py1 + altura_p * 0.22)), min(img_w, px2), min(img_h, int(py1 + altura_p * 0.68))


def detectar_e_classificar(frame):
    # Limita a largura máxima da imagem para evitar pico de memória RAM
    h_orig, w_orig = frame.shape[:2]
    if w_orig > 800:
        escala = 800 / w_orig
        frame = cv2.resize(frame, (800, int(h_orig * escala)))

    with torch.no_grad():
        resultados = detector_pose(frame, conf=0.5, imgsz=320, verbose=False)

    caixas = resultados[0].boxes

    if len(caixas) == 0:
        return frame, "Nenhuma pessoa detectada", 0.0

    maior_area = 0
    idx_principal = 0
    for i, box in enumerate(caixas):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        area = (x2 - x1) * (y2 - y1)
        if area > maior_area:
            maior_area = area
            idx_principal = i

    caixa_pessoa = tuple(map(int, caixas[idx_principal].xyxy[0]))
    kpts_xy = resultados[0].keypoints.xy[idx_principal].cpu().numpy()
    kpts_conf = resultados[0].keypoints.conf[idx_principal].cpu().numpy()

    h, w, _ = frame.shape
    rx1, ry1, rx2, ry2 = obter_caixa_roupa(kpts_xy, kpts_conf, caixa_pessoa, w, h)

    recorte_roupa = frame[ry1:ry2, rx1:rx2]
    classe, confianca = classificar_roupa(recorte_roupa)
    texto_resultado = f"{classe} ({confianca * 100:.1f}%)"

    cor = (0, 255, 0) if classe.lower().startswith("com") else (0, 165, 255)
    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), cor, 3)

    (largura_texto, altura_texto), _ = cv2.getTextSize(texto_resultado, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(frame, (rx1, max(0, ry1 - altura_texto - 15)), (rx1 + largura_texto + 10, ry1), cor, -1)
    cv2.putText(frame, texto_resultado, (rx1 + 5, max(20, ry1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    gc.collect()
    return frame, classe, confianca


modo = st.radio("Escolha como testar:", ["📁 Enviar uma foto", "📷 Tirar foto pela câmera"], horizontal=True)

arquivo_img = None
if modo == "📁 Enviar uma foto":
    arquivo_img = st.file_uploader("Selecione uma imagem", type=["jpg", "jpeg", "png", "webp"])
else:
    arquivo_img = st.camera_input("Tire uma foto da camisa")

if arquivo_img is not None:
    imagem_pil = Image.open(arquivo_img).convert("RGB")
    frame_bgr = cv2.cvtColor(np.array(imagem_pil), cv2.COLOR_RGB2BGR)

    with st.spinner("Analisando uniforme..."):
        resultado_bgr, classe, confianca = detectar_e_classificar(frame_bgr)

    resultado_rgb = cv2.cvtColor(resultado_bgr, cv2.COLOR_BGR2RGB)
    st.image(resultado_rgb, caption=f"Resultado: {classe} ({confianca * 100:.1f}%)", use_container_width=True)

    if classe.lower().startswith("com"):
        st.success(f"✅ **Uniforme Escolar Identificado!** (Confiança: {confianca * 100:.1f}%)")
    elif classe == "Nenhuma pessoa detectada":
        st.error("❌ Nenhuma pessoa foi detectada na imagem.")
    else:
        st.warning(f"⚠️ **Não parece ser um uniforme escolar** ({classe} — {confianca * 100:.1f}%)")
