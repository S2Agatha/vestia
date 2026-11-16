import os
import sys
import tkinter as tk
from tkinter import filedialog
import cv2
import h5py
import numpy as np
from tf_keras.models import load_model
from ultralytics import YOLO

# Desativa notação científica no NumPy
np.set_printoptions(suppress=True)


def caminho_recurso(caminho_relativo):
    """Retorna o caminho correto dos arquivos tanto no VS Code quanto dentro do .exe."""
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, caminho_relativo)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), caminho_relativo)


# --- CONFIGURAÇÕES ---
MODEL_PATH = caminho_recurso("keras_model.h5")
LABELS_PATH = caminho_recurso("labels.txt")
YOLO_PATH = caminho_recurso("yolov8n-pose.pt")
CONFIANCA_MINIMA = 0.85  # Exige 85% de certeza para validar o uniforme
# ---------------------

print("Carregando modelos (YOLOv8-Pose + Teachable Machine)...")
detector_pose = YOLO(YOLO_PATH)

with h5py.File(MODEL_PATH, "r") as f:
    model = load_model(f, compile=False)

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f.readlines()]


def classificar_roupa(recorte_bgr):
    """Ajusta o recorte da roupa para quadrado sem cortar as mangas (padding) e classifica."""
    if recorte_bgr.size == 0:
        return "Sem uniforme", 0.0

    h, w, _ = recorte_bgr.shape
    if h <= 10 or w <= 10:
        return "Muito distante", 0.0

    # Adiciona preenchimento (padding) nas laterais para deixar quadrado sem cortar as mangas
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

    # Trava de segurança: só considera "Com uniforme" se a certeza atingir o mínimo definido
    if "com" in nome_limpo.lower() and confidence_score < CONFIANCA_MINIMA:
        return "Sem uniforme", confidence_score

    return nome_limpo, confidence_score


def obter_caixa_roupa(keypoints_xy, keypoints_conf, caixa_pessoa, img_w, img_h):
    """Calcula a caixa delimitadora exata da camiseta usando ombros e quadril."""
    ombro_esq, ombro_dir = keypoints_xy[5], keypoints_xy[6]
    quadril_esq, quadril_dir = keypoints_xy[11], keypoints_xy[12]

    conf_ombros = min(keypoints_conf[5], keypoints_conf[6])
    conf_quadris = min(keypoints_conf[11], keypoints_conf[12])

    if conf_ombros > 0.5:
        x_ombros = [ombro_esq[0], ombro_dir[0]]
        y_ombros = [ombro_esq[1], ombro_dir[1]]

        largura_ombros = abs(ombro_esq[0] - ombro_dir[0])
        margem_x = int(largura_ombros * 0.25)  # Inclui as mangas da camiseta
        margem_topo = int(largura_ombros * 0.15)  # Pega até a gola, sem o rosto

        x1 = int(min(x_ombros) - margem_x)
        x2 = int(max(x_ombros) + margem_x)
        y1 = int(min(y_ombros) - margem_topo)

        if conf_quadris > 0.5:
            y2 = int(max(quadril_esq[1], quadril_dir[1]))
        else:
            y2 = int(y1 + largura_ombros * 1.35)

        return (
            max(0, x1),
            max(0, y1),
            min(img_w, x2),
            min(img_h, y2),
        )

    px1, py1, px2, py2 = caixa_pessoa
    altura_p = py2 - py1
    return (
        max(0, px1),
        max(0, int(py1 + altura_p * 0.22)),
        min(img_w, px2),
        min(img_h, int(py1 + altura_p * 0.68)),
    )


def detectar_e_classificar(frame):
    """Detecta a pessoa principal, isola apenas a roupa e classifica."""
    resultados = detector_pose(frame, conf=0.5, verbose=False)
    caixas = resultados[0].boxes

    if len(caixas) == 0:
        cv2.putText(
            frame,
            "Nenhuma pessoa detectada",
            (15, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        return frame

    # Encontra o índice da pessoa principal (maior área na tela)
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
    rx1, ry1, rx2, ry2 = obter_caixa_roupa(
        kpts_xy, kpts_conf, caixa_pessoa, w, h
    )

    # Recorta APENAS a roupa e classifica
    recorte_roupa = frame[ry1:ry2, rx1:rx2]
    classe, confianca = classificar_roupa(recorte_roupa)
    texto_resultado = f"{classe} ({confianca * 100:.1f}%)"

    # Verde apenas se passou na validação "Com uniforme", Laranja caso contrário
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

    return frame


def modo_upload():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    while True:
        print("\nSelecione uma imagem na janela que abriu...")
        caminho_imagem = filedialog.askopenfilename(
            title="Selecione uma imagem",
            filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp *.webp")],
        )

        if not caminho_imagem:
            print("Nenhuma imagem selecionada.")
        else:
            imagem = cv2.imread(caminho_imagem)
            if imagem is None:
                print("Erro ao carregar a imagem.")
            else:
                imagem_processada = detectar_e_classificar(imagem)

                altura, largura = imagem_processada.shape[:2]
                if altura > 850:
                    escala = 850 / altura
                    imagem_processada = cv2.resize(
                        imagem_processada, (int(largura * escala), 850)
                    )

                cv2.imshow(
                    "Resultado - Pressione qualquer tecla para sair",
                    imagem_processada,
                )
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        continuar = (
            input("\nDeseja ver outra imagem? (s/n): ").strip().lower()
        )
        if continuar not in ("s", "sim", "y", "yes"):
            print("Encerrando o modo de upload.")
            break

    root.destroy()


def modo_webcam():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erro: Não foi possível acessar a webcam.")
        return

    print("Webcam iniciada! Pressione 'ESC' ou 'q' na janela de vídeo para sair.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_processado = detectar_e_classificar(frame)
        cv2.imshow("Detector de Uniforme (Apenas Roupa)", frame_processado)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == 27 or tecla == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("\n" + "=" * 35)
    print("Escolha o modo de classificação:")
    print("1 - Fazer upload de um arquivo de imagem")
    print("2 - Usar a webcam em tempo real")
    print("=" * 35)

    opcao = input("Digite 1 ou 2: ").strip()

    if opcao == "1":
        modo_upload()
    elif opcao == "2":
        modo_webcam()
    else:
        print("Opção inválida. Execute o script novamente e digite 1 ou 2.")
