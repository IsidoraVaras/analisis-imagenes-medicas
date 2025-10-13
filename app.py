import streamlit as st
from PIL import Image           # Abrir y manipular imagenes
from ultralytics import YOLO    # Utilizar modelo de YOLO
import io                       # Manejar archivos de memoria
import torch                    # Trabajar con tensores y modelo CLIP
import clip                     # Modelo CLIP para verificacion de imagenes

# CONFIGURACIÓN DE LA PÁGINA

st.set_page_config(
    page_title="MediScan AI - Análisis Médico",
    layout="centered"
)

# ESTILOS PERSONALIZADOS CON CSS 

st.markdown("""
    <style>
        .stApp {
            background-color: #F8F9FA;
            color: #343a40;
            padding-top: 0 !important;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 700px;
            margin-left: auto;
            margin-right: auto;
        }
        h1, h2, h3 {
            color: #007bff;
            text-align: center;
            font-weight: 600;
        }
        .stMarkdown p {
            color: #6c757d !important;
            text-align: center;
        }
        .stFileUploader > div:first-child {
            background-color: #FFFFFF;
            border: 2px dashed #CED4DA;
            border-radius: 10px;
            padding: 30px;
            text-align: center;
            box-shadow: 0 0 15px rgba(0, 0, 0, 0.05);
        }
        .stFileUploader [data-testid="stFileUploadDropzone"] svg {
            color: #007bff;
            font-size: 3em;
            margin-bottom: 15px;
        }
        .stFileUploader button {
            background-color: #007bff;
            color: #ffffff;
            border: none;
            padding: 8px 15px;
            border-radius: 5px;
        }
        [data-testid="stAlert"] {
            display: block;
        }
        [data-testid="stImage"] {
            max-width: 65%;
            margin-left: auto;
            margin-right: auto;
            border: 1px solid #E9ECEF;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .stButton button[kind="primary"] {
            background-color: #007bff;
            color: #FFFFFF;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
            font-size: 1.1em;
            margin-top: 20px;
        }
        .footer {
            text-align: center;
            margin-top: 50px;
            color: #ADB5BD;
            font-size: 0.85rem;
        }
        .stEmpty {
            display: none;
        }
    </style>
""", unsafe_allow_html=True)

# CARGAR MODELO CLIP

@st.cache_resource
def load_clip_model():
    model, preprocess = clip.load("ViT-B/32", device="cpu")
    return model, preprocess

clip_model, clip_preprocess = load_clip_model()

# FUNCION PARA DETERMINAR SI ES ECOGRAFIA

# Funcion que indica si es una ecografia. Devuelve False o True
def is_ultrasound_image(image: Image.Image) -> bool:
    if image.mode != "RGB":
        image = image.convert("RGB")    # Tranformar a RGB cuando no lo es.
    image_input = clip_preprocess(image).unsqueeze(0) #Preprocesa imagen para CLIP.
    text_prompts = [
        "an ultrasound image", "a liver ultrasound", "a medical scan", "a radiology image",
        "a cat", "a person", "a dog", "a landscape", "a normal photo"
    ]
    text_tokens = clip.tokenize(text_prompts)
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input) #Extrae caracteristicas de imagen
        text_features = clip_model.encode_text(text_tokens)   #Extrae caracteristicas de texto
        similarities = (image_features @ text_features.T).softmax(dim=-1)  #Calcula similitud
        best_match = torch.argmax(similarities, dim=-1).item()
    return best_match in [0, 1, 2, 3] #Devuelve True si parece ecografia

# CARGAR MODELO YOLO

@st.cache_resource
def load_model():
    try:
        return YOLO("best.pt")
    except Exception as e:
        st.error(f"Error al cargar el modelo 'best.pt'. Detalle: {e}")
        return None

model = load_model()
class_names = model.names if model else {0: 'F0', 1: 'F1', 2: 'F2', 3: 'F3', 4: 'F4'}

# INTERFAZ PRINCIPAL

col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.markdown("<h1>MediScan AI</h1>", unsafe_allow_html=True)
    st.markdown("<p>Plataforma Avanzada de Análisis de Imágenes Médicas</p>", unsafe_allow_html=True)

st.markdown("---")
st.markdown('<p style="font-size: 1.5em; font-weight: 600; text-align: center;">Cargar Imagen Médica</p>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #6c757d;">Arrastra y suelta tu imagen aquí, o haz clic para buscar</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(io.BytesIO(uploaded_file.getvalue()))   # Abrir imagen en memoria
    image_placeholder = st.empty()                             # Placeholder para ocultar imagen
    image_placeholder.image(image, use_container_width=True)   # Mostrar imagen

    if st.button('Analizar imagen', type="primary", use_container_width=True):
        image_placeholder.empty()  # Ocultar imagen al analizar
        with st.spinner('Analizando la imagen...'):
            if not is_ultrasound_image(image):   #Verificar que sea ecografia
                st.warning("La imagen no parece ser una ecografía médica. Por favor sube una imagen válida.")
                st.stop()
            if not model:  # Verificar que YOLO este cargado
                st.error("No se puede realizar el análisis: el modelo YOLO no se cargó correctamente.")
            else:
                try:
                    results = model(image)  # Analizar imagen con YOLO
                    pred = results[0]
                    if hasattr(pred, 'probs'):

                        # Obtener clase y confianza
                        predicted_class_index = pred.probs.top1
                        confidence = pred.probs.top1conf.item() * 100
                        diagnosis = class_names[predicted_class_index]

                        # Mostrar resultados
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f'<p style="color:#343a40; font-size: 3.5em; font-weight: bold; text-align: center;">{diagnosis}</p>', unsafe_allow_html=True)
                            st.markdown('<p style="color:#6c757d; font-size: 1em; text-align: center;">Grado de fibrosis</p>', unsafe_allow_html=True)
                        with col2:
                            st.markdown(f'<p style="color:#343a40; font-size: 3.5em; font-weight: bold; text-align: center;">{confidence:.2f}%</p>', unsafe_allow_html=True)
                            st.markdown('<p style="color:#6c757d; font-size: 1em; text-align: center;">Puntuación de confianza</p>', unsafe_allow_html=True)

                        # Mensajes según diagnostico
                        st.markdown("<hr style='border: 1px solid #E9ECEF; margin: 20px 0;'>", unsafe_allow_html=True) 
                        if diagnosis in ['F0', 'F1']:
                            st.success(f"**{diagnosis}** indica un riesgo bajo o nulo de fibrosis avanzada.")
                        elif diagnosis in ['F2', 'F3']:
                            st.warning(f"**{diagnosis}**: Fibrosis moderada a severa. Se recomienda seguimiento médico.")
                        elif diagnosis == 'F4':
                            st.error(f"**F4 (Cirrosis)**: Atención médica inmediata recomendada.")
                    else:
                        st.warning("El modelo no devolvió un resultado de clasificación válido.")
                except Exception as e:
                    st.error(f"Error al procesar la imagen: {e}")

# PIE DE PÁGINA

st.markdown('<div class="footer">Análisis generado por Inteligencia Artificial.</div>', unsafe_allow_html=True)
