# 🤖 DovIA v2

**DovIA** es un modelo de lenguaje generativo construido completamente desde cero en Python y PyTorch.  
Entrenado con un corpus propio sobre **Cuba, historia universal, ciencia, profesiones, psicología, geografía, salud, filosofía y conversación cotidiana**.

---

## ✨ Arquitectura

| Componente | Implementación |
|---|---|
| Arquitectura base | Transformer decoder-only |
| Normalización | RMSNorm |
| Embeddings posicionales | RoPE (Rotary Position Embedding) |
| Atención | Grouped Query Attention (GQA) |
| Activación FFN | SwiGLU |
| Tokenizador | BPE desde cero con soporte para español |
| Sampling | Temperature · Top-K · Top-P · Repetition Penalty |
| Entrenamiento | AdamW + Cosine LR + Warmup + Gradient Clipping |

---

## 🧠 Conocimiento incluido

El corpus de entrenamiento cubre **11 áreas de conocimiento**:

| Área | Contenido |
|---|---|
| 🇨🇺 Cuba | Historia desde los taínos hasta hoy, geografía, cultura, música, deporte, sociedad |
| 🌍 Historia universal | Desde las primeras civilizaciones hasta el siglo XXI |
| 🔬 Ciencia y tecnología | Física, biología, química, computación, IA |
| 🧠 Psicología | Comportamiento humano, emociones, cognición, relaciones |
| 💼 Profesiones | Medicina, derecho, ingeniería, educación, oficios |
| 🗺️ Geografía | Todos los continentes, países, ríos, montañas |
| ❤️ Salud | Enfermedades, prevención, primeros auxilios, bienestar |
| 🤔 Filosofía y ética | Grandes pensadores, valores, democracia |
| 💬 Conversación | Saludos, preguntas comunes, comportamiento social |
| 🎭 Cultura | Cine, literatura, música, deportes, gastronomía |
| ❓ Respuestas directas | 20+ respuestas a preguntas frecuentes |

---

## 🚀 Inicio rápido

### 1. Instalar PyTorch

```bash
pip install torch numpy
```

### 2. Demo completa (entrena + chat en un comando)

```bash
python demo.py
```

Esto:
1. Muestra las estadísticas del corpus
2. Entrena el tokenizador BPE
3. Entrena el modelo (≈10-20 min en CPU, ≈2-3 min en GPU)
4. Abre el chat interactivo

### 3. Solo entrenamiento

```bash
python scripts/train.py
```

### 4. Chat con modelo entrenado

```bash
# Modo pregunta-respuesta
python scripts/generate.py --prompt "¿Qué es Cuba?"

# Modo chat interactivo
python scripts/generate.py --chat

# Con parámetros personalizados
python scripts/generate.py --chat --temperature 0.9 --max_tokens 200
```

---

## 💬 Ejemplos de preguntas

```
¿Quién fue Fidel Castro?
¿Cuál es la capital de Cuba?
¿Qué es la inteligencia artificial?
¿Cuándo ocurrió la Segunda Guerra Mundial?
¿Qué hace un médico?
¿Por qué el cielo es azul?
¿Cómo manejar el estrés?
¿Qué es la democracia?
Cuéntame sobre el béisbol en Cuba
¿Qué es el ADN?
```

---

## ⚙️ Configuración

Edita los hiperparámetros al inicio de `scripts/train.py`:

```python
DEFAULT_CONFIG = {
    "vocab_size": 6000,
    "context_length": 256,
    "d_model": 256,      # Dimensión del modelo
    "n_heads": 8,        # Cabezas de atención
    "n_kv_heads": 4,     # Cabezas KV (GQA)
    "n_layers": 6,       # Capas Transformer
    "d_ff": 1024,        # Dimensión FFN
    "epochs": 15,
    "lr": 3e-4,
    "batch_size": 8,
}
```

### Tamaños sugeridos

| Tamaño | d_model | n_layers | Params | RAM |
|---|---|---|---|---|
| Mini (demo) | 128 | 4 | ~3M | 1GB |
| Small (defecto) | 256 | 6 | ~15M | 2GB |
| Base | 512 | 8 | ~60M | 6GB |
| Medium | 768 | 12 | ~125M | 12GB |

---

## 📁 Estructura del proyecto

```
DovIA2/
├── src/
│   ├── model.py          # Transformer con GQA, RoPE, SwiGLU
│   ├── tokenizer.py      # BPE desde cero para español
│   └── dataset.py        # Dataset con ventana deslizante
├── scripts/
│   ├── train.py          # Entrenamiento completo
│   └── generate.py       # Generación y chat
├── data/
│   └── corpus.py         # Corpus de conocimiento (11 áreas)
├── checkpoints/          # Modelos guardados (se crea al entrenar)
├── demo.py               # Demo completa en un comando
└── requirements.txt
```

---

## 🔧 Ampliar el conocimiento

Para añadir más conocimiento, edita `data/corpus.py` y agrega textos a las listas existentes o crea nuevas secciones:

```python
MIS_DATOS = [
    "Texto sobre el tema que quieras...",
    "Más información...",
]

FULL_CORPUS = (
    CUBA + HISTORIA_UNIVERSAL + ... + MIS_DATOS
)
```

---

## ⚠️ Expectativas realistas

- Con el corpus incluido (~500 textos), DovIA generará frases coherentes relacionadas con los temas entrenados.
- Para respuestas más fluidas y naturales, se necesita un corpus más grande (miles de textos) y más épocas de entrenamiento.
- Para alcanzar el nivel de ChatGPT se necesitan miles de millones de tokens y semanas de entrenamiento en clusters de GPUs.
- **DovIA es un excelente punto de partida** para aprender cómo funcionan los LLMs por dentro.

---

## 📄 Licencia

MIT — libre para uso personal, educativo y comercial.

---

<div align="center">
<strong>DovIA v2</strong> — Tu propio modelo de lenguaje, construido desde cero 🇨🇺
</div>
