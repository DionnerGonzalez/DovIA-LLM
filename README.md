<div align="center">

<img src="https://img.shields.io/badge/DovIA-v2.0-blue?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/License-Apache%202.0-green?style=for-the-badge" />
<img src="https://img.shields.io/badge/PyTorch-2.0+-red?style=for-the-badge&logo=pytorch&logoColor=white" />
<img src="https://img.shields.io/badge/Python-3.9+-yellow?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/Author-DionnerGonzalez-purple?style=for-the-badge" />

<br/><br/>

# 🤖 DovIA

### Generative Language Model — Built from Scratch

*Un modelo de lenguaje generativo construido completamente desde cero,*  
*con conocimiento sobre Cuba, historia, ciencia, psicología, profesiones y más.*

<br/>

[🚀 Inicio Rápido](#-inicio-rápido) •
[🏗️ Arquitectura](#️-arquitectura) •
[🧠 Conocimiento](#-conocimiento-integrado) •
[💬 Ejemplos](#-ejemplos-de-preguntas) •
[🤝 Contribuir](#-contribuir)

</div>

---

## 📖 ¿Qué es DovIA?

**DovIA** es un modelo de lenguaje generativo (LLM) desarrollado íntegramente desde cero por **DionnerGonzalez**. Implementa una arquitectura Transformer moderna equivalente a la de modelos como **Gemma** y **LLaMA 3**, con componentes de última generación: Grouped Query Attention, Rotary Position Embeddings y la activación SwiGLU.

A diferencia de los grandes modelos corporativos, DovIA fue diseñada para ser **transparente, educativa y extensible**, con un corpus de conocimiento propio que cubre historia de Cuba, comportamiento humano, ciencia, profesiones y conversación cotidiana en español.

> *"No necesitas miles de GPUs para entender cómo funciona la inteligencia artificial. Solo necesitas curiosidad, código limpio y ganas de aprender."* — **DionnerGonzalez**

---

## ✨ Características principales

- 🧱 **Arquitectura Transformer completa** implementada desde cero en PyTorch
- 🔍 **Grouped Query Attention (GQA)** — la misma técnica de LLaMA 3
- 📍 **Rotary Position Embeddings (RoPE)** — posicionamiento de última generación
- ⚡ **SwiGLU Feed-Forward Network** — activación usada en Gemma y PaLM
- 📐 **RMSNorm** — normalización más eficiente que LayerNorm estándar
- 🔤 **Tokenizador BPE desde cero** — con soporte nativo para español
- 🇨🇺 **Corpus propio en español** — conocimiento real sobre Cuba y el mundo
- 💬 **Modo chat interactivo** — conversa directamente con el modelo
- 💾 **Sistema de checkpoints** — guarda y reanuda el entrenamiento
- 🎛️ **Sampling avanzado** — Temperature, Top-K, Top-P, Repetition Penalty

---

## 🏗️ Arquitectura

DovIA implementa un **Transformer decoder-only** con las siguientes técnicas modernas:

```
Tokens de entrada
       │
       ▼
  [Embedding Layer]
       │
       ▼  × N capas Transformer
┌──────────────────────────────────┐
│  RMSNorm                         │
│  ┌──────────────────────────┐    │
│  │  Grouped Query Attention │◄── RoPE embeddings
│  │   Q heads: n_heads       │
│  │   K/V heads: n_kv_heads  │
│  └──────────────────────────┘    │
│  + Conexión residual             │
│                                  │
│  RMSNorm                         │
│  ┌──────────────────────────┐    │
│  │  SwiGLU FFN              │    │
│  │   Gate · Up · Down       │    │
│  └──────────────────────────┘    │
│  + Conexión residual             │
└──────────────────────────────────┘
       │
       ▼
  [RMSNorm final]
       │
       ▼
  [LM Head] ← pesos compartidos con Embedding
       │
       ▼
  Siguiente token predicho
```

### Tamaños disponibles

| Variante | d_model | Capas | Heads | Paráms aprox. | RAM mínima |
|----------|---------|-------|-------|---------------|------------|
| DovIA-Mini | 128 | 4 | 4 | ~3M | 1 GB |
| **DovIA-Small** *(defecto)* | **256** | **6** | **8** | **~15M** | **2 GB** |
| DovIA-Base | 512 | 8 | 8 | ~60M | 6 GB |
| DovIA-Medium | 768 | 12 | 12 | ~125M | 12 GB |

---

## 🧠 Conocimiento integrado

DovIA incluye un corpus de entrenamiento propio con **318 textos** en **11 áreas temáticas**:

| Área | Textos | Contenido destacado |
|------|--------|---------------------|
| 🇨🇺 **Cuba** | 63 | Historia desde los taínos, Revolución, Período Especial, geografía, cultura, música, béisbol |
| 🌍 **Historia Universal** | 40 | Civilizaciones antiguas, guerras mundiales, independencias, siglo XX-XXI |
| 🔬 **Ciencia y Tecnología** | 30 | Física, biología, química, IA, computación, cambio climático |
| 🧠 **Psicología** | 30 | Emociones, cognición, comportamiento humano, sesgos, bienestar |
| 💼 **Profesiones** | 31 | Medicina, derecho, ingeniería, educación, economía, oficios |
| 🗺️ **Geografía** | 25 | Continentes, países, ríos, montañas, océanos, capitales |
| ❤️ **Salud** | 17 | Enfermedades, prevención, nutrición, primeros auxilios, pandemia |
| 🤔 **Filosofía y Ética** | 15 | Grandes pensadores, valores, democracia, derechos humanos |
| 💬 **Conversación** | 30 | Saludos, lenguaje cotidiano, comportamiento social, emergencias |
| 🎭 **Cultura General** | 17 | Literatura, cine, música, deportes, gastronomía |
| ❓ **Respuestas Directas** | 20 | Preguntas frecuentes con respuestas completas y detalladas |

---

## 📁 Estructura del proyecto

```
DovIA/
│
├── 📂 src/
│   ├── model.py          # Arquitectura Transformer (GQA, RoPE, SwiGLU, RMSNorm)
│   ├── tokenizer.py      # Tokenizador BPE con soporte para español
│   └── dataset.py        # Dataset con ventana deslizante para entrenamiento
│
├── 📂 scripts/
│   ├── train.py          # Entrenamiento con cosine LR, warmup y checkpoints
│   └── generate.py       # Generación de texto y modo chat interactivo
│
├── 📂 data/
│   └── corpus.py         # Corpus propio (318 textos, 11 áreas de conocimiento)
│
├── 📂 checkpoints/       # Modelos guardados (generado automáticamente)
│
├── demo.py               # 🎯 Demo completa: entrena + chat en 1 solo comando
├── requirements.txt      # Dependencias mínimas (solo torch y numpy)
├── LICENSE               # Apache License 2.0 — Copyright DionnerGonzalez
└── README.md             # Este archivo
```

---

## 🚀 Inicio rápido

### Requisitos previos

- Python **3.9** o superior
- **2 GB** de RAM mínimo (4 GB recomendado)
- GPU opcional pero acelera el entrenamiento considerablemente

### 1. Clonar el repositorio

```bash
git clone https://github.com/DionnerGonzalez/DovIA.git
cd DovIA
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Demo completa en un solo comando

```bash
python demo.py
```

> Esto hace todo automáticamente: entrena el tokenizador BPE, entrena el modelo con el corpus completo y abre el chat interactivo.

---

## 🎛️ Uso avanzado

### Solo entrenamiento

```bash
python scripts/train.py
```

Para personalizar los hiperparámetros, edita `DEFAULT_CONFIG` en `scripts/train.py`:

```python
DEFAULT_CONFIG = {
    "vocab_size":      6000,
    "context_length":  256,
    "d_model":         256,   # ↑ Aumenta para mayor capacidad
    "n_heads":         8,
    "n_kv_heads":      4,     # GQA: siempre menor que n_heads
    "n_layers":        6,     # ↑ Más capas = más inteligente
    "d_ff":            1024,
    "epochs":          15,
    "lr":              3e-4,
    "batch_size":      8,
}
```

### Generar texto con parámetros

```bash
# Pregunta directa
python scripts/generate.py --prompt "¿Qué es Cuba?"

# Control total sobre la generación
python scripts/generate.py \
  --prompt "Cuéntame sobre la historia de Cuba" \
  --temperature 0.85 \
  --max_tokens 200 \
  --top_k 50 \
  --top_p 0.92 \
  --rep_penalty 1.15
```

### Chat interactivo

```bash
python scripts/generate.py --chat
```

```
══════════════════════════════════════════════════════
  🤖  DovIA v2 — Chat Interactivo
  Escribe tu pregunta. Escribe 'salir' para terminar.
══════════════════════════════════════════════════════

Tú   : ¿Quién fue José Martí?
DovIA: José Martí fue el más importante prócer de la independencia
       cubana, poeta, escritor y político que murió en combate
       en 1895 en la batalla de Dos Ríos...

Tú   : ¿Qué hace un médico?
DovIA: Un médico general diagnostica y trata una amplia variedad
       de enfermedades y coordina el cuidado integral del paciente...
```

---

## 💬 Ejemplos de preguntas

```
🇨🇺  ¿Quién fue Fidel Castro?
🇨🇺  ¿Cuál es la capital de Cuba?
🇨🇺  ¿Qué ocurrió el 1 de enero de 1959?
🇨🇺  Cuéntame sobre el béisbol en Cuba
🌍  ¿Cuándo ocurrió la Segunda Guerra Mundial?
🔬  ¿Qué es la inteligencia artificial?
🔬  ¿Por qué el cielo es azul?
🔬  ¿Cómo funciona una vacuna?
🧠  ¿Cómo manejar el estrés?
💼  ¿Qué hace un ingeniero civil?
🤔  ¿Qué es la democracia?
❤️  ¿Cómo mejorar la memoria?
```

---

## 📊 Parámetros de generación

| Parámetro | Rango recomendado | Efecto |
|-----------|-------------------|--------|
| `--temperature` | 0.6 – 1.0 | Bajo = preciso y repetitivo · Alto = creativo |
| `--top_k` | 30 – 100 | Limita los candidatos por cada paso |
| `--top_p` | 0.85 – 0.95 | Nucleus sampling: diversidad controlada |
| `--rep_penalty` | 1.1 – 1.3 | Reduce repeticiones en el texto generado |
| `--max_tokens` | 80 – 300 | Longitud máxima de la respuesta |

---

## 🔧 Ampliar el conocimiento de DovIA

Para enseñarle nuevos temas, edita `data/corpus.py`:

```python
# Añade tus propios textos
MIS_DATOS = [
    "Texto sobre cualquier tema que quieras que DovIA aprenda...",
    "Cada elemento de la lista es un párrafo de conocimiento.",
    "Puedes añadir historia local, recetas, deportes, lo que quieras.",
]

# Inclúyelo en el corpus completo
FULL_CORPUS = (
    CUBA + HISTORIA_UNIVERSAL + CIENCIA + ... + MIS_DATOS
)
```

También puedes entrenar con archivos externos (`.txt` o `.jsonl`):

```python
# En scripts/train.py:
"data_file": "data/mi_texto.txt"    # Texto plano
"data_file": "data/datos.jsonl"     # JSON Lines: {"text": "..."}
```

---

## 🤝 Contribuir

¡Las contribuciones son bienvenidas! Para contribuir al proyecto:

1. Haz un **fork** del repositorio
2. Crea una rama nueva: `git checkout -b feature/mi-mejora`
3. Realiza tus cambios y haz commit: `git commit -m "Añade mi mejora"`
4. Sube la rama: `git push origin feature/mi-mejora`
5. Abre un **Pull Request**

**Áreas donde se agradecen contribuciones:**
- 📚 Ampliar el corpus de conocimiento en español
- 🌐 Añadir soporte multiidioma
- ⚡ Optimizaciones de rendimiento
- 🖥️ Interfaz web con Gradio o Streamlit
- 📖 Mejoras en la documentación

---

## 🗺️ Roadmap

- [ ] 🌐 Interfaz web con Gradio
- [ ] 🎓 Fine-tuning con instrucciones (SFT)
- [ ] ⚡ Flash Attention para entrenamiento más rápido
- [ ] 🔗 Entrenamiento distribuido (DDP)
- [ ] 📦 Exportar a ONNX para inferencia optimizada
- [ ] 📚 Ampliar corpus a 10.000+ textos
- [ ] 📱 Versión cuantizada para dispositivos de bajo consumo

---

## 📄 Licencia

```
Copyright 2025 DionnerGonzalez

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

> Esta licencia garantiza que el nombre **DionnerGonzalez** debe aparecer en cualquier uso,
> distribución o modificación de este software. Cualquier modificación debe indicar
> claramente los cambios realizados respecto al original.

---

<div align="center">

Desarrollado con ❤️ por **DionnerGonzalez**

⭐ *Si este proyecto te fue útil, dale una estrella en GitHub* ⭐

<img src="https://img.shields.io/badge/DovIA-v2.0-blue?style=flat-square" />
<img src="https://img.shields.io/badge/Made%20in-Cuba%20🇨🇺-red?style=flat-square" />
<img src="https://img.shields.io/badge/Built%20with-PyTorch%20🔥-orange?style=flat-square" />
<img src="https://img.shields.io/badge/License-Apache%202.0-green?style=flat-square" />

</div>
