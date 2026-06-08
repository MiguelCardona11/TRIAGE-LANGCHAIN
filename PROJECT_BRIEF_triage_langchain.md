# Project Brief — Triage de correo en LangChain (Nivel 2)

> Documento de especificación para construir el proyecto. Léelo completo antes de empezar.
> Carpeta destino: `C:\Users\ASUS\Desktop\Universidad\2026-1\Sistemas Inteligentes\Trabajo Final\triage-langchain`

---

## 1. Contexto académico

Proyecto Integrador de **Sistemas Inteligentes 1** (Universidad de Caldas). El proyecto tiene 3 niveles
acumulativos sobre un mismo caso de uso. El **Nivel 1 (n8n)** ya está terminado. Este brief cubre el
**Nivel 2 (LangChain)**: reimplementar **el mismo caso de uso** del Nivel 1 usando LangChain (versión ≥ 0.2),
documentando cada componente y grabando un video de funcionamiento.

La rúbrica del Nivel 2 evalúa: equivalencia funcional con el Nivel 1 (25%), componentes LangChain
—ChatPromptTemplate, LLM, Chains/Agents, Tools— (25%), contexto y RAG / cadena de pensamiento (20%),
calidad del código y README (20%), y video (10%).

## 2. Caso de uso: Triage automático de correo electrónico

Sistema que procesa correos no leídos de una cuenta de Gmail y, para cada uno:

1. Lo **clasifica** en una de cuatro categorías: `Urgente`, `Solicitud de informacion`,
   `Spam o promocion`, `Otro`.
2. Según la categoría, **enruta y actúa**:
   - `Urgente` o `Solicitud de informacion` → **redacta un borrador de respuesta** en español y lo crea
     en el hilo del correo (NO lo envía, solo deja el borrador).
   - `Spam o promocion` o `Otro` → solo registra/etiqueta (no genera borrador).
3. **Deduplicación**: antes de crear un borrador, revisa el hilo del correo; si el hilo **ya tiene un
   borrador activo**, se omite la creación (no duplicar borradores).

## 3. Diseño de referencia del Nivel 1 (n8n) — replicar este comportamiento

El flujo n8n del Nivel 1 hace exactamente esto (la versión LangChain debe ser funcionalmente equivalente):

- **Trigger**: Gmail Trigger, sondea correos NO leídos cada minuto (máx. 3 por ciclo).
- **Clasificación**: un AI Agent con el modelo `llama-3.3-70b-versatile` (Groq) + un parser de salida
  estructurada. Devuelve un JSON con los campos: `categoria`, `urgencia`, `resumen`,
  `requiere_respuesta`, `borrador_respuesta`. Temperatura 0.2.
- **Enrutamiento**: un nodo Switch que dirige por `categoria` a 4 ramas (con fallback = `Otro`).
- **Acción**: ramas Urgente/Solicitud → crear borrador (Gmail draft) en el hilo; ramas Spam/Otro →
  aplicar etiqueta.
- **Deduplicación**: antes del borrador, un nodo lee el hilo (Gmail thread get) y un IF comprueba si
  algún mensaje del hilo tiene la etiqueta `DRAFT`; si la tiene, omite.

## 4. Stack técnico (todo gratuito)

- **Python** 3.11+, entorno virtual (`venv`).
- **LLM**: `ChatGroq` (paquete `langchain-groq`), modelo `llama-3.3-70b-versatile`, `temperature=0.2`.
  La API key de Groq ya existe (free tier). Se lee de la variable de entorno `GROQ_API_KEY`.
- **Gmail**: `GmailToolkit` del paquete `langchain-google-community[gmail]`. Expone las tools:
  `GmailSearch`, `GmailGetMessage`, `GmailGetThread`, `GmailCreateDraft`, `GmailSendMessage`.
  Requiere un archivo `credentials.json` (OAuth de escritorio) en la raíz; genera `token.json` en el
  primer uso. **Usar la misma cuenta de Gmail del Nivel 1.**
- **Salida estructurada**: Pydantic + `llm.with_structured_output(...)`.
- **Prompt**: `ChatPromptTemplate` de `langchain_core.prompts`.
- **Orquestación / "Chain"**: cadena LCEL para la clasificación (`prompt | llm.with_structured_output`)
  + lógica en Python para enrutar y llamar las tools de Gmail. (Alternativa: agente ReAct con
  `create_react_agent` de `langgraph.prebuilt`, pero la cadena + orquestación es más determinista y más
  fácil de defender en la sustentación oral, donde hay que explicar cada línea.)
- **RAG (opcional, suma 20% en la rúbrica)**: vector store local con plantillas de respuesta
  (`langchain-chroma` + `langchain-huggingface` + `sentence-transformers`, todo gratis). Al redactar el
  borrador, recupera la plantilla más parecida para guiar la respuesta. Si NO se implementa, justificar
  en la slide 08 por qué no aplica.

## 5. Componentes a implementar (mapeo con la diapositiva 07)

| Componente | Implementación |
|---|---|
| LLM | `ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)` |
| ChatPromptTemplate | system message con las reglas de triage + human template con el correo |
| Tools | tools de `GmailToolkit`: buscar no leídos, leer hilo, crear borrador |
| Chain / Agent | cadena LCEL de clasificación + orquestación en Python |
| Cadena de pensamiento (CoT) | Zero-shot CoT: añadir un campo `razonamiento` a la salida estructurada |

## 6. Esquema de salida estructurada (Pydantic)

```python
from pydantic import BaseModel, Field
from typing import Literal

class Triage(BaseModel):
    razonamiento: str = Field(description="Razonamiento paso a paso antes de decidir la categoria (CoT)")
    categoria: Literal["Urgente", "Solicitud de informacion", "Spam o promocion", "Otro"]
    urgencia: Literal["alta", "media", "baja"]
    resumen: str = Field(description="Una sola frase con el contenido del correo")
    requiere_respuesta: bool = Field(description="True solo si el correo espera una respuesta")
    borrador_respuesta: str = Field(description="Si requiere_respuesta es True, respuesta breve, cordial y profesional en espanol; si no, cadena vacia")
```

## 7. System prompt de clasificación (reutilizar el del Nivel 1)

```
Eres un asistente de triage de correo electronico. Clasifica cada correo y devuelve la respuesta en el
formato estructurado solicitado.

Categorias posibles (campo categoria):
- Urgente: requiere accion o respuesta inmediata.
- Solicitud de informacion: pide datos, aclaraciones o documentos.
- Spam o promocion: publicidad, newsletters no solicitados o phishing.
- Otro: cualquier correo que no encaje en lo anterior.

Reglas:
- urgencia debe ser alta, media o baja.
- requiere_respuesta es true solo si el correo espera una respuesta del destinatario.
- borrador_respuesta: si requiere_respuesta es true, redacta una respuesta breve, cordial y profesional
  en espanol; si es false, deja la cadena vacia.
- resumen: una sola frase con el contenido del correo.
- razonamiento: explica brevemente por que elegiste la categoria antes de decidir.
```

Human template: incluir `De: {from}`, `Asunto: {subject}`, `Cuerpo: {body}`.

## 8. Lógica de orquestación (equivalencia con n8n)

```
1. Buscar correos no leidos (GmailSearch con query "is:unread", limite ~3).
2. Para cada correo:
   a. Extraer remitente, asunto y cuerpo.
   b. Invocar la cadena de clasificacion -> objeto Triage.
   c. Enrutar por categoria:
      - Urgente / Solicitud de informacion:
          i.  Leer el hilo (GmailGetThread con el threadId).
          ii. Si el hilo YA tiene un borrador (mensaje con label DRAFT) -> omitir.
          iii.Si no -> crear borrador (GmailCreateDraft) con borrador_respuesta, dirigido al remitente,
              en el mismo hilo.
      - Spam o promocion / Otro:
          -> registrar la decision (print/log). Etiquetado opcional via el recurso de Gmail API.
3. Imprimir un resumen: por cada correo, su categoria, urgencia y accion tomada.
```

Nota: el `GmailToolkit` no trae una tool de "add label" por defecto; para etiquetar se puede usar el
recurso de la API subyacente (`build_resource_service`). Para el Nivel 2 lo esencial es
clasificacion + borrador + deduplicacion; el etiquetado es secundario y puede registrarse por consola.

## 9. Estructura del repositorio

```
triage-langchain/
├── triage_langchain.ipynb      # notebook principal, celdas comentadas por componente
├── requirements.txt
├── README.md                   # descripcion + diagrama de arquitectura + comparativa n8n vs LangChain
├── .env.example                # plantilla sin la key real
├── .gitignore                  # excluir .env, credentials.json, token.json, .venv/
└── workflow/
    └── flujo_principal.json    # JSON exportado del workflow n8n del Nivel 1 (n8n -> menu -> Download)
```

### requirements.txt

```
langchain
langchain-core
langchain-groq
langchain-google-community[gmail]
python-dotenv
jupyter
# Opcional (RAG de plantillas):
# langchain-chroma
# langchain-huggingface
# sentence-transformers
```

### .gitignore (mínimo)

```
.venv/
.env
credentials.json
token.json
__pycache__/
.ipynb_checkpoints/
```

## 10. README.md — debe incluir

- Descripción del proyecto y del caso de uso.
- Diagrama de arquitectura (puede ser en texto/Mermaid): no leidos → clasificación (Groq) → enrutamiento
  → deduplicación → borrador/etiqueta.
- **Comparativa N8n vs LangChain** (tabla), por ejemplo:

  | Aspecto | N8n (Nivel 1) | LangChain (Nivel 2) |
  |---|---|---|
  | Entrada | Gmail Trigger (polling no leidos) | GmailSearch "is:unread" |
  | LLM | nodo Groq | ChatGroq |
  | Clasificación | parser de salida estructurada | Pydantic + with_structured_output |
  | Enrutamiento | nodo Switch | if/elif en Python |
  | Borrador | nodo Gmail (draft create) | tool GmailCreateDraft |
  | Deduplicación | Gmail thread get + IF | GmailGetThread + chequeo de label DRAFT |

- Instrucciones de instalación y de credenciales (Groq + Gmail).
- **Declaración de uso de IA** (lo exige la integridad académica del proyecto): indicar qué partes fueron
  asistidas por IA y cómo.

## 11. Credenciales (instrucciones para el README y para correr)

1. **Groq**: crear key en `console.groq.com` y ponerla en `.env` como `GROQ_API_KEY`.
2. **Gmail** (Google Cloud Console): crear proyecto → habilitar Gmail API → configurar pantalla de
   consentimiento OAuth (Externo, agregarse como usuario de prueba) → crear credencial OAuth de tipo
   "Aplicación de escritorio" → descargar como `credentials.json` en la raíz. El primer run abre el
   navegador y genera `token.json`. Usar la misma cuenta de Gmail del Nivel 1.

## 12. Criterios de aceptación

- [ ] El notebook corre de principio a fin sin errores tras configurar credenciales.
- [ ] Clasifica correctamente correos de prueba en las 4 categorías.
- [ ] Crea un borrador real en Gmail para Urgente/Solicitud.
- [ ] No duplica el borrador si el hilo ya tiene uno.
- [ ] Cada componente (LLM, prompt, tools, chain, CoT) está identificado y comentado en el notebook.
- [ ] README con diagrama y comparativa n8n vs LangChain.
- [ ] Secretos NO versionados (.gitignore correcto).

## 13. Notas

- No subir secretos al repositorio bajo ninguna circunstancia.
- El estudiante debe poder explicar cada fragmento de código en la sustentación oral.
- Mantener los nombres de categoría SIN tildes (`Solicitud de informacion`, `Spam o promocion`) para que
  coincidan exactamente con la clasificación, igual que en el Nivel 1.
