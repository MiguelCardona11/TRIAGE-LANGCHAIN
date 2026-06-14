#!/usr/bin/env python3
"""
Triage automatico de correo con LangChain — Nivel 2
=====================================================

Reimplementacion en LangChain (>= 0.2) del workflow n8n del Nivel 1:
clasifica correos no leidos de Gmail y, segun la categoria, redacta un
borrador de respuesta (sin enviarlo) o solo registra la decision,
evitando duplicar borradores.

Componentes (mapeo diapositiva 07):
  - LLM:              ChatGroq (llama-3.3-70b-versatile, temp=0.2)
  - ChatPromptTemplate: system message + human template con variables
  - Tools:             GmailToolkit (search, get_message, get_thread, create_draft)
  - Chain / Agent:     cadena LCEL  prompt | llm.with_structured_output(Triage)
  - CoT:               campo razonamiento (Zero-shot CoT)
  - RAG (opcional):    Chroma + HuggingFaceEmbeddings para plantillas de respuesta

Uso:
    python triage_langchain.py               # procesa bandeja real (crea borradores)
    python triage_langchain.py --dry-run      # pasada en seco (no escribe en Gmail)
    python triage_langchain.py --demo         # clasifica 4 correos de ejemplo (sin Gmail)
    python triage_langchain.py --rag          # activa RAG de plantillas
"""

import argparse
import base64
import os
import sys
from email.mime.text import MIMEText
from typing import Literal, Union

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

load_dotenv()
assert os.getenv("GROQ_API_KEY"), "Falta GROQ_API_KEY en .env"

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_google_community import GmailToolkit
from langchain_google_community.gmail.utils import (
    build_resource_service,
    get_gmail_credentials,
)

from googleapiclient.errors import HttpError

CATEGORIAS_RESPONDER = {"Urgente", "Solicitud de informacion"}

USE_RAG = False
recuperador = None

PLANTILLAS = [
    "Estimado/a, gracias por su mensaje. Atenderemos su caso con prioridad y le confirmaremos "
    "la solucion a la mayor brevedad. Quedamos atentos.",
    "Hola, gracias por su consulta. Le adjuntamos/enviamos la informacion solicitada. Si necesita "
    "algun dato adicional, con gusto se lo facilitamos.",
    "Buenas, agradecemos su correo. Hemos registrado su solicitud y le daremos respuesta dentro de "
    "nuestro horario de atencion. Gracias por su paciencia.",
    "Estimado/a, confirmamos la recepcion de su solicitud de documentos. Procedemos a prepararlos y "
    "se los remitiremos en breve.",
]


class Triage(BaseModel):
    """Salida estructurada del triage de un correo."""

    razonamiento: str = Field(
        description="Razonamiento paso a paso antes de decidir la categoria (CoT)"
    )
    categoria: Literal[
        "Urgente", "Solicitud de informacion", "Spam o promocion", "Otro"
    ] = Field(description="Categoria del correo")
    urgencia: Literal["alta", "media", "baja"] = Field(description="Nivel de urgencia")
    resumen: str = Field(description="Una sola frase con el contenido del correo")
    requiere_respuesta: Union[bool, str] = Field(
        description="True solo si el correo espera una respuesta del destinatario"
    )
    borrador_respuesta: str = Field(
            description=(
                "Si requiere_respuesta es True, respuesta breve, cordial y profesional en "
                "espanol; si no, cadena vacia"
            )
        )

    @field_validator("requiere_respuesta", mode="before")
    @classmethod
    def coerce_bool(cls, v):
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes", "si", "sí")
        return v


SYSTEM_TRIAGE = """Eres un asistente de triage de correo electronico. Clasifica cada correo y devuelve la respuesta en el formato estructurado solicitado.

Categorias posibles (campo categoria):
- Urgente: requiere accion o respuesta inmediata.
- Solicitud de informacion: pide datos, aclaraciones o documentos.
- Spam o promocion: publicidad, newsletters no solicitados o phishing.
- Otro: cualquier correo que no encaje en lo anterior.

Reglas:
- urgencia debe ser alta, media o baja.
- requiere_respuesta es un booleano True solo si el correo espera una respuesta del destinatario, de lo contrario es False. No puede ser string
- ALERTAS DE SEGURIDAD, notificaciones del sistema, boletines y correos automaticos: requiere_respuesta=False. Son urgentes de leer pero NO requieren que escribas una respuesta.
- borrador_respuesta: si requiere_respuesta es true, redacta una respuesta breve, cordial y profesional en espanol; si es false, deja la cadena vacia ("").
- resumen: una sola frase con el contenido del correo.
- resumen: una sola frase con el contenido del correo.
- razonamiento: explica brevemente por que elegiste la categoria antes de decidir."""

prompt_triage = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_TRIAGE),
    ("human",
     "Clasifica el siguiente correo:\n\n"
     "De: {remitente}\n"
     "Asunto: {asunto}\n"
     "Cuerpo: {cuerpo}"),
])


correos_demo = [
    {
        "remitente": "jefe@empresa.com",
        "asunto": "URGENTE: caida del servidor de produccion",
        "cuerpo": "El sitio esta caido desde hace 10 minutos. Necesito que lo revises ya y me confirmes.",
    },
    {
        "remitente": "cliente@correo.com",
        "asunto": "Consulta sobre la factura 0234",
        "cuerpo": "Buenas tardes, podrian enviarme el desglose de la factura 0234? Gracias.",
    },
    {
        "remitente": "ofertas@promos.com",
        "asunto": "70 de descuento solo HOY, compra ya",
        "cuerpo": "Aprovecha esta oferta unica. Haz clic aqui para ganar un premio.",
    },
    {
        "remitente": "boletin@universidad.edu",
        "asunto": "Boletin mensual de la facultad",
        "cuerpo": "Resumen de actividades del mes. No requiere ninguna accion de tu parte.",
    },
]


def init_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
    )


def init_chain(llm):
    return prompt_triage | llm.with_structured_output(Triage)


def init_gmail():
    credentials = get_gmail_credentials(
        token_file="token.json",
        scopes=["https://mail.google.com/"],
        client_sercret_file="credentials.json",
    )
    api_resource = build_resource_service(credentials=credentials)
    toolkit = GmailToolkit(api_resource=api_resource)
    tools = {t.name: t for t in toolkit.get_tools()}
    return api_resource, tools


def init_rag():
    global USE_RAG, recuperador
    if not USE_RAG:
        return
    try:
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vector_store = Chroma.from_texts(
            texts=PLANTILLAS,
            embedding=embeddings,
            collection_name="plantillas_respuesta",
        )
        recuperador = vector_store.as_retriever(search_kwargs={"k": 1})
        print("[RAG] Vector store de plantillas listo.")
    except ImportError:
        print("[RAG] Faltan dependencias (langchain-chroma, langchain-huggingface, "
              "sentence-transformers). Continuo SIN RAG.")
        USE_RAG = False


def recuperar_plantilla(texto: str) -> str:
    if not USE_RAG or recuperador is None:
        return ""
    docs = recuperador.invoke(texto)
    return docs[0].page_content if docs else ""


def extraer_campos(msg: dict) -> dict:
    return {
        "remitente": msg.get("sender", "") or "",
        "asunto": msg.get("subject", "") or "",
        "cuerpo": (msg.get("body") or msg.get("snippet") or "").strip(),
    }


def etiquetar_mensaje(api_resource, message_id: str, etiquetas: list[str]) -> dict:
    try:
        return api_resource.users().messages().modify(
            userId="me",
            id=message_id,
            body={"addLabelIds": etiquetas},
        ).execute()
    except HttpError as e:
        print(f"[WARN] No se pudo etiquetar mensaje {message_id}: {e}")
        return {}


def hilo_tiene_borrador(api_resource, thread_id: str) -> bool:
    try:
        hilo = api_resource.users().threads().get(userId="me", id=thread_id).execute()
        for mensaje in hilo.get("messages", []):
            if "DRAFT" in mensaje.get("labelIds", []):
                return True
    except HttpError as e:
        print(f"[WARN] No se pudo leer el hilo {thread_id}: {e}")
    return False


def crear_borrador_en_hilo(api_resource, to: str, asunto: str, cuerpo: str, thread_id: str) -> dict:
    if not cuerpo or not cuerpo.strip():
        raise ValueError("El cuerpo del borrador esta vacio, no se creara el draft")
    asunto_resp = asunto if asunto.lower().startswith("re:") else "Re: " + asunto
    mime = MIMEText(cuerpo, "plain", "utf-8")
    mime["To"] = to
    mime["Subject"] = asunto_resp
    raw = base64.urlsafe_b64encode(mime.as_bytes()).decode()
    return api_resource.users().drafts().create(
        userId="me",
        body={"message": {"raw": raw, "threadId": thread_id}},
    ).execute()


def redactar_borrador(llm, triage, campos: dict) -> str:
    if USE_RAG:
        consulta = campos["asunto"] + " " + campos["cuerpo"]
        plantilla = recuperar_plantilla(consulta)
        prompt_borrador = ChatPromptTemplate.from_messages([
            ("system",
             "Eres un asistente que redacta respuestas de correo en espanol: breves, cordiales y "
             "profesionales. Usa la siguiente plantilla como guia de tono y estructura (adaptala al "
             "caso, no la copies literalmente):\n\n{plantilla}"),
            ("human",
             "Correo original:\nDe: {remitente}\nAsunto: {asunto}\nCuerpo: {cuerpo}\n\n"
             "Redacta unicamente el cuerpo de la respuesta."),
        ])
        cadena_borrador = prompt_borrador | llm | StrOutputParser()
        return cadena_borrador.invoke({
            "plantilla": plantilla,
            "remitente": campos["remitente"],
            "asunto": campos["asunto"],
            "cuerpo": campos["cuerpo"],
        }).strip()
    return triage.borrador_respuesta


def procesar_bandeja(api_resource, tools, cadena_clasificacion, llm,
                      max_correos: int = 3, crear_borradores: bool = True,
                      agente_rl=None):
    tool_buscar = tools["search_gmail"]
    try:
        correos = tool_buscar.invoke({
            "query": "is:unread",
            "resource": "messages",
            "max_results": max_correos,
        })
    except HttpError as e:
        print(f"[ERROR] No se pudo buscar correos: {e}")
        return

    if not correos:
        print("No hay correos no leidos.")
        return

    resumen = []
    for msg in correos:
        campos = extraer_campos(msg)
        thread_id = msg.get("threadId")
        message_id = msg.get("id") or msg.get("messageId")

        try:
            triage = cadena_clasificacion.invoke(campos)
        except Exception as e:
            print(f"[ERROR] Clasificacion fallida para '{campos['asunto']}': {e}")
            continue

        # Refinamiento con RL (Nivel 3): el agente puede corregir la decision del LLM
        if agente_rl is not None:
            from triage_rl import TriageEnv
            estado_rl = TriageEnv.codificar_estado(
                triage.categoria, triage.urgencia,
                len(campos["cuerpo"]),
                tiene_enlaces="http" in campos["cuerpo"].lower() or "www." in campos["cuerpo"].lower(),
                tiene_adjuntos=False,
            )
            accion_rl = agente_rl.accion_optima(estado_rl)
        else:
            accion_rl = None

        # Si hay RL, usar su accion; si no, usar el enrutamiento del LLM
        if accion_rl is not None:
            from triage_rl import Accion as RLAction
            if accion_rl == RLAction.CREAR_BORRADOR:
                # Seguridad: aunque RL diga CREAR_BORRADOR, respetar si el LLM dice que no requiere respuesta
                if not triage.requiere_respuesta or not triage.borrador_respuesta.strip():
                    accion = "OMITIDO por RL (LLM indica que no requiere respuesta o borrador vacio)"
                elif hilo_tiene_borrador(api_resource, thread_id):
                    accion = "OMITIDO (el hilo ya tiene un borrador)"
                elif not crear_borradores:
                    accion = "SIMULADO (borrador no creado: pasada en seco)"
                else:
                    try:
                        cuerpo = redactar_borrador(llm, triage, campos)
                        crear_borrador_en_hilo(
                            api_resource,
                            to=campos["remitente"],
                            asunto=campos["asunto"],
                            cuerpo=cuerpo,
                            thread_id=thread_id,
                        )
                        accion = "BORRADOR creado en el hilo (RL)"
                    except (HttpError, ValueError) as e:
                        accion = f"ERROR al crear borrador: {e}"
            elif accion_rl == RLAction.MOVER_SPAM:
                if not crear_borradores or not message_id:
                    accion = f"SIMULADO (SPAM no aplicado)"
                else:
                    etiquetar_mensaje(api_resource, message_id, ["SPAM"])
                    accion = "ETIQUETADO como SPAM (RL)"
            else:  # MARCAR_REVISION o NADA
                if not crear_borradores or not message_id:
                    accion = f"SIMULADO (UNREAD no aplicado)"
                else:
                    etiquetar_mensaje(api_resource, message_id, ["UNREAD"])
                    accion = "ETIQUETADO como UNREAD (RL)"
        elif triage.categoria in CATEGORIAS_RESPONDER and triage.requiere_respuesta:
            if hilo_tiene_borrador(api_resource, thread_id):
                accion = "OMITIDO (el hilo ya tiene un borrador)"
            elif not crear_borradores:
                accion = "SIMULADO (borrador no creado: pasada en seco)"
            else:
                try:
                    cuerpo = redactar_borrador(llm, triage, campos)
                    crear_borrador_en_hilo(
                        api_resource,
                        to=campos["remitente"],
                        asunto=campos["asunto"],
                        cuerpo=cuerpo,
                        thread_id=thread_id,
                    )
                    accion = "BORRADOR creado en el hilo"
                except HttpError as e:
                    accion = f"ERROR al crear borrador: {e}"
        elif triage.categoria == "Spam o promocion":
            if not crear_borradores or not message_id:
                accion = f"SIMULADO{' (no dry-run)' if not crear_borradores else ''} (etiqueta SPAM{' ' if message_id else ' no '}aplicada)"
            else:
                etiquetar_mensaje(api_resource, message_id, ["SPAM"])
                accion = "ETIQUETADO como SPAM (movido a spam)"
        else:
            if not crear_borradores or not message_id:
                accion = f"SIMULADO{' (no dry-run)' if not crear_borradores else ''} (etiqueta UNREAD{' ' if message_id else ' no '}aplicada)"
            else:
                etiquetar_mensaje(api_resource, message_id, ["UNREAD"])
                accion = "ETIQUETADO como UNREAD (pendiente de revision)"

        print(f"  [{triage.categoria} / {triage.urgencia}] {campos['asunto'][:55]}")
        print(f"    Razonamiento: {triage.razonamiento[:120]}")
        print(f"    -> {accion}")
        resumen.append((campos["asunto"], triage.categoria, triage.urgencia, accion))

    print("\n" + "=" * 78)
    print("RESUMEN DEL TRIAGE")
    print("=" * 78)
    for asunto, categoria, urgencia, accion in resumen:
        print(f"- [{categoria} / {urgencia}] {asunto[:55]}")
        print(f"    -> {accion}")


def demo_offline(cadena_clasificacion):
    print("=" * 78)
    print("PRUEBA OFFLINE — 4 correos de ejemplo")
    print("=" * 78)
    for c in correos_demo:
        try:
            t = cadena_clasificacion.invoke(c)
        except Exception as e:
            print(f"[ERROR] Clasificacion fallida: {e}")
            continue
        print("-" * 70)
        print(f"  Asunto:         {c['asunto']}")
        print(f"  Categoria:      {t.categoria} | urgencia: {t.urgencia}")
        print(f"  Requiere resp.: {t.requiere_respuesta}")
        print(f"  Resumen:        {t.resumen}")
        print(f"  Razonamiento:   {t.razonamiento}")
        if t.borrador_respuesta:
            print(f"  Borrador:       {t.borrador_respuesta[:200]}...")


def main():
    parser = argparse.ArgumentParser(description="Triage automatico de correo con LangChain")
    parser.add_argument("--dry-run", action="store_true",
                        help="Clasifica y enruta pero NO crea borradores en Gmail")
    parser.add_argument("--demo", action="store_true",
                        help="Ejecuta solo la prueba offline (4 correos de ejemplo, sin Gmail)")
    parser.add_argument("--rag", action="store_true",
                        help="Activa RAG de plantillas de respuesta")
    parser.add_argument("--max-correos", type=int, default=3,
                        help="Numero maximo de correos no leidos a procesar (default: 3)")
    parser.add_argument("--rl", type=str, nargs="?", const="q_table_triage.npy",
                        help="Activa capa de refuerzo Q-Learning (post-procesamiento). "
                             "Opcional: ruta de la Q-table (default: q_table_triage.npy)")
    parser.add_argument("--rl-demo", action="store_true",
                        help="Entrena RL y muestra comparacion LLM vs LLM+RL sobre correos demo")
    args = parser.parse_args()

    global USE_RAG
    USE_RAG = args.rag

    print("[1/4] Inicializando LLM...")
    llm = init_llm()

    print("[2/4] Construyendo cadena de clasificacion (LCEL)...")
    cadena_clasificacion = init_chain(llm)

    if USE_RAG:
        print("[2b/4] Inicializando RAG...")
        init_rag()

    if args.rl_demo:
        from triage_rl import demo_offline_rl
        demo_offline_rl(llm, cadena_clasificacion)
        return

    agente_rl = None
    if args.rl:
        from triage_rl import QLearningAgent, N_ESTADOS, N_ACCIONES, TriageEnv
        ruta_q = args.rl
        if os.path.exists(ruta_q):
            agente_rl = QLearningAgent(N_ESTADOS, N_ACCIONES)
            agente_rl.cargar(ruta_q)
            print(f"[2c/4] Agente RL cargado desde {ruta_q}")
        else:
            print(f"[WARN] No se encontro {ruta_q}. Entrenando agente RL ahora...")
            from triage_rl import entrenar
            agente_rl, *resto = entrenar(
                llm=llm, cadena_clasificacion=cadena_clasificacion,
                episodios=300, n_por_clase=40, usar_llm=True,
                ruido_llm=0.0, guardar_q=ruta_q,
            )

    if args.demo:
        print("[3/4] Modo demo: clasificacion offline (sin Gmail)")
        if agente_rl:
            from triage_rl import demo_offline_rl
            demo_offline_rl(llm, cadena_clasificacion)
        else:
            demo_offline(cadena_clasificacion)
        return

    print("[3/4] Conectando a Gmail...")
    try:
        api_resource, tools = init_gmail()
    except FileNotFoundError:
        print("[ERROR] No se encontro credentials.json. Consulta el README.")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Error al conectar con Gmail: {e}")
        sys.exit(1)

    print(f"[4/4] Procesando bandeja (max {args.max_correos} correos, "
          f"crear_borradores={not args.dry_run})...")
    procesar_bandeja(
        api_resource, tools, cadena_clasificacion, llm,
        max_correos=args.max_correos,
        crear_borradores=not args.dry_run,
        agente_rl=agente_rl,
    )


if __name__ == "__main__":
    main()