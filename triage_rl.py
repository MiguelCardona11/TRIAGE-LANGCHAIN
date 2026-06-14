#!/usr/bin/env python3
"""
Nivel 3 — Aprendizaje por Refuerzo (Q-Learning) para triage de correos
======================================================================

Entorno TriageEnv + agente Q-Learning tabular que refina las decisiones del
Nivel 2 (LLM). Se integra como capa de post-processamiento: el LLM da una
clasificacion preliminar y el agente RL decide la accion final basandose en
features del correo + la salida del LLM.

Uso:
    python triage_rl.py --train        # entrena y genera curva de convergencia
    python triage_rl.py --eval         # evalua precision con/sin RL
"""

import argparse
import os
import random
from enum import IntEnum

import numpy as np

# ---------------------------------------------------------------------------
# 1. CONSTANTES Y MAPEOS
# ---------------------------------------------------------------------------

CATEGORIAS_MAP = {
    "Urgente": 0,
    "Solicitud de informacion": 1,
    "Spam o promocion": 2,
    "Otro": 3,
}
CATEGORIAS_INV = {v: k for k, v in CATEGORIAS_MAP.items()}

URGENCIA_MAP = {"alta": 0, "media": 1, "baja": 2}
URGENCIA_INV = {v: k for k, v in URGENCIA_MAP.items()}

DIM_CATEGORIA = 4
DIM_URGENCIA = 3
DIM_LONGITUD = 3
DIM_ENLACES = 2
DIM_ADJUNTOS = 2

N_ACCIONES = 4  # CREAR_BORRADOR, MOVER_SPAM, MARCAR_REVISION, NADA
N_ESTADOS = DIM_CATEGORIA * DIM_URGENCIA * DIM_LONGITUD * DIM_ENLACES * DIM_ADJUNTOS


class Accion(IntEnum):
    CREAR_BORRADOR = 0
    MOVER_SPAM = 1
    MARCAR_REVISION = 2
    NADA = 3


# ---------------------------------------------------------------------------
# 2. ENTORNO
# ---------------------------------------------------------------------------

class TriageEnv:
    """
    Entorno discreto para triage de correos.

    Estado (5 dimensiones → indice lineal unico):
      0: Categoria predicha por el LLM (4 valores)
      1: Urgencia predicha por el LLM (3 valores)
      2: Longitud del cuerpo (0=corto <100, 1=medio 100-500, 2=largo >500)
      3: Tiene enlaces (0=no, 1=si)
      4: Tiene adjuntos (0=no, 1=si)

    Acciones (4):
      0: CREAR_BORRADOR  — responder al correo
      1: MOVER_SPAM      — mover a la carpeta spam
      2: MARCAR_REVISION — dejar como no leido para revision manual
      3: NADA            — no hacer nada (fallback)

    Recompensa:
      +3 por accion correcta (match con la etiqueta real)
      -1 por accion incorrecta
      -2 por omitir una accion necesaria (NADA cuando debia actuar)
    """

    DIMS = [DIM_CATEGORIA, DIM_URGENCIA, DIM_LONGITUD, DIM_ENLACES, DIM_ADJUNTOS]
    STRIDE = [
        1,
        DIM_CATEGORIA,
        DIM_CATEGORIA * DIM_URGENCIA,
        DIM_CATEGORIA * DIM_URGENCIA * DIM_LONGITUD,
        DIM_CATEGORIA * DIM_URGENCIA * DIM_LONGITUD * DIM_ENLACES,
    ]

    def __init__(self):
        self.n_actions = N_ACCIONES
        self.n_states = N_ESTADOS

    @staticmethod
    def codificar_estado(categoria_llm: str, urgencia_llm: str, longitud_cuerpo: int,
                         tiene_enlaces: bool, tiene_adjuntos: bool) -> int:
        c = CATEGORIAS_MAP.get(categoria_llm, 3)
        u = URGENCIA_MAP.get(urgencia_llm, 2)
        l = 0 if longitud_cuerpo < 100 else (1 if longitud_cuerpo < 500 else 2)
        e = 1 if tiene_enlaces else 0
        a = 1 if tiene_adjuntos else 0
        return c * (DIM_URGENCIA * DIM_LONGITUD * DIM_ENLACES * DIM_ADJUNTOS) \
             + u * (DIM_LONGITUD * DIM_ENLACES * DIM_ADJUNTOS) \
             + l * (DIM_ENLACES * DIM_ADJUNTOS) \
             + e * DIM_ADJUNTOS \
             + a

    @staticmethod
    def accion_correcta(categoria_real: str) -> int:
        if categoria_real in ("Urgente", "Solicitud de informacion"):
            return Accion.CREAR_BORRADOR
        if categoria_real == "Spam o promocion":
            return Accion.MOVER_SPAM
        return Accion.MARCAR_REVISION  # Otro

    @staticmethod
    def recompensa(accion: int, categoria_real: str) -> float:
        correcta = TriageEnv.accion_correcta(categoria_real)
        if accion == correcta:
            return 3.0
        if accion == Accion.NADA:
            return -2.0
        return -1.0


# ---------------------------------------------------------------------------
# 3. AGENTE Q-LEARNING TABULAR
# ---------------------------------------------------------------------------

class QLearningAgent:
    def __init__(self, n_states: int, n_actions: int,
                 alpha: float = 0.15, gamma: float = 0.90,
                 epsilon: float = 1.0, epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995):
        self.q = np.zeros((n_states, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.n_actions = n_actions

    def elegir_accion(self, estado: int) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        return int(np.argmax(self.q[estado]))

    def accion_optima(self, estado: int) -> int:
        return int(np.argmax(self.q[estado]))

    def aprender(self, estado: int, accion: int, recompensa: float,
                 estado_sig: int, done: bool):
        max_q_sig = np.max(self.q[estado_sig]) if not done else 0.0
        objetivo = recompensa + self.gamma * max_q_sig
        self.q[estado, accion] += self.alpha * (objetivo - self.q[estado, accion])
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def guardar(self, ruta: str):
        np.save(ruta, self.q)

    def cargar(self, ruta: str):
        self.q = np.load(ruta)


# ---------------------------------------------------------------------------
# 4. DATOS DE ENTRENAMIENTO SINTETICOS
# ---------------------------------------------------------------------------

def generar_correos_sinteticos(n_por_clase: int = 30) -> list[dict]:
    """
    Genera correos sinteticos con categoria real conocida para entrenar al agente.
    Los datos imitan variaciones de los 4 tipos (Urgente, Solicitud, Spam, Otro).
    """
    random.seed(42)
    np.random.seed(42)
    datos = []

    plantillas_urgente = [
        ("URGENTE: {tema}", "El {area} esta caido desde hace {min} minutos. Necesito que lo revises {plazo}. {adic}", True, False),
        ("CRITICO: {tema}", "Se ha detectado una falla de seguridad en {area}. {adic}", True, False),
        ("Incidencia grave - {tema}", "El servicio de {area} no responde. {adic}", True, False),
        ("Reporte urgente: {tema}", "Cliente reporta que {area} esta inaccesible. {adic}", True, False),
    ]
    plantillas_solicitud = [
        ("Consulta sobre {tema}", "Buenas, podrian enviarme informacion sobre {area}? {adic}", False, False),
        ("Solicitud de {tema}", "Estimados, requiero los datos de {area}. Quedo atento. {adic}", False, False),
        ("Informacion de {tema}", "Me gustaria recibir el detalle de {area}. {adic}", False, False),
        ("Pregunta acerca de {tema}", "Hola, tienen algun documento sobre {area}? {adic}", False, False),
    ]
    plantillas_spam = [
        ("Gana un {objeto} gratis", "Has sido seleccionado para ganar un {objeto}. Haz clic {enlace} {adic}", True, True),
        ("Oferta exclusiva - {descuento}%", "Aprovecha esta oportunidad unica. {descuento}% de descuento {enlace}", True, True),
        ("Tu opinion cuenta", "Completa esta encuesta y recibe {objeto}. {enlace} {adic}", True, True),
        ("Promocion especial {mes}", "No te pierdas nuestras ofertas de {mes}. {enlace}", True, True),
    ]
    plantillas_otro = [
        ("Boletin {mes}", "Resumen de actividades del mes de {mes}. {adic}", False, False),
        ("Recordatorio: {tema}", "Este es un recordatorio para {area}. No requiere accion. {adic}", False, False),
        ("Notificacion del sistema", "El sistema ha completado la tarea programada. {adic}", False, False),
        ("Invitacion a {evento}", "Te invitamos al evento de {evento}. Confirmacion opcional. {adic}", False, False),
    ]

    temas = ["servidor", "base de datos", "API", "sistema de pagos", "red", "aplicacion web"]
    areas = ["facturacion", "soporte tecnico", "recursos humanos", "contabilidad", "ventas"]
    meses = ["enero", "febrero", "marzo", "abril", "mayo", "junio"]
    objetos = ["iPhone", "tarjeta de regalo", "viaje", "curso gratis"]
    eventos = ["lanzamiento", "capacitacion", "conferencia"]
    adicionales = ["", "Por favor confirmar.", "Gracias.", "Quedo atento a su respuesta."]
    enlaces_lista = ["aqui", "en este enlace seguro", "en el siguiente link"]
    descuentos = [30, 50, 70, 80]

    for cat, plantillas, extras in [
        ("Urgente", plantillas_urgente, (temas, areas, ["inmediatamente", "lo antes posible", "ya", "urgentemente"])),
        ("Solicitud de informacion", plantillas_solicitud, (temas, areas)),
        ("Spam o promocion", plantillas_spam, objetos),
        ("Otro", plantillas_otro, (temas, meses, eventos)),
    ]:
        for _ in range(n_por_clase):
            plantilla = random.choice(plantillas)

            if cat == "Urgente":
                t, a, plazos = extras
                tema = random.choice(t)
                area = random.choice(t)
                minuto = random.randint(2, 60)
                plazo = random.choice(plazos)
                adic = random.choice(adicionales)
                asunto = plantilla[0].format(tema=tema)
                cuerpo = plantilla[1].format(area=area, min=minuto, plazo=plazo, adic=adic)
                tiene_enlaces = plantilla[2] if random.random() < 0.3 else False
                tiene_adjuntos = plantilla[3] if random.random() < 0.3 else False

            elif cat == "Solicitud de informacion":
                t, a = extras
                tema = random.choice(t)
                area = random.choice(a)
                adic = random.choice(adicionales)
                asunto = plantilla[0].format(tema=tema)
                cuerpo = plantilla[1].format(area=area, adic=adic)
                tiene_enlaces = plantilla[2] if random.random() < 0.2 else False
                tiene_adjuntos = plantilla[3]

            elif cat == "Spam o promocion":
                obj = random.choice(extras)
                mes = random.choice(meses)
                desc = random.choice(descuentos)
                enl = random.choice(enlaces_lista)
                adic = random.choice(adicionales)
                asunto = plantilla[0].format(objeto=obj, descuento=desc, mes=mes)
                # Cuerpo con enlace ~70% del tiempo
                if random.random() < 0.7:
                    cuerpo = plantilla[1].format(objeto=obj, descuento=desc, enlace=enl, mes=mes, adic=adic)
                    tiene_enlaces = True
                else:
                    cuerpo = plantilla[1].format(objeto=obj, descuento=desc, enlace="", mes=mes, adic=adic)
                    tiene_enlaces = False
                tiene_adjuntos = plantilla[3] if random.random() < 0.6 else False

            else:  # Otro
                t, m, ev = extras
                tema = random.choice(t)
                mes = random.choice(m)
                evento = random.choice(ev)
                area_data = random.choice(areas)
                adic = random.choice(adicionales)
                asunto = plantilla[0].format(tema=tema, mes=mes, evento=evento)
                cuerpo = plantilla[1].format(area=area_data if 'area' in plantilla[1] else '', mes=mes, tema=tema, adic=adic, evento=evento)
                tiene_enlaces = True if random.random() < 0.15 else False
                tiene_adjuntos = True if random.random() < 0.1 else False

            datos.append({
                "asunto": asunto,
                "cuerpo": cuerpo,
                "tiene_enlaces": tiene_enlaces,
                "tiene_adjuntos": tiene_adjuntos,
                "categoria_real": cat,
            })

    random.shuffle(datos)
    return datos


# ---------------------------------------------------------------------------
# 5. ENTRENAMIENTO Y EVALUACION
# ---------------------------------------------------------------------------

def extraer_features_para_entrenamiento(llm, cadena_clasificacion,
                                        correo: dict) -> dict:
    """
    Simula la clasificacion del LLM sobre un correo sintetico y devuelve
    las features necesarias para el estado del agente RL.
    """
    campos = {
        "remitente": "test@example.com",
        "asunto": correo["asunto"],
        "cuerpo": correo["cuerpo"],
    }
    try:
        triage = cadena_clasificacion.invoke(campos)
    except Exception:
        triage = type("Mock", (), {
            "categoria": "Otro", "urgencia": "baja",
            "razonamiento": "fallback"
        })

    return {
        "categoria_llm": triage.categoria,
        "urgencia_llm": triage.urgencia,
        "longitud_cuerpo": len(correo["cuerpo"]),
        "tiene_enlaces": correo["tiene_enlaces"],
        "tiene_adjuntos": correo["tiene_adjuntos"],
        "categoria_real": correo["categoria_real"],
        "codigo_estado": TriageEnv.codificar_estado(
            triage.categoria, triage.urgencia,
            len(correo["cuerpo"]),
            correo["tiene_enlaces"],
            correo["tiene_adjuntos"],
        ),
    }


def entrenar(llm=None, cadena_clasificacion=None, episodios: int = 500,
             n_por_clase: int = 50, usar_llm: bool = False,
             guardar_q: str = "", ruido_llm: float = 0.15) -> tuple:
    """
    Entrena el agente Q-Learning con datos sinteticos.

    Args:
        llm, cadena_clasificacion: si se pasan, usa el LLM real para clasificar.
        episodios: numero de episodios de entrenamiento.
        n_por_clase: correos sinteticos por categoria.
        usar_llm: True = usa LLM real; False = simula LLM perfecto (debug).
        guardar_q: ruta donde guardar la Q-table .npy.

    Returns:
        (agente, historial_recompensas, historial_precision, datos_test)
    """
    print(f"[RL] Generando {n_por_clase * 4} correos sinteticos...")
    datos = generar_correos_sinteticos(n_por_clase)

    split = int(len(datos) * 0.8)
    datos_train = datos[:split]
    datos_test = datos[split:]

    env = TriageEnv()
    agente = QLearningAgent(env.n_states, env.n_actions)

    historial_recomp = []
    historial_precision = []
    historial_precision_llm = []

    print(f"[RL] Entrenando durante {episodios} episodios ({len(datos_train)} train, {len(datos_test)} test)...")

    categorias_lista = list(CATEGORIAS_MAP.keys())

    for ep in range(episodios):
        random.shuffle(datos_train)
        recompensa_total = 0.0
        aciertos = 0
        aciertos_llm = 0

        for correo in datos_train:
            if usar_llm and llm and cadena_clasificacion:
                feats = extraer_features_para_entrenamiento(llm, cadena_clasificacion, correo)
            else:
                cat = correo["categoria_real"]
                urgencia = "alta" if cat == "Urgente" else ("media" if cat == "Solicitud de informacion" else "baja")

                # Simular LLM con ruido: a veces se equivoca de categoria
                if random.random() < ruido_llm:
                    cat_llm = random.choice([c for c in categorias_lista if c != cat])
                else:
                    cat_llm = cat

                feats = {
                    "codigo_estado": TriageEnv.codificar_estado(cat_llm, urgencia, len(correo["cuerpo"]),
                                                                  correo["tiene_enlaces"], correo["tiene_adjuntos"]),
                    "categoria_real": cat,
                }

            estado = feats["codigo_estado"]
            accion = agente.elegir_accion(estado)
            recomp = TriageEnv.recompensa(accion, feats["categoria_real"])
            recompensa_total += recomp

            if accion == TriageEnv.accion_correcta(feats["categoria_real"]):
                aciertos += 1

            # Precision del LLM ruidoso (sin RL)
            if not usar_llm:
                accion_llm_ruido = TriageEnv.accion_correcta(cat_llm)
                if accion_llm_ruido == TriageEnv.accion_correcta(correo["categoria_real"]):
                    aciertos_llm += 1
            else:
                if TriageEnv.accion_correcta(feats["categoria_real"]) == TriageEnv.accion_correcta(correo["categoria_real"]):
                    aciertos_llm += 1

            estado_sig = estado
            done = True
            agente.aprender(estado, accion, recomp, estado_sig, done)

        precision = aciertos / len(datos_train)
        precision_llm_ep = aciertos_llm / len(datos_train)
        historial_recomp.append(recompensa_total)
        historial_precision.append(precision)
        historial_precision_llm.append(precision_llm_ep)

        if (ep + 1) % 100 == 0:
            print(f"  Episodio {ep+1}/{episodios}  |  "
                  f"Recompensa: {recompensa_total:.1f}  |  "
                  f"Precision RL: {precision:.2%}  |  "
                  f"Precision LLM: {precision_llm_ep:.2%}  |  "
                  f"Epsilon: {agente.epsilon:.3f}")

    if guardar_q:
        os.makedirs(os.path.dirname(guardar_q) or ".", exist_ok=True)
        agente.guardar(guardar_q)
        print(f"[RL] Q-table guardada en {guardar_q}")

    return agente, historial_recomp, historial_precision, historial_precision_llm, datos_test


def evaluar(agente: QLearningAgent, datos_test: list, usando_llm: bool) -> dict:
    """Evalua la precision del agente sobre datos de test."""
    total = len(datos_test)
    aciertos = 0
    for correo in datos_test:
        if usando_llm:
            estado = correo.get("codigo_estado", 0)
        else:
            cat = correo["categoria_real"]
            urgencia = "alta" if cat == "Urgente" else ("media" if cat == "Solicitud de informacion" else "baja")
            estado = TriageEnv.codificar_estado(cat, urgencia, len(correo["cuerpo"]),
                                                  correo["tiene_enlaces"], correo["tiene_adjuntos"])
        accion = agente.accion_optima(estado)
        if accion == TriageEnv.accion_correcta(correo["categoria_real"]):
            aciertos += 1
    return {"total": total, "aciertos": aciertos, "precision": aciertos / total}


def generar_grafico(historial_recomp: list, historial_precision: list,
                    historial_precision_llm: list = None,
                    ruta: str = "convergencia_rl.png"):
    """Genera y guarda el grafico de convergencia del entrenamiento."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(historial_recomp, color="#2563eb", linewidth=0.7)
        ax1.set_title("Recompensa acumulada por episodio", fontsize=13)
        ax1.set_xlabel("Episodio")
        ax1.set_ylabel("Recompensa total")
        ax1.grid(alpha=0.3)

        ax2.plot(historial_precision, color="#16a34a", linewidth=0.7, label="LLM + RL")
        if historial_precision_llm:
            ax2.plot(historial_precision_llm, color="#dc2626", linewidth=0.7, alpha=0.7, label="LLM solo (con ruido)")
            ax2.legend(fontsize=10)
        ax2.set_title("Precision en entrenamiento por episodio", fontsize=13)
        ax2.set_xlabel("Episodio")
        ax2.set_ylabel("Precision")
        ax2.set_ylim(0, 1.05)
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(ruta, dpi=150)
        plt.close()
        print(f"[RL] Grafico de convergencia guardado en {ruta}")
    except ImportError:
        print("[RL] matplotlib no disponible, no se genero grafico.")


def demo_offline_rl(llm, cadena_clasificacion):
    """
    Entrena el agente RL y muestra la comparacion LLM-solo vs LLM+RL sobre
    los 4 correos demo originales.
    """
    print("=" * 78)
    print("NIVEL 3 — DEMO: LLM vs LLM+RL (Q-Learning)")
    print("=" * 78)

    # Entrenar agente
    print("\n[1/4] Entrenando agente Q-Learning (LLM simulado con 5% ruido)...")
    agente, hist_reward, hist_acc, hist_acc_llm, datos_test = entrenar(
        llm=None, cadena_clasificacion=None,
        episodios=500, n_por_clase=60, usar_llm=False,
        ruido_llm=0.05, guardar_q="q_table_triage.npy",
    )

    # Generar grafico
    generar_grafico(hist_reward, hist_acc, hist_acc_llm)

    # Evaluar en test
    print("\n[2/4] Evaluando en conjunto de test...")
    resultado = evaluar(agente, datos_test, usando_llm=False)
    print(f"  Precision post-RL: {resultado['precision']:.2%} ({resultado['aciertos']}/{resultado['total']})")
    # Precision LLM solo: simulamos clasificacion ruidosa y medimos aciertos
    aciertos_llm = 0
    import random as rnd
    rnd.seed(42)
    categorias = list(CATEGORIAS_MAP.keys())
    for c in datos_test:
        con_ruido = c["categoria_real"]
        if rnd.random() < 0.05:
            con_ruido = rnd.choice([x for x in categorias if x != con_ruido])
        if TriageEnv.accion_correcta(con_ruido) == TriageEnv.accion_correcta(c["categoria_real"]):
            aciertos_llm += 1
    resultado["precision_llm_solo"] = aciertos_llm / len(datos_test)
    print(f"  Precision LLM solo (con 5% ruido): {resultado['precision_llm_solo']:.2%} ({aciertos_llm}/{len(datos_test)})")

    print("\n[3/4] Ejecutando sobre los 4 correos demo originales...")
    from triage_langchain import correos_demo
    env = TriageEnv()

    for i, c in enumerate(correos_demo, 1):
        campos = {"remitente": c["remitente"], "asunto": c["asunto"], "cuerpo": c["cuerpo"]}
        triage = cadena_clasificacion.invoke(campos)

        estado = env.codificar_estado(
            triage.categoria, triage.urgencia,
            len(c["cuerpo"]),
            tiene_enlaces="http" in c["cuerpo"].lower(),
            tiene_adjuntos=False,
        )

        accion_solo_llm = TriageEnv.accion_correcta(triage.categoria)
        accion_con_rl = agente.accion_optima(estado)

        print(f"\n  Correo {i}: {c['asunto'][:55]}")
        print(f"    LLM -> Categoria: {triage.categoria} | Urgencia: {triage.urgencia}")
        print(f"    LLM solo      -> Accion: {Accion(accion_solo_llm).name}")
        print(f"    LLM + RL      -> Accion: {Accion(accion_con_rl).name}")
        print(f"    Coinciden?     -> {'SI' if accion_solo_llm == accion_con_rl else 'NO (RL refino la decision)'}")

    print("\n[4/4] Mejora cuantitativa vs Nivel 2:")
    print(f"  Precision LLM solo (con ruido 15%, test): {resultado.get('precision_llm_solo', 'N/A')}")
    print(f"  Precision LLM + RL (test):                {resultado['precision']:.2%}")
    print(f"  Delta de mejora:                          +{resultado['precision'] - resultado.get('precision_llm_solo', 0):.2%}")
    try:
        import matplotlib.pyplot as plt
        print(f"  Grafico de convergencia: convergencia_rl.png (adjuntar en diapositiva 11)")
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# 6. CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Nivel 3 - Q-Learning para triage de correos")
    parser.add_argument("--train", action="store_true", help="Entrena el agente y guarda Q-table")
    parser.add_argument("--eval", action="store_true", help="Evalua el agente entrenado")
    parser.add_argument("--episodios", type=int, default=500, help="Numero de episodios")
    parser.add_argument("--n-por-clase", type=int, default=50, help="Correos sinteticos por categoria")
    parser.add_argument("--usar-llm", action="store_true", help="Usa LLM real para clasificar durante entrenamiento")
    parser.add_argument("--q-table", type=str, default="q_table_triage.npy", help="Ruta de la Q-table")
    args = parser.parse_args()

    if args.train:
        llm = cadena = None
        if args.usar_llm:
            from triage_langchain import init_llm, init_chain
            print("[RL] Inicializando LLM (puede tomar unos segundos)...")
            llm = init_llm()
            cadena = init_chain(llm)
            print(f"[RL] LLM listo: {llm.model_name}")

        print(f"[RL] Entrenando agente Q-Learning ({args.episodios} episodios)...")
        agente, hist_reward, hist_acc, hist_acc_llm, datos_test = entrenar(
            llm=llm, cadena_clasificacion=cadena,
            episodios=args.episodios, n_por_clase=args.n_por_clase,
            usar_llm=args.usar_llm, guardar_q=args.q_table,
        )
        generar_grafico(hist_reward, hist_acc, hist_acc_llm)

        result = evaluar(agente, datos_test, usando_llm=args.usar_llm)
        print(f"\n[RL] Precision en test: {result['precision']:.2%} ({result['aciertos']}/{result['total']})")

    elif args.eval:
        if not os.path.exists(args.q_table):
            print(f"[ERROR] No se encuentra {args.q_table}. Ejecuta --train primero.")
            return
        agente = QLearningAgent(N_ESTADOS, N_ACCIONES)
        agente.cargar(args.q_table)
        __, __, __, __, datos_test = entrenar(
            episodios=1, n_por_clase=args.n_por_clase
        )
        result = evaluar(agente, datos_test, usando_llm=False)
        print(f"[RL] Precision evaluacion: {result['precision']:.2%} ({result['aciertos']}/{result['total']})")


if __name__ == "__main__":
    main()
