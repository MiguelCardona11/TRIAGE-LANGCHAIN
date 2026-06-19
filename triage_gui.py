import streamlit as st
import sys
import io
import os
from contextlib import redirect_stdout, redirect_stderr

st.set_page_config(
    page_title="Triage LangChain",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "page" not in st.session_state:
    st.session_state.page = "inicio"


def capturar_output(func, *args, **kwargs):
    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        try:
            func(*args, **kwargs)
            ok = True
        except Exception as e:
            buf.write(f"\n[ERROR] {e}")
            ok = False
    return buf.getvalue(), ok


def init_llm_y_cadena():
    from triage_langchain import init_llm, init_chain
    llm = init_llm()
    cadena = init_chain(llm)
    return llm, cadena


COLOR_CATEGORIA = {
    "Urgente": "#dc2626",
    "Solicitud de informacion": "#2563eb",
    "Spam o promocion": "#ca8a04",
    "Otro": "#6b7280",
}

EMOJI_ACCION = {
    "CREAR_BORRADOR": "✏️",
    "MOVER_SPAM": "🗑️",
    "MARCAR_REVISION": "👁️",
    "NADA": "⏭️",
}


# ---------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------
st.sidebar.image(
    "https://cdn.jsdelivr.net/gh/anomalyco/opencode/assets/logo.svg",
    width=40,
)
st.sidebar.title("📧 Triage LangChain")
st.sidebar.caption("Clasificación inteligente de correos")

pagina = st.sidebar.radio(
    "Navegación",
    ["Inicio", "Clasificar", "Demo RL", "Entrenar RL"],
    index=["inicio", "clasificar", "demo_rl", "entrenar_rl"].index(
        st.session_state.page
    ),
    format_func=lambda x: {
        "Inicio": "🏠 Inicio",
        "Clasificar": "📨 Clasificar bandeja",
        "Demo RL": "🧠 Demo RL",
        "Entrenar RL": "📊 Entrenar RL",
    }[x],
)
st.session_state.page = {
    "Inicio": "inicio",
    "Clasificar": "clasificar",
    "Demo RL": "demo_rl",
    "Entrenar RL": "entrenar_rl",
}[pagina]

st.sidebar.divider()
st.sidebar.caption("Nivel 2 · LangChain + Groq")
st.sidebar.caption("Nivel 3 · Q-Learning")
st.sidebar.caption("Hecho con ❤️ para IA")

# ---------------------------------------------------------------
# PÁGINA: INICIO
# ---------------------------------------------------------------
if st.session_state.page == "inicio":
    st.title("📧 Sistema de Triage Automático de Correos")
    st.markdown(
        """
        Clasifica correos no leídos de Gmail en **4 categorías** usando
        **LangChain + Groq (Nivel 2)** y refina decisiones con
        **Q-Learning (Nivel 3)**.

        ---
        """
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            """
            <div style='border-left:4px solid #dc2626;padding:0.8rem;background:#fef2f2;border-radius:6px'>
            <h4 style='color:#dc2626;margin:0'>🔴 Urgente</h4>
            <small>Crea borrador de respuesta automático</small>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div style='border-left:4px solid #2563eb;padding:0.8rem;background:#eff6ff;border-radius:6px'>
            <h4 style='color:#2563eb;margin:0'>🔵 Solicitud</h4>
            <small>Crea borrador de respuesta automático</small>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div style='border-left:4px solid #ca8a04;padding:0.8rem;background:#fefce8;border-radius:6px'>
            <h4 style='color:#ca8a04;margin:0'>🟡 Spam</h4>
            <small>Mueve a carpeta SPAM</small>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            """
            <div style='border-left:4px solid #6b7280;padding:0.8rem;background:#f3f4f6;border-radius:6px'>
            <h4 style='color:#6b7280;margin:0'>⚪ Otro</h4>
            <small>Deja como UNREAD (revisión manual)</small>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("📐 Arquitectura")
        st.markdown(
            """
            ```
            Gmail → LLM (Groq) → Triage
                   ↓
            (opcional) RL → Acción final
                   ↓
            ✏️ Borrador | 🗑️ SPAM | 👁️ UNREAD
            ```
            """
        )
    with col_b:
        st.subheader("⚙️ Stack")
        st.markdown(
            """
            | Capa | Tecnología |
            |---|---|
            | LLM | `llama-3.3-70b-versatile` (Groq) |
            | Framework | LangChain ≥ 0.2 (LCEL) |
            | RL (N3) | Q-Learning tabular (144 estados) |
            | Gmail | GmailToolkit + API v1 |
            | UI | Streamlit |
            """
        )

    st.markdown("---")
    st.info("👈 Usa el menú lateral para navegar entre las secciones.")

# ---------------------------------------------------------------
# PÁGINA: CLASIFICAR
# ---------------------------------------------------------------
elif st.session_state.page == "clasificar":
    st.title("📨 Clasificar bandeja de Gmail")

    with st.sidebar:
        max_correos = st.slider("Máx. correos", 1, 20, 3, help="Número máximo de correos no leídos a procesar")
        dry_run = st.checkbox("Simulación (dry run)", value=False, help="No crea borradores ni modifica Gmail")
        usar_rl = st.checkbox("Usar RL (Q-Learning)", value=True if os.path.exists("q_table_triage.npy") else False)

    st.markdown("### ⚙️ Configuración aplicada")
    c1, c2, c3 = st.columns(3)
    c1.metric("Correos", max_correos)
    c2.metric("Dry run", "Sí" if dry_run else "No")
    c3.metric("RL", "Sí" if usar_rl else "No")

    if not os.path.exists("credentials.json"):
        st.error("No se encuentra `credentials.json` en el directorio. La conexión a Gmail fallará.")

    if st.button("🚀 Ejecutar triage", type="primary", use_container_width=True):
        output_placeholder = st.empty()
        status_placeholder = st.status("Ejecutando pipeline...", expanded=True)

        with status_placeholder:
            st.write("🔄 Inicializando LLM y cadena de clasificación...")
            try:
                llm, cadena = init_llm_y_cadena()
            except Exception as e:
                st.error(f"Error al inicializar LLM: {e}")
                st.stop()

            st.write("🔄 Conectando a Gmail...")
            try:
                from triage_langchain import init_gmail
                api_resource, tools = init_gmail()
            except FileNotFoundError:
                st.error("No se encontró credentials.json. Coloca el archivo en el directorio.")
                st.stop()
            except Exception as e:
                st.error(f"Error al conectar con Gmail: {e}")
                st.stop()

            agente_rl = None
            if usar_rl:
                st.write("🔄 Cargando agente RL...")
                if os.path.exists("q_table_triage.npy"):
                    from triage_rl import QLearningAgent, N_ESTADOS, N_ACCIONES
                    agente_rl = QLearningAgent(N_ESTADOS, N_ACCIONES)
                    agente_rl.cargar("q_table_triage.npy")
                    st.write("✅ Agente RL cargado desde `q_table_triage.npy`")
                else:
                    st.warning("No se encontró q_table_triage.npy. RL desactivado.")

            st.write("🔄 Procesando bandeja...")

        from triage_langchain import procesar_bandeja

        output, ok = capturar_output(
            procesar_bandeja,
            api_resource, tools, cadena, llm,
            max_correos=max_correos,
            crear_borradores=not dry_run,
            agente_rl=agente_rl,
        )

        status_placeholder.update(state="complete" if ok else "error", expanded=False)

        with st.expander("📋 Log completo", expanded=True):
            st.text(output)

        if not ok:
            st.error("El pipeline falló. Revisa el log.")
        else:
            st.success("Pipeline ejecutado exitosamente. Revisa el log para ver resultados detallados.")

# ---------------------------------------------------------------
# PÁGINA: DEMO RL
# ---------------------------------------------------------------
elif st.session_state.page == "demo_rl":
    st.title("🧠 Demo: LLM vs LLM + RL")

    st.markdown(
        """
        Entrena el agente Q-Learning y compara sus decisiones contra el LLM solo
        sobre los **4 correos demo** del proyecto.
        """
    )

    if st.button("🏃 Ejecutar demo RL", type="primary", use_container_width=True):
        status = st.status("Ejecutando demo RL...", expanded=True)

        with status:
            st.write("🔄 Inicializando LLM...")
            try:
                llm, cadena = init_llm_y_cadena()
            except Exception as e:
                st.error(f"Error al inicializar LLM: {e}")
                st.stop()

        from triage_rl import demo_offline_rl

        output, ok = capturar_output(demo_offline_rl, llm, cadena)

        status.update(state="complete" if ok else "error", expanded=False)

        col_log, col_plot = st.columns([3, 2])

        with col_log:
            with st.expander("📋 Log completo", expanded=True):
                st.text(output)

            if ok:
                resultados = []
                for linea in output.split("\n"):
                    if "Correo" in linea and ":" in linea:
                        resultados.append(linea.strip())
                    if "LLM ->" in linea:
                        resultados.append(linea.strip())
                    if "LLM solo" in linea:
                        resultados.append(linea.strip())
                    if "LLM + RL" in linea:
                        resultados.append(linea.strip())
                    if "Coinciden?" in linea:
                        resultados.append(linea.strip())

                if resultados:
                    st.markdown("### 📊 Resultados por correo")
                    for r in resultados:
                        st.markdown(f"- {r}")

        with col_plot:
            if os.path.exists("convergencia_rl.png"):
                st.image("convergencia_rl.png", caption="Curva de convergencia del entrenamiento RL")
            else:
                st.info("Gráfico de convergencia no encontrado. Se genera durante el entrenamiento.")

    else:
        if os.path.exists("convergencia_rl.png"):
            st.image("convergencia_rl.png", caption="Última curva de convergencia generada")

# ---------------------------------------------------------------
# PÁGINA: ENTRENAR RL
# ---------------------------------------------------------------
elif st.session_state.page == "entrenar_rl":
    st.title("📊 Entrenar agente Q-Learning")

    st.markdown(
        """
        Re-entrena el agente desde cero. Genera correos sintéticos, ejecuta
        Q-Learning por `n` episodios y guarda la Q-table en `q_table_triage.npy`.
        """
    )

    with st.sidebar:
        episodios = st.number_input("Episodios", min_value=50, max_value=2000, value=500, step=50)
        n_por_clase = st.number_input("Correos / categoría", min_value=10, max_value=200, value=60, step=10)
        ruido = st.slider("Ruido simulado del LLM", 0.0, 0.3, 0.05, 0.01, help="Probabilidad de que el LLM simulado se equivoque")

    if st.button("🎯 Entrenar agente", type="primary", use_container_width=True):
        from triage_rl import entrenar, generar_grafico, evaluar

        status = st.status("Entrenando agente RL...", expanded=True)
        with status:
            st.write(f"🔄 Entrenando {episodios} episodios con {n_por_clase*4} correos sintéticos...")
            st.write(f"   Ruido del LLM simulado: {ruido:.0%}")

        buf = io.StringIO()
        with redirect_stdout(buf), redirect_stderr(buf):
            try:
                agente, hist_reward, hist_acc, hist_acc_llm, datos_test = entrenar(
                    llm=None, cadena_clasificacion=None,
                    episodios=episodios, n_por_clase=n_por_clase,
                    usar_llm=False, ruido_llm=ruido,
                    guardar_q="q_table_triage.npy",
                )
                generar_grafico(hist_reward, hist_acc, hist_acc_llm)
                resultado_eval = evaluar(agente, datos_test, usando_llm=False)
                ok = True
            except Exception as e:
                buf.write(f"\n[ERROR] {e}")
                ok = False
        output = buf.getvalue()

        status.update(state="complete" if ok else "error", expanded=False)

        col1, col2 = st.columns(2)
        with col1:
            with st.expander("📋 Log de entrenamiento", expanded=True):
                st.text(output)
            if ok and hist_acc_llm:
                mejora = hist_acc[-1] - hist_acc_llm[-1]
                st.metric("Mejora vs LLM solo", f"+{mejora:.1%}")

        with col2:
            if ok:
                st.image("convergencia_rl.png", caption="Curva de convergencia")
                st.metric("Precisión en test", f"{resultado_eval['precision']:.1%}")
                st.metric("Aciertos", f"{resultado_eval['aciertos']}/{resultado_eval['total']}")

    elif os.path.exists("convergencia_rl.png"):
        st.image("convergencia_rl.png", caption="Última curva de convergencia")
        st.metric("Q-table en", "q_table_triage.npy" if os.path.exists("q_table_triage.npy") else "No encontrada")
