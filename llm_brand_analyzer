"""
llm_brand_analyzer_streamlit.py
---------------------------------
SEO Brand Visibility Pipeline — Streamlit Web App

USAGE:
    streamlit run llm_brand_analyzer_streamlit.py

DEPENDENCIES:
    pip install streamlit openai pandas openpyxl
"""

import io
import time
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import streamlit as st
from openai import OpenAI, OpenAIError

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SEO Brand Visibility Pipeline",
    page_icon="⬡",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────────────────────
# LANGUAGE PROMPTS
# ──────────────────────────────────────────────────────────────────────────────

LANG_PROMPTS = {
    "English": {
        "gen_system":  ("Return ONLY a numbered list of exactly 3 natural user questions. "
                        "No preamble, no explanations. Write in English."),
        "gen_user":    "Generate 3 natural user questions based on this keyword: {keyword}",
        "resp_system": "You are a knowledgeable assistant. Answer thoroughly and naturally in English.",
        "sent_system": "Respond with EXACTLY one word: Positive, Neutral, or Negative.",
        "sent_user":   ("Classify the sentiment toward {brand} in this response "
                        "as Positive, Neutral, or Negative.\n\nResponse:\n{text}"),
        "why_system":  "You are an SEO and brand strategy analyst. Be concise. Write in English.",
        "why_user":    ("Analyze why {brand} was not mentioned in this response. "
                        "What factors could explain this?\n\nResponse:\n{text}"),
    },
    "Spanish": {
        "gen_system":  ("Devuelve ÚNICAMENTE una lista numerada de exactamente 3 preguntas "
                        "naturales de usuario. Sin preámbulo ni explicaciones. Escribe en español."),
        "gen_user":    "Genera 3 preguntas naturales de usuario basadas en esta palabra clave: {keyword}",
        "resp_system": "Eres un asistente experto. Responde de forma completa y natural en español.",
        "sent_system": "Responde con EXACTAMENTE una palabra: Positivo, Neutral o Negativo.",
        "sent_user":   ("Clasifica el sentimiento hacia {brand} en esta respuesta "
                        "como Positivo, Neutral o Negativo.\n\nRespuesta:\n{text}"),
        "why_system":  "Eres un analista de SEO y estrategia de marca. Sé conciso. Escribe en español.",
        "why_user":    ("Analiza por qué {brand} no fue mencionado en esta respuesta. "
                        "¿Qué factores podrían explicarlo?\n\nRespuesta:\n{text}"),
    },
    "French": {
        "gen_system":  ("Retourne UNIQUEMENT une liste numérotée de exactement 3 questions "
                        "naturelles d'utilisateur. Pas de préambule ni d'explications. Écris en français."),
        "gen_user":    "Génère 3 questions naturelles d'utilisateur basées sur ce mot-clé : {keyword}",
        "resp_system": "Tu es un assistant compétent. Réponds de manière complète et naturelle en français.",
        "sent_system": "Réponds avec EXACTEMENT un mot : Positif, Neutre ou Négatif.",
        "sent_user":   ("Classe le sentiment envers {brand} dans cette réponse "
                        "comme Positif, Neutre ou Négatif.\n\nRéponse:\n{text}"),
        "why_system":  "Tu es un analyste SEO et stratégie de marque. Sois concis. Écris en français.",
        "why_user":    ("Analyse pourquoi {brand} n'a pas été mentionné dans cette réponse. "
                        "Quels facteurs pourraient l'expliquer ?\n\nRéponse:\n{text}"),
    },
    "German": {
        "gen_system":  ("Gib NUR eine nummerierte Liste mit genau 3 natürlichen Benutzerfragen zurück. "
                        "Keine Einleitung, keine Erklärungen. Schreibe auf Deutsch."),
        "gen_user":    "Erstelle 3 natürliche Benutzerfragen basierend auf diesem Keyword: {keyword}",
        "resp_system": "Du bist ein kompetenter Assistent. Antworte ausführlich und natürlich auf Deutsch.",
        "sent_system": "Antworte mit GENAU einem Wort: Positiv, Neutral oder Negativ.",
        "sent_user":   ("Klassifiziere die Stimmung gegenüber {brand} in dieser Antwort "
                        "als Positiv, Neutral oder Negativ.\n\nAntwort:\n{text}"),
        "why_system":  "Du bist ein SEO- und Markenstrategie-Analyst. Sei prägnant. Schreibe auf Deutsch.",
        "why_user":    ("Analysiere, warum {brand} in dieser Antwort nicht erwähnt wurde. "
                        "Welche Faktoren könnten das erklären?\n\nAntwort:\n{text}"),
    },
}

SENTIMENT_POSITIVE = {"English": "Positive", "Spanish": "Positivo", "French": "Positif",  "German": "Positiv"}
SENTIMENT_NEUTRAL  = {"English": "Neutral",  "Spanish": "Neutral",  "French": "Neutre",   "German": "Neutral"}
SENTIMENT_NEGATIVE = {"English": "Negative", "Spanish": "Negativo", "French": "Négatif",  "German": "Negativ"}

# ──────────────────────────────────────────────────────────────────────────────
# OPENAI HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def make_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key.strip())


def _chat(client: OpenAI, system: str, user: str, model: str) -> str:
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user",   "content": user}],
            temperature=0.7,
        )
        return r.choices[0].message.content.strip()
    except OpenAIError as e:
        return f"[API ERROR: {e}]"


def generate_prompts_for_keyword(client, keyword, model, lang="English"):
    lp  = LANG_PROMPTS[lang]
    raw = _chat(client, lp["gen_system"], lp["gen_user"].format(keyword=keyword), model)
    lines     = [l.strip() for l in raw.splitlines() if l.strip()]
    questions = []
    for line in lines:
        for sep in (". ", ") ", " - "):
            if len(line) > 2 and line[0].isdigit() and line[1:3] == sep:
                line = line[3:].strip()
                break
        if line:
            questions.append(line)
    return (questions or lines)[:3]


def get_response(client, prompt, model, lang="English"):
    return _chat(client, LANG_PROMPTS[lang]["resp_system"], prompt, model)


def detect_brands(text, target, competitors):
    lo = text.lower()
    return (target.lower() in lo,
            [c for c in competitors if c.lower() in lo])


def analyze_sentiment(client, text, brand, model, lang="English"):
    lp = LANG_PROMPTS[lang]
    r  = _chat(client, lp["sent_system"],
               lp["sent_user"].format(brand=brand, text=text), model).strip()
    for canonical in (SENTIMENT_POSITIVE[lang], SENTIMENT_NEUTRAL[lang], SENTIMENT_NEGATIVE[lang]):
        if r.lower() == canonical.lower():
            return canonical
    return SENTIMENT_NEUTRAL[lang]


def analyze_why_missing(client, text, brand, model, lang="English"):
    lp = LANG_PROMPTS[lang]
    return _chat(client, lp["why_system"],
                 lp["why_user"].format(brand=brand, text=text), model)

# ──────────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ──────────────────────────────────────────────────────────────────────────────

def init_state():
    defaults = {
        "prompts_df":  None,
        "results_df":  None,
        "analysis_df": None,
        "log":         [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


def log(msg, level="info"):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    st.session_state["log"].append((ts, level, msg))

# ──────────────────────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────────────────────

st.markdown(
    "<h2 style='margin-bottom:0'>⬡ SEO Brand Visibility Pipeline</h2>"
    "<p style='color:grey;margin-top:2px'>Analyze how LLMs mention your brand across AI-generated responses</p>",
    unsafe_allow_html=True,
)
st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR — CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Configuration")

    # API Key
    st.subheader("🔑 OpenAI API Key")
    api_key = st.text_input("API Key", type="password", placeholder="sk-...",
                            help="Your OpenAI API key. Never stored anywhere.")

    st.subheader("🌐 Model & Language")
    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"])
    lang  = st.selectbox("Language", ["English", "Spanish", "French", "German"])

    st.subheader("⚡ Performance")
    workers = st.slider("Parallel workers", 1, 20, 5,
                        help="More workers = faster, but uses more API quota.")
    delay   = st.slider("Delay per worker (s)", 0.0, 5.0, 0.2, step=0.1)

    st.subheader("🔬 Analysis Options")
    do_sentiment = st.checkbox("Sentiment analysis", value=True,
                               help="When brand is present, classify tone.")
    do_why       = st.checkbox("Why-missing analysis", value=True,
                               help="When brand is absent, ask the model why.")

    st.subheader("💾 Export Format")
    export_fmt = st.radio("Format", ["CSV", "Excel (.xlsx)"], horizontal=True)

    st.divider()
    if st.button("↺ Restart (clear all data)", use_container_width=True, type="secondary"):
        st.session_state["prompts_df"]  = None
        st.session_state["results_df"]  = None
        st.session_state["analysis_df"] = None
        st.session_state["log"]         = []
        st.success("Workspace cleared.")
        st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────────────────────

tab_prompts, tab_results, tab_analysis, tab_log = st.tabs(
    ["✦ Step 1 — Prompts", "⬛ Step 2 — Results", "◈ Step 3 — Analysis", "◻ Log"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

with tab_prompts:
    st.markdown("### Generate or import prompts from your keywords")

    col_upload, col_settings = st.columns([2, 1])

    with col_upload:
        uploaded_kw = st.file_uploader(
            "Upload keywords file (CSV or Excel)",
            type=["csv", "xlsx", "xls"],
            help="Must have a column with your keywords (e.g. 'keyword').",
        )
        kw_col = st.text_input("Keyword column name", value="keyword")

    with col_settings:
        st.markdown("**Brand Settings**")
        target_brand  = st.text_input("Target brand (required)", value="Anthropic")
        competitors_raw = st.text_area(
            "Competitor brands (one per line, max 5)",
            value="OpenAI\nGoogle DeepMind\nMistral",
            height=120,
        )
        competitors = [c.strip() for c in competitors_raw.splitlines() if c.strip()][:5]

    st.divider()

    gen_col, imp_col = st.columns(2)

    with gen_col:
        st.markdown("**① Generate prompts from uploaded keywords**")
        if st.button("⚡ Generate Prompts", type="primary", use_container_width=True):
            if not api_key:
                st.error("Please enter your OpenAI API key in the sidebar.")
            elif uploaded_kw is None:
                st.error("Please upload a keywords file first.")
            else:
                try:
                    ext = uploaded_kw.name.rsplit(".", 1)[-1].lower()
                    df  = pd.read_csv(uploaded_kw) if ext == "csv" else pd.read_excel(uploaded_kw)
                    if kw_col not in df.columns:
                        st.error(f"Column '{kw_col}' not found. Available: {list(df.columns)}")
                    else:
                        df = df.dropna(subset=[kw_col])
                        df[kw_col] = df[kw_col].astype(str).str.strip()
                        df = df[df[kw_col] != ""].reset_index(drop=True)
                        keywords = df[kw_col].tolist()
                        client   = make_client(api_key)
                        records  = []

                        progress_bar = st.progress(0, text="Generating prompts…")
                        lock    = __import__("threading").Lock()
                        results = {}

                        def gen_task(idx, kw):
                            qs = generate_prompts_for_keyword(client, kw, model, lang)
                            time.sleep(delay)
                            with lock:
                                results[idx] = (kw, qs)

                        with ThreadPoolExecutor(max_workers=workers) as pool:
                            futures = {pool.submit(gen_task, i, kw): i
                                       for i, kw in enumerate(keywords)}
                            done = 0
                            for f in as_completed(futures):
                                try:
                                    f.result()
                                except Exception as e:
                                    log(f"Error: {e}", "error")
                                done += 1
                                progress_bar.progress(done / len(keywords),
                                                      text=f"Generating… {done}/{len(keywords)}")

                        for i in sorted(results):
                            kw, qs = results[i]
                            for q in qs:
                                records.append({kw_col: kw, "prompt": q})

                        st.session_state["prompts_df"] = pd.DataFrame(records)
                        log(f"✓ {len(records)} prompts generated from {len(keywords)} keywords.", "success")
                        progress_bar.empty()
                        st.success(f"✅ {len(records)} prompts generated!")
                        st.rerun()

                except Exception as e:
                    st.error(f"Error reading file: {e}")

    with imp_col:
        st.markdown("**Or import prompts directly from a file**")
        uploaded_prompts = st.file_uploader(
            "Upload prompts file (must have 'keyword' + 'prompt' columns)",
            type=["csv", "xlsx"],
            key="prompts_import",
        )
        if st.button("📥 Import Prompts", use_container_width=True):
            if uploaded_prompts is None:
                st.error("Please upload a prompts file first.")
            else:
                ext = uploaded_prompts.name.rsplit(".", 1)[-1].lower()
                df  = pd.read_csv(uploaded_prompts) if ext == "csv" else pd.read_excel(uploaded_prompts)
                if kw_col not in df.columns or "prompt" not in df.columns:
                    st.error(f"File must have '{kw_col}' and 'prompt' columns.")
                else:
                    st.session_state["prompts_df"] = df
                    log(f"Imported {len(df)} prompts.", "success")
                    st.success(f"✅ Imported {len(df)} prompts!")
                    st.rerun()

    # Show / edit prompts table
    if st.session_state["prompts_df"] is not None:
        st.divider()
        pdf = st.session_state["prompts_df"]
        st.markdown(f"**{len(pdf)} prompts ready** — you can edit them below before running Step 2.")
        edited = st.data_editor(
            pdf,
            use_container_width=True,
            num_rows="dynamic",
            key="prompts_editor",
        )
        if st.button("💾 Save edits"):
            st.session_state["prompts_df"] = edited
            st.success("Prompts updated.")

        # Export prompts
        buf = io.BytesIO()
        if export_fmt == "Excel (.xlsx)":
            edited.to_excel(buf, index=False)
            st.download_button("⬇️ Download Prompts (.xlsx)", buf.getvalue(),
                               file_name="seo_prompts.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            csv_bytes = edited.to_csv(index=False).encode()
            st.download_button("⬇️ Download Prompts (.csv)", csv_bytes,
                               file_name="seo_prompts.csv", mime="text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RESULTS
# ══════════════════════════════════════════════════════════════════════════════

with tab_results:
    st.markdown("### ② Fetch LLM Responses & Detect Brand Mentions")

    if st.session_state["prompts_df"] is None:
        st.info("⬅️ Generate or import prompts first (Step 1).")
    else:
        pdf = st.session_state["prompts_df"]
        st.write(f"Ready to process **{len(pdf)} prompts**. Brand to track: **{target_brand or '(not set)'}**")
        if competitors:
            st.write(f"Competitors to detect: {', '.join(competitors)}")

        if st.button("🚀 Fetch Responses & Detect Brands", type="primary", use_container_width=True):
            if not api_key:
                st.error("Please enter your OpenAI API key in the sidebar.")
            elif not target_brand:
                st.error("Please enter a target brand name in the Step 1 tab.")
            else:
                client  = make_client(api_key)
                rows    = pdf.to_dict("records")
                total   = len(rows)
                lock    = __import__("threading").Lock()
                results = {}

                progress_bar = st.progress(0, text="Fetching responses…")

                def resp_task(idx, rec):
                    kw     = rec.get(kw_col, "")
                    prompt = rec.get("prompt", "")
                    resp   = get_response(client, prompt, model, lang)
                    time.sleep(delay)
                    mentioned, comp_found = detect_brands(resp, target_brand, competitors)
                    entry = {
                        kw_col: kw,
                        "prompt": prompt,
                        "response": resp,
                        "target_brand_mentioned": mentioned,
                        "competitor_brands_detected": ", ".join(comp_found),
                        "sentiment": None,
                        "why_missing": None,
                    }
                    with lock:
                        results[idx] = entry

                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = {pool.submit(resp_task, i, r): i for i, r in enumerate(rows)}
                    done = 0
                    for f in as_completed(futures):
                        try:
                            f.result()
                        except Exception as e:
                            log(f"Error: {e}", "error")
                        done += 1
                        progress_bar.progress(done / total,
                                              text=f"Fetching… {done}/{total}")

                rdf = pd.DataFrame([results[i] for i in sorted(results)])
                st.session_state["results_df"] = rdf
                n_yes = int(rdf["target_brand_mentioned"].sum())
                n_no  = total - n_yes
                log(f"✓ Responses fetched. Mentioned: {n_yes} | Missing: {n_no}", "success")
                progress_bar.empty()
                st.success(f"✅ Done! **{n_yes}** mentions / **{n_no}** missing out of {total} responses.")
                st.rerun()

    # Show results table
    if st.session_state["results_df"] is not None:
        rdf = st.session_state["results_df"]
        st.divider()

        # Summary metrics
        n_yes = int(rdf["target_brand_mentioned"].sum())
        n_no  = len(rdf) - n_yes
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Responses", len(rdf))
        m2.metric("✅ Brand Mentioned", n_yes)
        m3.metric("❌ Brand Missing", n_no)

        st.markdown("**Results Table**")

        # Colour-coded display
        display_df = rdf[[kw_col, "prompt", "target_brand_mentioned",
                           "competitor_brands_detected"]].copy()
        display_df["target_brand_mentioned"] = display_df["target_brand_mentioned"].map(
            {True: "✓ Yes", False: "✗ No"})

        st.dataframe(display_df, use_container_width=True, height=300)

        # Row detail expander
        st.markdown("**Inspect full response**")
        selected_idx = st.number_input("Row index (0-based)", min_value=0,
                                       max_value=len(rdf) - 1, value=0, step=1)
        with st.expander("Show full response", expanded=False):
            st.write(rdf.iloc[selected_idx]["response"])

        # Export
        buf = io.BytesIO()
        export_df = rdf.copy()
        if export_fmt == "Excel (.xlsx)":
            export_df.to_excel(buf, index=False)
            st.download_button("⬇️ Download Results (.xlsx)", buf.getvalue(),
                               file_name="seo_results.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            st.download_button("⬇️ Download Results (.csv)",
                               export_df.to_csv(index=False).encode(),
                               file_name="seo_results.csv", mime="text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

with tab_analysis:
    st.markdown("### ③ Sentiment & Why-Missing Analysis")

    if st.session_state["results_df"] is None:
        st.info("⬅️ Fetch responses first (Step 2).")
    else:
        rdf   = st.session_state["results_df"]
        brand = target_brand

        if st.button("🔍 Run Analysis", type="primary", use_container_width=True):
            if not api_key:
                st.error("Please enter your OpenAI API key in the sidebar.")
            elif not brand:
                st.error("Please enter a target brand name in the Step 1 tab.")
            else:
                client  = make_client(api_key)
                rows    = rdf.to_dict("records")
                total   = len(rows)
                lock    = __import__("threading").Lock()
                results = {}

                progress_bar = st.progress(0, text="Analysing…")

                def analysis_task(idx, rec):
                    mentioned = rec["target_brand_mentioned"]
                    if mentioned and do_sentiment:
                        rec["sentiment"] = analyze_sentiment(
                            client, rec["response"], brand, model, lang)
                        time.sleep(delay)
                    if not mentioned and do_why:
                        rec["why_missing"] = analyze_why_missing(
                            client, rec["response"], brand, model, lang)
                        time.sleep(delay)
                    with lock:
                        results[idx] = rec

                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = {pool.submit(analysis_task, i, dict(r)): i
                               for i, r in enumerate(rows)}
                    done = 0
                    for f in as_completed(futures):
                        try:
                            f.result()
                        except Exception as e:
                            log(f"Error: {e}", "error")
                        done += 1
                        progress_bar.progress(done / total,
                                              text=f"Analysing… {done}/{total}")

                adf = pd.DataFrame([results[i] for i in sorted(results)])
                st.session_state["analysis_df"] = adf
                st.session_state["results_df"]  = adf.copy()
                log("✓ Analysis complete.", "success")
                progress_bar.empty()
                st.success("✅ Analysis complete!")
                st.rerun()

    # Show analysis results
    if st.session_state["analysis_df"] is not None:
        adf  = st.session_state["analysis_df"]
        lang_key = lang

        pos_val = SENTIMENT_POSITIVE[lang_key]
        neu_val = SENTIMENT_NEUTRAL[lang_key]
        neg_val = SENTIMENT_NEGATIVE[lang_key]

        n_pos = (adf["sentiment"] == pos_val).sum()
        n_neu = (adf["sentiment"] == neu_val).sum()
        n_neg = (adf["sentiment"] == neg_val).sum()

        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric(f"😊 {pos_val}", n_pos)
        m2.metric(f"😐 {neu_val}", n_neu)
        m3.metric(f"😠 {neg_val}", n_neg)

        display_cols = [kw_col, "prompt", "target_brand_mentioned", "sentiment", "why_missing"]
        available    = [c for c in display_cols if c in adf.columns]
        display_adf  = adf[available].copy()
        display_adf["target_brand_mentioned"] = display_adf["target_brand_mentioned"].map(
            {True: "✓ Yes", False: "✗ No"})

        st.dataframe(display_adf, use_container_width=True, height=350)

        # Row detail
        st.markdown("**Inspect full analysis**")
        sel_idx = st.number_input("Row index", min_value=0,
                                  max_value=len(adf) - 1, value=0, step=1,
                                  key="analysis_idx")
        row = adf.iloc[sel_idx]
        with st.expander("Why missing / sentiment detail", expanded=False):
            detail = row.get("why_missing") or row.get("sentiment") or "—"
            st.write(detail)

        # Export
        buf = io.BytesIO()
        if export_fmt == "Excel (.xlsx)":
            adf.to_excel(buf, index=False)
            st.download_button("⬇️ Download Full Analysis (.xlsx)", buf.getvalue(),
                               file_name="seo_analysis.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            st.download_button("⬇️ Download Full Analysis (.csv)",
                               adf.to_csv(index=False).encode(),
                               file_name="seo_analysis.csv", mime="text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — LOG
# ══════════════════════════════════════════════════════════════════════════════

with tab_log:
    st.markdown("### Activity Log")

    col_clear, _ = st.columns([1, 5])
    with col_clear:
        if st.button("🗑 Clear log"):
            st.session_state["log"] = []
            st.rerun()

    log_entries = st.session_state["log"]
    if not log_entries:
        st.caption("No activity yet.")
    else:
        LEVEL_COLORS = {
            "success": "🟢",
            "error":   "🔴",
            "warn":    "🟡",
            "accent":  "🔵",
            "info":    "⚪",
        }
        log_text = "\n".join(
            f"{LEVEL_COLORS.get(lvl, '⚪')} [{ts}]  {msg}"
            for ts, lvl, msg in reversed(log_entries)
        )
        st.code(log_text, language=None)
