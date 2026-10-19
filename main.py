import os
import operator
import certifi
import requests
import streamlit as st
from typing import Annotated, TypedDict
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent

# ==========================================
# ENV
# ==========================================
os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")
# TAVILY_API_KEY بيتقرا تلقائي من البيئة

MODELS = ["gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-3.1-flash-lite"]
SPECIALISTS = ["researcher", "analyst"]
ICONS = {"supervisor": "🧠", "researcher": "🔎", "analyst": "📊", "writer": "✍️"}


# ==========================================
# TOOLS
# ==========================================
@tool
def get_country_info(country: str) -> str:
    """Get basic info about a country: capital, population, region, currencies, languages.
    Use the English country name, e.g. 'Egypt'."""
    try:
        base = "https://restcountries.com/v3.1/name/" + country
        fields = "name,capital,population,region,currencies,languages"
        r = requests.get(base, params={"fields": fields, "fullText": "true"}, timeout=15)
        if r.status_code != 200:
            r = requests.get(base, params={"fields": fields}, timeout=15)
        if r.status_code != 200:
            return f"No data found for country: {country}"

        d = r.json()[0]
        currencies = ", ".join(
            f"{code} ({c.get('name', '')})" for code, c in d.get("currencies", {}).items()
        )
        languages = ", ".join(d.get("languages", {}).values())
        return (
            f"Country: {d['name']['common']}\n"
            f"Capital: {', '.join(d.get('capital', ['N/A']))}\n"
            f"Region: {d.get('region', 'N/A')}\n"
            f"Population: {d.get('population', 'N/A'):,}\n"
            f"Currencies: {currencies}\n"
            f"Languages: {languages}"
        )
    except Exception as e:
        return f"Error getting country info: {e}"


@tool
def get_weather(city: str) -> str:
    """Get the current weather of a city. Use the English city name, e.g. 'Cairo'."""
    try:
        r = requests.get(
            "https://api.weatherstack.com/current",
            params={"access_key": WEATHERSTACK_API_KEY, "query": city},
            timeout=15,
        ).json()
        if "current" not in r:
            return f"Could not fetch weather for {city}"
        cur = r["current"]
        return (
            f"City: {city}\n"
            f"Temperature: {cur['temperature']}°C\n"
            f"Weather: {cur['weather_descriptions'][0]}\n"
            f"Humidity: {cur['humidity']}%"
        )
    except Exception as e:
        return f"Error getting weather: {e}"


@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert an amount between two currencies using 3-letter codes, e.g. USD to EGP."""
    try:
        base, target = from_currency.upper(), to_currency.upper()
        r = requests.get(f"https://open.er-api.com/v6/latest/{base}", timeout=15).json()
        rate = r.get("rates", {}).get(target)
        if rate is None:
            return f"Could not find the exchange rate {base} -> {target}"
        return f"{amount} {base} = {amount * rate:,.2f} {target} (rate: {rate})"
    except Exception as e:
        return f"Error converting currency: {e}"


# ==========================================
# MULTI-AGENT GRAPH (supervisor pattern)
# ==========================================
class State(TypedDict):
    question: str
    notes: Annotated[list, operator.add]  # كل agent بيضيف ملاحظاته هنا
    next: str
    hops: int


def to_text(content):
    if isinstance(content, list):
        return "".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        )
    return content


def format_notes(notes):
    if not notes:
        return "(nothing yet)"
    return "\n\n".join(f"[{n['agent']}]\n{n['text']}" for n in notes)


@st.cache_resource
def build_graph(model_name: str):
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0,
        google_api_key=GOOGLE_API_KEY,
    )

    # ---------- Specialist agents (كل واحد بأدواته ودوره) ----------
    def make_specialist(name, tools, role):
        agent = create_react_agent(llm, tools, prompt=role)

        def node(state: State):
            msg = (
                f"User question: {state['question']}\n\n"
                f"Notes from teammates so far:\n{format_notes(state['notes'])}\n\n"
                "Do only your part using your tools, then report your findings briefly."
            )
            result = agent.invoke({"messages": [("user", msg)]})
            text = to_text(result["messages"][-1].content)
            return {"notes": [{"agent": name, "text": text}]}

        return node

    researcher = make_specialist(
        "researcher",
        [get_country_info, TavilySearch(max_results=3)],
        "You are the Researcher. Find country facts (capital, population, currency, "
        "languages) and relevant recent news or travel info. Use English names with tools. "
        "Report only facts you found.",
    )

    analyst = make_specialist(
        "analyst",
        [get_weather, convert_currency],
        "You are the Analyst. Get the current weather of the relevant city and do any "
        "currency conversion the user needs. Use the capital/currency found by the "
        "researcher if available. Use English names with tools. Report only numbers you got.",
    )

    # ---------- Writer (من غير أدوات، بيكتب الإجابة النهائية) ----------
    def writer(state: State):
        prompt = (
            "You are the Writer. Using ONLY the team's notes below, write the final answer "
            "to the user's question. Reply in the same language as the question "
            "(Arabic or English). Be clear, well organized, and concise.\n\n"
            f"Question: {state['question']}\n\nTeam notes:\n{format_notes(state['notes'])}"
        )
        text = to_text(llm.invoke(prompt).content)
        return {"notes": [{"agent": "writer", "text": text}]}

    # ---------- Supervisor (بيقرر مين يشتغل بعد كده) ----------
    def supervisor(state: State):
        done = {n["agent"] for n in state["notes"]}
        hops = state.get("hops", 0)

        prompt = (
            "You are a supervisor managing a team:\n"
            "- researcher: country facts, capital, language, web news\n"
            "- analyst: current weather and currency conversion\n"
            "- writer: writes the final answer (always LAST)\n\n"
            f"User question: {state['question']}\n\n"
            f"Team notes so far:\n{format_notes(state['notes'])}\n\n"
            f"Already done: {sorted(done) or 'none'}\n"
            "Pick who should work next. Call each specialist at most once, and only if "
            "the question needs them. When enough info is collected, pick writer.\n"
            "Reply with ONE word only: researcher, analyst, or writer."
        )
        text = to_text(llm.invoke(prompt).content).strip().lower()

        choice = next((w for w in ["researcher", "analyst", "writer"] if w in text), None)

        # حماية: لو الرد غلط أو agent اتكرر أو الحلقة طولت → اختار الأنسب
        if choice in SPECIALISTS and choice in done:
            choice = None
        if choice is None or hops >= 4:
            remaining = [a for a in SPECIALISTS if a not in done]
            choice = "writer" if (hops >= 4 or not remaining) else remaining[0]

        return {"next": choice, "hops": hops + 1}

    graph = StateGraph(State)
    graph.add_node("supervisor", supervisor)
    graph.add_node("researcher", researcher)
    graph.add_node("analyst", analyst)
    graph.add_node("writer", writer)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        lambda s: s["next"],
        {"researcher": "researcher", "analyst": "analyst", "writer": "writer"},
    )
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("analyst", "supervisor")
    graph.add_edge("writer", END)

    return graph.compile()


def run_team(graph, question: str, container):
    """Stream the team, show each agent's work live, return (final_answer, steps)."""
    steps, final = [], ""
    init = {"question": question, "notes": [], "next": "", "hops": 0}

    for update in graph.stream(init, stream_mode="updates"):
        for node, data in update.items():
            if node == "supervisor":
                text = f"يختار: `{data['next']}`"
            else:
                text = data["notes"][-1]["text"]
                if node == "writer":
                    final = text
            steps.append((node, text))
            with container:
                render_step(node, text)
    return final, steps


def render_step(agent: str, text: str):
    icon = ICONS.get(agent, "🤖")
    if agent == "supervisor":
        st.markdown(f"{icon} **Supervisor** → {text}")
    else:
        with st.container(border=True):
            st.markdown(f"{icon} **{agent.capitalize()}**")
            st.markdown(text)


# ==========================================
# UI
# ==========================================
st.set_page_config(page_title="Trip Planner Multi-Agent", page_icon="🧳", layout="centered")

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("⚙️ الإعدادات")
    model_name = st.selectbox("الموديل", MODELS)
    st.caption("لو ظهر 404 أو 429 غيّر الموديل.")

    if st.button("🗑️ مسح المحادثة", use_container_width=True):
        st.session_state.history = []
        st.rerun()

    st.divider()
    st.subheader("💡 جرّب")
    examples = [
        "عايز أسافر اليابان: إيه أهم معلومات عنها وطقسها، وكام 500 دولار بالين؟",
        "Plan a short trip to Turkey: facts, weather, and 300 USD in local currency",
        "إيه عاصمة البرازيل وعملتها؟",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.pending = ex

    st.divider()
    st.markdown("**الفريق:**")
    st.markdown("🧠 Supervisor\n\n🔎 Researcher\n\n📊 Analyst\n\n✍️ Writer")

st.title("🧳 Trip Planner Multi-Agent")
st.caption("فريق من 4 agents: المدير بيوزّع الشغل، والمتخصصين بينفذوا، والكاتب بيجمّع")

for item in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(item["question"])
    with st.chat_message("assistant"):
        with st.expander(f"👥 شغل الفريق ({len(item['steps'])} خطوة)"):
            for agent, text in item["steps"]:
                render_step(agent, text)
        st.markdown(item["final"])

prompt = st.chat_input("اسأل عن رحلة: دولة، طقس، عملة...")
if "pending" in st.session_state:
    prompt = st.session_state.pop("pending")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        graph = build_graph(model_name)
        try:
            with st.status("👥 الفريق شغال...", expanded=True) as status:
                final, steps = run_team(graph, prompt, status)
                status.update(label=f"✅ خلص ({len(steps)} خطوة)", state="complete")
            st.markdown(final)
            st.session_state.history.append(
                {"question": prompt, "steps": steps, "final": final}
            )
        except Exception as e:
            st.error(f"حصل خطأ: {e}")