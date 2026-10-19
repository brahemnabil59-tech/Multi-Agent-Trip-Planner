# 🧳 Trip Planner Multi-Agent

A multi-agent AI system that plans a trip: a team of 4 specialized agents works together to answer questions about a destination, its weather, and currency conversion.

## 👥 The Team

| Agent | Role | Tools |
|---|---|---|
| 🧠 **Supervisor** | Reads the question and team notes, decides who works next | none |
| 🔎 **Researcher** | Country facts and recent travel info | REST Countries, Tavily search |
| 📊 **Analyst** | Current weather and currency conversion | Weatherstack, exchange-rate API |
| ✍️ **Writer** | Combines the notes into the final answer | none |

## ✨ Features

- 🤖 **Supervisor pattern** built with LangGraph (`StateGraph`)
- 🛠️ Each specialist has its own tools and role prompt
- 📝 Shared state: agents pass notes to each other
- 👀 **Live view** of every agent's work in the UI
- 🛡️ Guards against loops: no agent repeats, max 4 supervisor decisions
- 🌐 Arabic & English
- 🔁 Switch between Gemini models from the sidebar

## 🔄 How It Works

1. The user asks a question.
2. The **Supervisor** picks the next agent.
3. The **Researcher** / **Analyst** run their tools and add notes to the shared state, then control returns to the Supervisor.
4. When enough info is collected, the Supervisor picks the **Writer**.
5. The **Writer** writes the final answer in the user's language.

## 🛠️ Tech Stack

Python · Streamlit · LangGraph · LangChain · Google Gemini · Tavily

## 🚀 Getting Started

**1. Clone the repo**
```bash
git clone https://github.com/brahemnabil59-tech/Multi-Agent-Trip-Planner.git
cd Multi-Agent-Trip-Planner
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add your API keys**

Create a `.env` file (see `.env.example`):
```env
GOOGLE_API_KEY=your_gemini_key
TAVILY_API_KEY=your_tavily_key
WEATHERSTACK_API_KEY=your_weatherstack_key
```

- Gemini: [Google AI Studio](https://aistudio.google.com/)
- Tavily: [tavily.com](https://tavily.com/)
- Weatherstack: [weatherstack.com](https://weatherstack.com/)

**4. Run**
```bash
streamlit run multi_agent_app.py
```

Open http://localhost:8501

## 💡 Example Questions

- `Plan a short trip to Turkey: facts, weather, and 300 USD in local currency`
- `عايز أسافر اليابان: إيه أهم معلومات عنها وطقسها، وكام 500 دولار بالين؟`
- `إيه عاصمة البرازيل وعملتها؟`

## ⚠️ Notes

A multi-agent run makes more LLM calls than a single agent (each agent has its own loop), so free-tier quotas are used faster.

## 📌 Related Projects

- [Capital & Weather App](https://github.com/brahemnabil59-tech/capital-weather-app): fixed LLM pipeline
- [Travel AI Agent](https://github.com/brahemnabil59-tech/Travel-AI-Agent): single tool-calling agent with memory

## 👤 Author

**Ebrahem Nabil** — AI Engineer
[GitHub](https://github.com/brahemnabil59-tech)
