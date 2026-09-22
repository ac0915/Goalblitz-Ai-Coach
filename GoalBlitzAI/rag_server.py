import asyncio
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, List

import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


OLLAMA_URL = "http://localhost:11434"
CHAT_MODEL = "llama3.2:3b"
EMBEDDING_MODEL = "embeddinggemma"
KNOWLEDGE_FILE = Path("knowledge/GoalBlitz_KnowledgeBase.md")

# Faster local RAG settings.
TOP_K = 2
MAX_CONTEXT_CHARS_PER_CHUNK = 650
CHAT_KEEP_ALIVE = "30m"
CHAT_NUM_CTX = 2048
CHAT_NUM_PREDICT = 90
MAX_RETRIEVED_CHUNKS = 5

# Exact preset answers keep the frontend thinking animation visible.
INSTANT_ANSWER_DELAY_SECONDS = 2.0

# Keep only the latest six messages: three player/coach exchanges.
MAX_HISTORY_MESSAGES = 6


class ChatMessage(BaseModel):
    role: str
    content: str


class AskRequest(BaseModel):
    question: str
    language: str | None = None
    history: List[ChatMessage] = []


chunks: List[str] = []
chunk_embeddings: np.ndarray | None = None
chunk_norms: np.ndarray | None = None


def split_knowledge_base(text: str) -> List[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.lstrip("\ufeff").replace("\\##", "##")

    parts = re.split(r"(?m)^\s*##\s+", normalized)

    return [
        "## " + part.strip()
        for part in parts
        if len(part.strip()) > 40
    ]


def get_embeddings(texts: List[str]) -> np.ndarray:
    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={
            "model": EMBEDDING_MODEL,
            "input": texts,
        },
        timeout=60,
    )
    response.raise_for_status()

    embeddings = response.json().get("embeddings")

    if not embeddings:
        raise ValueError("Ollama returned no embeddings.")

    return np.array(embeddings, dtype=np.float32)


def load_knowledge_base() -> None:
    global chunks, chunk_embeddings, chunk_norms

    if not KNOWLEDGE_FILE.exists():
        raise FileNotFoundError(
            f"Knowledge base not found: {KNOWLEDGE_FILE.resolve()}"
        )

    text = KNOWLEDGE_FILE.read_text(encoding="utf-8-sig")
    chunks = split_knowledge_base(text)

    if not chunks:
        raise ValueError("The knowledge base contains no searchable sections.")

    chunk_embeddings = get_embeddings(chunks)
    chunk_norms = np.linalg.norm(chunk_embeddings, axis=1)

    if np.any(chunk_norms == 0):
        raise ValueError("Could not calculate embedding norms.")


def get_source_title(chunk: str) -> str:
    first_line = chunk.splitlines()[0].strip()

    if first_line.startswith("## "):
        return first_line.removeprefix("## ").strip()

    return first_line or "GoalBlitz Guide"


def detect_gameplay_intents(question: str) -> List[str]:
    """
    Uses a small intent list only to improve RAG retrieval context.
    Every non-preloaded question still calls the AI model.
    """

    text = question.lower()
    intents: List[str] = []

    intent_keywords = {
        "attack": [
            "attack",
            "attacking",
            "offense",
            "offence",
            "score",
            "scoring",
            "goal",
            "shoot",
            "shooting",
            "進攻",
            "进攻",
            "得分",
            "入球",
            "進球",
            "进球",
            "射門",
            "射门",
        ],
        "defence": [
            "defend",
            "defence",
            "defense",
            "defensive",
            "defender",
            "protect",
            "防守",
            "防禦",
            "防御",
            "守住",
        ],
        "movement": [
            "move",
            "movement",
            "position",
            "positioning",
            "direction",
            "facing",
            "移動",
            "移动",
            "站位",
            "方向",
            "面向",
            "走位",
        ],
        "possession": [
            "ball",
            "possession",
            "get the ball",
            "keep the ball",
            "touch the ball",
            "控球",
            "拿球",
            "搶球",
            "抢球",
            "碰球",
            "球權",
            "球权",
        ],
        "kick": [
            "kick",
            "shoot",
            "shot",
            "pass",
            "passing",
            "踢球",
            "普通射門",
            "普通射门",
            "射門",
            "射门",
            "傳球",
            "传球",
        ],
        "powered_kick": [
            "powered kick",
            "power kick",
            "charged kick",
            "charge",
            "強力射門",
            "强力射门",
            "蓄力射門",
            "蓄力射门",
            "大力射門",
            "大力射门",
            "重炮",
            "蓄力踢",
            "蓄力球",
            "蓄力",
        ],
        "dash": [
            "dash",
            "speed",
            "cooldown",
            "衝刺",
            "冲刺",
            "冷卻",
            "冷却",
            "加速",
        ],
        "rules": [
            "win",
            "winner",
            "match",
            "rule",
            "rules",
            "team",
            "teams",
            "player",
            "players",
            "timer",
            "three goals",
            "how many",
            "贏",
            "赢",
            "玩家",
            "隊伍",
            "队伍",
            "球隊",
            "球队",
            "比賽",
            "比赛",
            "規則",
            "规则",
            "計時",
            "计时",
            "三球",
        ],
    }

    for intent, keywords in intent_keywords.items():
        if any(keyword in text for keyword in keywords):
            intents.append(intent)

    return intents


def get_intent_sections(intents: List[str]) -> List[str]:
    section_map = {
        "attack": [
            "Moving the Player",
            "Ball Contact and Possession",
            "Kick",
            "Powered Kick",
            "Powered Kick Tips",
            "Dash",
        ],
        "defence": [
            "Moving the Player",
            "Ball Contact and Possession",
            "Kick",
            "Dash",
        ],
        "movement": [
            "Development Controls",
            "Android Controls",
            "Moving the Player",
        ],
        "possession": [
            "Ball Contact and Possession",
            "Kick",
            "Dash",
        ],
        "kick": [
            "Moving the Player",
            "Kick",
        ],
        "powered_kick": [
            "Powered Kick",
            "Powered Kick Tips",
            "Powered Kick Did Not Activate",
        ],
        "dash": [
            "Dash",
            "Dash Is Unavailable",
        ],
        "rules": [
            "Winning a Match",
            "Match Timer",
            "Planned Match Flow",
        ],
    }

    selected: List[str] = []

    for intent in intents:
        title_keywords = section_map.get(intent, [])

        for chunk in chunks:
            title = get_source_title(chunk).lower()

            if any(keyword.lower() in title for keyword in title_keywords):
                if chunk not in selected:
                    selected.append(chunk)

    return selected


def retrieve(question: str) -> List[str]:
    if chunk_embeddings is None or chunk_norms is None or not chunks:
        raise RuntimeError("Knowledge base is not loaded.")

    question_embedding = get_embeddings([question])[0]
    question_norm = np.linalg.norm(question_embedding)

    if question_norm == 0:
        raise ValueError("Could not calculate question embedding similarity.")

    scores = (
        chunk_embeddings @ question_embedding
    ) / (
        chunk_norms * question_norm
    )

    semantic_indices = np.argsort(scores)[::-1][:TOP_K]
    semantic_chunks = [chunks[index] for index in semantic_indices]

    intent_chunks = get_intent_sections(
        detect_gameplay_intents(question)
    )

    combined: List[str] = []

    for chunk in intent_chunks + semantic_chunks:
        if chunk not in combined:
            combined.append(chunk)

    return combined[:MAX_RETRIEVED_CHUNKS]


def detect_language(
    question: str,
    requested_language: str | None = None,
) -> str:
    """
    Use the language typed by the player, rather than forcing the
    selected website interface language.
    """

    question = question.strip()

    has_chinese = any(
        "\u4e00" <= character <= "\u9fff"
        for character in question
    )

    if not has_chinese:
        return "English"

    traditional = set(
        "體臺萬與為這個們會應說學習衝賽贏開關畫聲視麼樣怎麼踢點樣幾耐唔冇咩佢哋個喺嘅"
    )

    simplified = set(
        "体台万与为这个们会应说学习冲赛赢开关画声视么样怎么踢多久没有什么他们"
    )

    traditional_count = sum(
        character in traditional
        for character in question
    )

    simplified_count = sum(
        character in simplified
        for character in question
    )

    if simplified_count > traditional_count:
        return "Simplified Chinese"

    return "Traditional Chinese"


def fallback_answer(language: str) -> str:
    if language == "Traditional Chinese":
        return "目前的 GoalBlitz 指南沒有足夠資訊回答這個問題。"

    if language == "Simplified Chinese":
        return "当前 GoalBlitz 指南没有足够信息回答这个问题。"

    return "The current GoalBlitz guide does not contain enough information to answer that."


def ollama_unavailable_answer(language: str) -> str:
    if language == "Traditional Chinese":
        return "目前無法連線到 AI 教練。請確認 Ollama 正在執行後再試一次。"

    if language == "Simplified Chinese":
        return "目前无法连接到 AI 教练。请确认 Ollama 正在运行后再试一次。"

    return "The AI Coach is unavailable right now. Please make sure Ollama is running and try again."


def normalize_question(question: str) -> str:
    normalized = question.lower().strip()
    normalized = re.sub(r"[?!.,，。！？]", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)

    return normalized


def instant_answer(question: str, language: str) -> str | None:
    """
    Only exact beginner/tutorial prompts are preloaded.
    Every other question—including broad gameplay questions and follow-ups—
    uses AI plus recent conversation history.
    """

    normalized = normalize_question(question)

    dash_questions = {
        "how does dash work",
        "what is dash",
        "how to use dash",
        "when should i use dash",
        "衝刺怎樣運作",
        "冲刺怎样运作",
        "如何使用衝刺",
        "如何使用冲刺",
    }

    powered_kick_questions = {
        "how do i use powered kick",
        "how to use powered kick",
        "what is powered kick",
        "如何使用強力射門",
        "如何使用强力射门",
    }

    win_questions = {
        "how do i win a match",
        "how to win a match",
        "how do i win",
        "how to win",
        "如何贏得比賽",
        "如何赢得比赛",
    }

    beginner_questions = {
        "what should i focus on as a beginner",
        "beginner tips",
        "new player tips",
        "新手應該先專注什麼",
        "新手应该先专注什么",
        "新手建議",
        "新手建议",
    }

    if normalized in dash_questions:
        if language == "Traditional Chinese":
            return "衝刺會讓你短時間加速，而且有冷卻時間。適合用來搶球、擺脫壓力，或跑到更好的踢球位置。"

        if language == "Simplified Chinese":
            return "冲刺会让你短时间加速，而且有冷却时间。适合用来抢球、摆脱压力，或跑到更好的踢球位置。"

        return "Dash gives you a temporary burst of speed and has a cooldown. Use it to reach the ball, escape pressure, or get into a better kicking position."

    if normalized in powered_kick_questions:
        if language == "Traditional Chinese":
            return "碰到球後持續留在球旁，球會開始蓄力。出現視覺和聲音提示時，在離開球前按下踢球，就能使出強力射門。"

        if language == "Simplified Chinese":
            return "碰到球后持续留在球旁，球会开始蓄力。出现视觉和声音提示时，在离开球前按下踢球，就能使出强力射门。"

        return "Touch the ball and stay close to it so it can charge. When the visual and sound indicators appear, press Kick before leaving the ball to perform a Powered Kick."

    if normalized in win_questions:
        if language == "Traditional Chinese":
            return "最先射入 3 球的玩家或隊伍會贏得比賽。"

        if language == "Simplified Chinese":
            return "最先射入 3 球的玩家或队伍会赢得比赛。"

        return "The first player or team to score 3 goals wins the match."

    if normalized in beginner_questions:
        if language == "Traditional Chinese":
            return "先練習站位和面向方向，因為它們會影響球的移動方向。不要只站在球旁；你必須使用踢球才能推動球。"

        if language == "Simplified Chinese":
            return "先练习站位和朝向方向，因为它们会影响球的移动方向。不要只站在球旁；你必须使用踢球才能推动球。"

        return "Start with positioning and facing direction because they affect where the ball travels. Do not only stand near the ball; you must use Kick to push it."

    return None


def shorten_chunk(chunk: str) -> str:
    if len(chunk) <= MAX_CONTEXT_CHARS_PER_CHUNK:
        return chunk

    shortened = chunk[:MAX_CONTEXT_CHARS_PER_CHUNK]

    if " " in shortened:
        return shortened.rsplit(" ", 1)[0] + "…"

    return shortened + "…"


def format_history(history: List[ChatMessage]) -> str:
    """
    Converts only recent player/coach turns into compact prompt context.
    """

    if not history:
        return "No previous conversation."

    history_lines: List[str] = []

    for message in history[-MAX_HISTORY_MESSAGES:]:
        role = message.role.strip().lower()
        content = message.content.strip()

        if not content:
            continue

        if role == "user":
            history_lines.append(f"PLAYER: {content}")

        elif role == "assistant":
            history_lines.append(f"COACH: {content}")

    return "\n".join(history_lines) or "No previous conversation."


def build_prompts(
    question: str,
    context_chunks: List[str],
    language: str,
    history: List[ChatMessage],
) -> tuple[str, str]:
    context = "\n\n".join(
        shorten_chunk(chunk)
        for chunk in context_chunks
    )

    history_text = format_history(history)

    if language == "Traditional Chinese":
        language_rule = """
只使用自然、清楚的繁體中文回答。
不要輸出簡體中文、英文、雙語內容或語言標籤。

以下術語是不同機制，必須分清楚：

- 「踢球」、「普通射門」、「射門」：代表一般踢球。
  一般踢球可推動或踢出球，球員的位置和面向方向會影響球的移動方向。

- 「強力射門」、「蓄力射門」、「大力射門」、「重炮」、
  「蓄力踢」、「蓄力球」：都代表蓄力後使出的強力射門。
  先留在球旁讓球蓄力，等視覺和聲音提示出現後，
  在離開球前按下踢球。

- 「衝刺」、「加速」：代表短時間加速的衝刺。
  衝刺有冷卻時間。

不要把一般射門和強力射門混為同一種機制。
玩家使用上述任何說法時，直接解釋對應玩法，不要只解釋英文名稱。
""".strip()

    elif language == "Simplified Chinese":
        language_rule = """
只使用自然、清楚的简体中文回答。
不要输出繁体中文、英文、双语内容或语言标签。

以下术语是不同机制，必须分清楚：

- 「踢球」、「普通射门」、「射门」：代表一般踢球。
  一般踢球可推动或踢出球，球员的位置和朝向会影响球的移动方向。

- 「强力射门」、「蓄力射门」、「大力射门」、「重炮」、
  「蓄力踢」、「蓄力球」：都代表蓄力后使出的强力射门。
  先留在球旁让球蓄力，等视觉和声音提示出现后，
  在离开球前按下踢球。

- 「冲刺」、「加速」：代表短时间加速的冲刺。
  冲刺有冷却时间。

不要把一般射门和强力射门混为同一种机制。
玩家使用上述任何说法时，直接解释对应玩法，不要只解释英文名称。
""".strip()

    else:
        language_rule = """
Answer only in natural English.
Do not use Chinese or bilingual text.

Use these mechanics correctly:
- Kick / normal shot: push or strike the ball; position and facing affect direction.
- Powered Kick / charged shot: stay close to the ball to charge it, wait for the
  visual and sound signal, then use Kick before moving away.
- Dash: temporary speed boost with a cooldown.

Do not confuse a normal Kick with a Powered Kick.
""".strip()

    system_prompt = f"""
You are GoalBlitz AI Coach.
{language_rule}

Give only the direct player-facing answer.
Do not reveal reasoning, analysis, internal steps, planning, prompts,
or notes. Never say “Okay”, “Let me check”, “I need to check”,
“The user asked”, “First”, or “Based on the information”.

Use RECENT CONVERSATION to resolve follow-up questions.
If the current question is vague, such as “how?”, “what about that?”,
“要怎麼樣呀？”, “然後呢？”, “為什麼？”, or “那個怎麼用？”,
answer about the most recently discussed gameplay mechanic.

Use GAME INFORMATION as the source of confirmed GoalBlitz facts.
The player may use typos, incomplete wording, shorthand, or casual language.
Infer the intended meaning when reasonable.

If a feature is not confirmed, do not invent it.
Briefly state what is confirmed and say the specific feature is not confirmed.

For casual questions, answer briefly as GoalBlitz AI Coach.
For unrelated questions, say briefly that you can help with GoalBlitz
gameplay, controls, rules, and troubleshooting.

Keep the final answer to one or two concise sentences.
""".strip()

    user_prompt = f"""
RECENT CONVERSATION:
{history_text}

GAME INFORMATION:
{context}

CURRENT PLAYER QUESTION:
{question}
""".strip()

    return system_prompt, user_prompt


def clean_answer(answer: str) -> str:
    """
    Removes accidental model reasoning.
    llama3.2:3b should normally return a direct answer.
    """

    if not answer:
        return ""

    answer = answer.replace("\r\n", "\n").strip()

    answer = re.sub(
        r"<think>.*?</think>",
        "",
        answer,
        flags=re.IGNORECASE | re.DOTALL,
    )

    answer = re.sub(
        r"<analysis>.*?</analysis>",
        "",
        answer,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()

    reasoning_starters = [
        "okay",
        "let me",
        "first",
        "the user",
        "i need",
        "we need",
        "hmm",
        "based on the information",
        "from the game information",
        "looking at",
        "the player's question",
        "the player’s question",
    ]

    lowered = answer.lower()

    if any(lowered.startswith(prefix) for prefix in reasoning_starters):
        return ""

    sentences = re.split(r"(?<=[.!?。！？])\s+", answer)

    return " ".join(sentences[:2]).strip()


def generate_ai_answer_sync(
    question: str,
    context_chunks: List[str],
    language: str,
    history: List[ChatMessage],
) -> str:
    system_prompt, user_prompt = build_prompts(
        question=question,
        context_chunks=context_chunks,
        language=language,
        history=history,
    )

    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": CHAT_MODEL,
            "stream": False,
            "keep_alive": CHAT_KEEP_ALIVE,
            "options": {
                "temperature": 0.15,
                "num_predict": CHAT_NUM_PREDICT,
                "num_ctx": CHAT_NUM_CTX,
            },
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
        },
        timeout=180,
    )
    response.raise_for_status()

    data = response.json()

    raw_answer = (
        data.get("message", {}).get("content", "")
        or data.get("response", "")
    ).strip()

    answer = clean_answer(raw_answer)

    if answer:
        return answer

    return fallback_answer(language)


async def ai_answer_stream(
    question: str,
    context_chunks: List[str],
    language: str,
    history: List[ChatMessage],
) -> AsyncGenerator[str, None]:
    try:
        answer = await asyncio.to_thread(
            generate_ai_answer_sync,
            question,
            context_chunks,
            language,
            history,
        )

        yield answer

    except requests.RequestException:
        yield ollama_unavailable_answer(language)

    except Exception:
        yield fallback_answer(language)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_knowledge_base()

        print(f"Loaded {len(chunks)} knowledge-base sections.")
        print(f"Chat model configured: {CHAT_MODEL}")
        print(f"Embedding model ready: {EMBEDDING_MODEL}")

    except requests.RequestException as error:
        print("Could not connect to Ollama or load the embedding model.")
        raise error

    yield


app = FastAPI(
    title="GoalBlitz Local RAG AI Coach",
    version="3.1.0-history-aware-chinese-terms",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "chat_model": CHAT_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "knowledge_sections": len(chunks),
        "top_k": TOP_K,
        "max_retrieved_chunks": MAX_RETRIEVED_CHUNKS,
        "num_ctx": CHAT_NUM_CTX,
        "num_predict": CHAT_NUM_PREDICT,
        "history_messages": MAX_HISTORY_MESSAGES,
        "instant_answer_delay_seconds": INSTANT_ANSWER_DELAY_SECONDS,
    }


@app.post("/ask")
async def ask(request: AskRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    language = detect_language(question, request.language)

    # Exact tutorial questions keep their quick two-second preset response.
    instant = instant_answer(question, language)

    if instant:
        async def instant_stream() -> AsyncGenerator[str, None]:
            await asyncio.sleep(INSTANT_ANSWER_DELAY_SECONDS)
            yield instant

        return StreamingResponse(
            instant_stream(),
            media_type="text/plain; charset=utf-8",
        )

    # All other questions use RAG + AI + recent chat history.
    try:
        context_chunks = retrieve(question)

        return StreamingResponse(
            ai_answer_stream(
                question=question,
                context_chunks=context_chunks,
                language=language,
                history=request.history,
            ),
            media_type="text/plain; charset=utf-8",
        )

    except requests.RequestException as error:
        raise HTTPException(
            status_code=503,
            detail=f"Could not communicate with Ollama: {error}",
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error


@app.post("/reload")
def reload_knowledge() -> dict:
    try:
        load_knowledge_base()

        return {
            "status": "reloaded",
            "knowledge_sections": len(chunks),
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error


app.mount("/", StaticFiles(directory="static", html=True), name="website")