"""A customer service agent for Wei-Restaurant (Gradio app with RAG & Tool Calling).

Every question goes to the agent first. The agent decides which tool to use:
- Menu facts and prices come from SQLite & python code.
- Shop rules and policies come from a RAG search over shop_rules.txt.
- Anything else goes to a human staff.

Run it with: python app_gradio_v1_3.py
"""

import csv
import json
import os
import re
from pathlib import Path
import sqlite3

import gradio as gr
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# 1. API setup
# ---------------------------------------------------------------------------
load_dotenv(Path(__file__).parent / ".env")
CHAT_MODEL = os.getenv("MODEL_NAME")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-004")

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("LLM_API_URL"),
    timeout=60.0
)

embed_client = OpenAI(
    api_key=os.getenv("EMBEDDING_API_KEY") or os.getenv("API_KEY"),
    base_url=os.getenv("EMBEDDING_API_URL") or os.getenv("LLM_API_URL"),
    timeout=60.0
)


# ---------------------------------------------------------------------------
# 2. SQLite Database & RAG Text Indexing
# ---------------------------------------------------------------------------
connection = sqlite3.connect("shop_app.db", check_same_thread=False, timeout=10)
connection.row_factory = sqlite3.Row
cursor = connection.cursor()

cursor.execute("DROP TABLE IF EXISTS shop_info")
cursor.execute("CREATE TABLE shop_info (topic TEXT PRIMARY KEY, content TEXT NOT NULL)")
cursor.execute("DROP TABLE IF EXISTS menu")
cursor.execute("""CREATE TABLE menu (item_id INTEGER PRIMARY KEY, name TEXT, name_zh TEXT,
                  category TEXT, price INTEGER, available INTEGER)""")

with open("shop_info.csv", newline="", encoding="utf-8") as f:
    cursor.executemany("INSERT INTO shop_info VALUES (?, ?)",
                       [(r["topic"], r["content"]) for r in csv.DictReader(f)])

with open("menu.csv", newline="", encoding="utf-8") as f:
    cursor.executemany("INSERT INTO menu VALUES (?, ?, ?, ?, ?, ?)",
                       [(r["item_id"], r["name"], r["name_zh"], r["category"], r["price"], r["available"])
                        for r in csv.DictReader(f)])
connection.commit()
SHOP_TOPICS = [r["topic"] for r in cursor.execute("SELECT topic FROM shop_info")]


def embed(texts):
    """Turn a list of texts into vectors using the configured embedding API."""
    response = embed_client.embeddings.create(model=EMBED_MODEL, input=texts)
    return np.array([item.embedding for item in response.data], dtype="float32")


# 讀取餐廳規範文字檔 (RAG 知識庫)
# 內建雙語標題 (title 與 title_en) 供多國語言切換
rules_file = Path(__file__).parent / "shop_rules.txt"
if rules_file.exists():
    with open(rules_file, encoding="utf-8") as f:
        CHUNKS = []
        for part in f.read().split("\n## ")[1:]:
            title, _, text = part.partition("\n")
            CHUNKS.append({
                "title": title.strip(), 
                "title_en": title.strip(), 
                "text": text.strip()
            })
else:
    CHUNKS = [
        {
            "title": "營業時間與點餐",
            "title_en": "Operating Hours & Orders",
            "text": "每日 10:30 至 21:00 營業。午餐 10:30-14:30，晚餐 17:00-20:30。最後點餐時間為 20:30。"
        },
        {
            "title": "訂位與排隊",
            "title_en": "Reservations & Queue",
            "text": "現場候位需至櫃檯領取號碼牌。6人以上團體聚餐需提前至少一天電話預約。"
        },
        {
            "title": "攜帶寵物",
            "title_en": "Pet Policy",
            "text": "歡迎攜帶寵物，但需放入寵物提籃或推車內，全程不可露出頭部，且不可落地。"
        },
        {
            "title": "外食與開瓶費",
            "title_en": "Outside Food & Corkage Fee",
            "text": "禁帶外食飲料（嬰兒副食品除外）。自行攜帶酒類開瓶費為每瓶 NT$300。"
        },
        {
            "title": "停車優惠",
            "title_en": "Parking Discount",
            "text": "餐廳無專用停車場。消費滿 NT$1000 可折抵特約停車場（大學路100號後方）2小時停車費。"
        },
        {
            "title": "設施與 Wi-Fi",
            "title_en": "Facilities & Wi-Fi",
            "text": "提供免費 Wi-Fi（密碼 wei2026888）、兒童座椅與無障礙坡道。"
        },
        {
            "title": "折扣與壽星優惠",
            "title_en": "Discounts & Birthday Offer",
            "text": "憑學生證享 9 折優惠。當月壽星點主餐即贈送精緻手工甜點乙份。"
        }
    ]

CHUNK_VECTORS = embed([f"{s['title']}\n{s['text']}" for s in CHUNKS])
MIN_SCORE = 0.40

print(f"Ready: {len(SHOP_TOPICS)} shop topics, {len(CHUNKS)} RAG rule chunks.")


# ---------------------------------------------------------------------------
# 3. Tool Definitions
# ---------------------------------------------------------------------------
def get_shop_info(topic):
    """Return one basic piece of shop information from SQLite."""
    row = cursor.execute("SELECT content FROM shop_info WHERE topic = ?", (topic,)).fetchone()
    if row is None:
        return {"found": False, "topics_available": SHOP_TOPICS}
    return {"found": True, "topic": topic, "content": row["content"]}


def get_price(item_name):
    """Find menu items by English or Chinese name, returning prices and availability."""
    like = f"%{item_name.strip()}%"
    rows = cursor.execute("SELECT name, name_zh, price, available FROM menu "
                          "WHERE name LIKE ? OR name_zh LIKE ?", (like, like)).fetchall()
    if not rows:
        return {"found": False, "note": f"'{item_name}' is not on the menu."}
    return {"found": True, "items": [{"name": r["name"], "name_zh": r["name_zh"], "price": r["price"],
                                      "available": bool(r["available"])} for r in rows]}


def calculate_total(items):
    """Add up an order bill."""
    lines, total = [], 0
    for entry in items:
        found = get_price(entry["name"])
        if not found["found"] or len(found["items"]) != 1:
            return {"total": None, "note": f"Cannot identify '{entry['name']}'. Please check item name."}
        item = found["items"][0]
        lines.append(f"{entry['quantity']} x {item['name']} ({item['name_zh']}) = NT${item['price'] * entry['quantity']}")
        total += item["price"] * entry["quantity"]
    return {"lines": lines, "total": total}


def search_shop_rules(question, k=2):
    """Perform RAG vector search to retrieve relevant shop rules and policies."""
    q = embed([question])[0]
    scores = CHUNK_VECTORS @ q / (np.linalg.norm(CHUNK_VECTORS, axis=1) * np.linalg.norm(q))
    best = [i for i in np.argsort(scores)[::-1][:k] if scores[i] >= MIN_SCORE]
    if not best:
        return {"found": False, "note": "The shop rules do not cover this question."}
    return {
        "found": True, 
        "chunks": [
            {
                "title": CHUNKS[i]["title"],
                "title_en": CHUNKS[i]["title_en"],
                "text": CHUNKS[i]["text"],
                "score": round(float(scores[i]), 3)
            } for i in best
        ]
    }


def transfer_to_human():
    """Transfer complex customer inquiries or complaints to human staff."""
    return {"message": "Please call our staff at 07-123-4567, every day from 10:30 to 21:00.",
            "note": ("First tell the customer why in one sentence, then give this message. "
                     "Do not call this tool again.")}


TOOL_FUNCTIONS = {
    "get_shop_info": get_shop_info,
    "get_price": get_price,
    "calculate_total": calculate_total,
    "search_shop_rules": search_shop_rules,
    "transfer_to_human": transfer_to_human,
}

TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "get_shop_info",
        "description": ("Get basic shop information: opening hours, address, phone, payment options, "
                        "combo deals, or delivery."),
        "parameters": {"type": "object",
                       "properties": {"topic": {"type": "string", "enum": SHOP_TOPICS}},
                       "required": ["topic"]}}},
    {"type": "function", "function": {
        "name": "get_price",
        "description": "Get price and availability of a menu item (e.g. 'Ribeye Steak', '肋眼牛排').",
        "parameters": {"type": "object",
                       "properties": {"item_name": {"type": "string"}},
                       "required": ["item_name"]}}},
    {"type": "function", "function": {
        "name": "calculate_total",
        "description": "Add up an order of several items. Always use this tool to calculate bills.",
        "parameters": {"type": "object",
                       "properties": {"items": {"type": "array", "items": {
                           "type": "object",
                           "properties": {"name": {"type": "string"}, "quantity": {"type": "integer"}},
                           "required": ["name", "quantity"]}}},
                       "required": ["items"]}}},
    {"type": "function", "function": {
        "name": "search_shop_rules",
        "description": ("Search detailed shop rules & policies: pet policy, parking validation, "
                        "birthday discounts, corkage fees, Wi-Fi, deposits, and booking rules."),
        "parameters": {"type": "object",
                       "properties": {"question": {"type": "string"}},
                       "required": ["question"]}}},
    {"type": "function", "function": {
        "name": "transfer_to_human",
        "description": "Use for complaints, manager requests, or questions unanswerable by other tools.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
]

SYSTEM_PROMPT = """
You are the customer service agent of Wei-Restaurant, University Road branch (an Italian & European restaurant).
All prices are in New Taiwan dollars, written like NT$260.

RULES
1. Answer only from tool results. Never answer from memory, and never invent anything.
2. Menu prices and availability -> get_price.
3. Order totals -> calculate_total.
4. Basic shop info (hours, address, phone) -> get_shop_info.
5. Shop policies & rules (pets, parking, discounts, Wi-Fi, corkage, bookings) -> search_shop_rules.
6. Complaints, manager requests, or unhandled topics -> transfer_to_human.
7. Reply politely in three sentences or less. Match the language of the user (English or Traditional Chinese).
"""

MAX_STEPS = 6


def run_agent(chat_history):
    """Answer the last user question using OpenAI Tool Calling + RAG."""
    question_content = chat_history[-1]["content"]
    question_str = str(question_content) if not isinstance(question_content, str) else question_content
    language = "Traditional Chinese" if re.search(r"[一-鿿]", question_str) else "English"

    messages = ([{"role": "system", "content": SYSTEM_PROMPT}] + chat_history[:-1]
                + [{"role": "system", "content": f"Reply in {language}."}]
                + chat_history[-1:])
    trace = []

    for step in range(MAX_STEPS):
        message = client.chat.completions.create(
            model=CHAT_MODEL, messages=messages, tools=TOOL_SPECS, temperature=0.2,
        ).choices[0].message

        if not message.tool_calls:
            return (message.content or "").strip(), trace

        messages.append(message)

        for call in message.tool_calls:
            name = call.function.name
            arguments = json.loads(call.function.arguments or "{}")
            function = TOOL_FUNCTIONS.get(name)
            try:
                result = function(**arguments) if function else {"error": f"Unknown tool: {name}"}
            except Exception as error:
                result = {"error": f"{type(error).__name__}: {error}"}

            trace.append({"step": step + 1, "tool": name, "arguments": arguments, "result": result})
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": json.dumps(result, ensure_ascii=False)})

    return "Sorry, I could not finish that request. Please call our staff at 07-123-4567.", trace


# ---------------------------------------------------------------------------
# 4. Gradio Interface (Bilingual English & Chinese)
# ---------------------------------------------------------------------------
GREETING = ("Hello! I am the online agent of Wei-Restaurant, University Road branch. "
            "Ask me about our menu, prices, total bill, or policies like parking and birthday deals.\n\n"
            "您好！我是 Wei-Restaurant 大學路店的線上客服。歡迎詢問菜單、價格、計算金額，或是停車與壽星優惠等店內規定！")

EMPTY_TRACE = "### Tools used\n\n(nothing yet)"

SAMPLE_GROUPS_EN = {
    "Prices & Database (get_price, get_shop_info)": [
        "How much is the Ribeye Steak?",
        "What time do you open for lunch and dinner?",
        "Is there a student discount?",
    ],
    "RAG Search (search_shop_rules)": [
        "Can I bring my dog to the restaurant?",
        "Do you offer free parking validation?",
        "What is the birthday discount this month?",
    ],
    "Bill Calculation (calculate_total)": [
        "I want 2 Truffle Mushroom Pastas and a Classic Tiramisu. How much is it?",
    ],
    "Which Does Not Exist in db": [
        "Do you sell Japanese ramen or boba tea?",
        "What is the weather in Kaohsiung tomorrow?",
    ],
    "Transfer to Human (transfer_to_human)": [
        "My steak was cold and the service was bad. I want to speak to a manager.",
    ],
}

SAMPLE_GROUPS_ZH = {
    "價格與店家資訊（get_price、get_shop_info）": [
        "爐烤肋眼牛排多少錢？",
        "你們午餐和晚餐時間幾點到幾點？",
        "憑學生證有折扣嗎？",
    ],
    "店內規定 RAG 檢索（search_shop_rules）": [
        "請問可以帶寵物用餐嗎？",
        "開車去有停車折抵嗎？",
        "壽星當月用餐有什麼優惠？",
    ],
    "點餐算錢（calculate_total）": [
        "我要兩份黑松露野菇義大利麵和一份提拉米蘇，總共多少錢？",
    ],
    "測試不存在的品項與無關問題": [
        "你們店裡有賣日式拉麵或珍珠奶茶嗎？",
        "請問明天高雄天氣怎麼樣？",
    ],
    "客訴與專人轉接（transfer_to_human）": [
        "剛才送來的餐點是冷的，我要找經理投訴！",
    ],
}


def format_trace(trace, language="English"):
    if not trace:
        return "### Tools used\n\n(no tool was needed for this answer)"
    lines = ["### Tools used", ""]
    for entry in trace:
        lines += [f"**{entry['step']}. {entry['tool']}**", "",
                  f"arguments: `{json.dumps(entry['arguments'], ensure_ascii=False)[:160]}`", ""]
        chunks = entry["result"].get("chunks")
        if chunks:
            for s in chunks:
                title_display = s.get("title_en") if language == "English" else s.get("title")
                lines.append(f"- {title_display} (score {s['score']})")
            lines.append("")
        else:
            lines += [f"result: `{json.dumps(entry['result'], ensure_ascii=False)[:260]}`", ""]
    return "\n".join(lines)


def start_chat():
    return [{"role": "assistant", "content": GREETING}]


def add_user_message(user_text, chat_history):
    user_text = user_text.strip()
    if not user_text:
        return "", chat_history
    return "", chat_history + [{"role": "user", "content": user_text}]


def answer_question(chat_history):
    if not chat_history or chat_history[-1]["role"] != "user":
        return chat_history, EMPTY_TRACE
    try:
        user_msg = str(chat_history[-1]["content"])
        language = "Traditional Chinese" if re.search(r"[一-鿿]", user_msg) else "English"
        
        answer, trace = run_agent(chat_history)
        panel = format_trace(trace, language=language)
    except Exception as error:
        answer = f"(system message) API error: {error}"
        panel = "### Tools used\n\n(the call failed)"
    return chat_history + [{"role": "assistant", "content": answer}], panel


with gr.Blocks(title="Wei-Restaurant Agent + RAG") as demo:
    gr.Markdown(
        "## Wei-Restaurant Customer Service Agent (Tool Calling + RAG)\n"
    )

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(value=start_chat(), height=430, label="Agent")
            with gr.Row():
                user_box = gr.Textbox(placeholder="Ask about menu, prices, rules, or discounts / 詢問菜單、價格、規定...",
                                      show_label=False, scale=8)
                send_button = gr.Button("Send", variant="primary", scale=1)
                clear_button = gr.Button("Clear", scale=1)

            gr.Markdown("**Sample questions / 範例問題** (click one, then press Send)")
            with gr.Tabs():
                with gr.Tab("English"):
                    for group, questions in SAMPLE_GROUPS_EN.items():
                        gr.Examples(examples=questions, inputs=user_box, label=group)
                with gr.Tab("中文"):
                    for group, questions in SAMPLE_GROUPS_ZH.items():
                        gr.Examples(examples=questions, inputs=user_box, label=group)

        with gr.Column(scale=2):
            trace_panel = gr.Markdown(EMPTY_TRACE)

    send_button.click(add_user_message, [user_box, chatbot], [user_box, chatbot]).then(
        answer_question, chatbot, [chatbot, trace_panel])
    user_box.submit(add_user_message, [user_box, chatbot], [user_box, chatbot]).then(
        answer_question, chatbot, [chatbot, trace_panel])

    clear_button.click(start_chat, outputs=chatbot).then(
        lambda: ("", EMPTY_TRACE), outputs=[user_box, trace_panel])


if __name__ == "__main__":
    demo.launch()