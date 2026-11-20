"""Customer Streamlit frontend.

The frontend does NOT import Chatbot/Pinecone. It calls the central FastAPI service.
"""
import os
import threading
import time
import uuid

import requests
import streamlit as st

from textutil import strip_citations

st.set_page_config(page_title="NovaCart | Smart gadgets for everyday life", page_icon="🛒", layout="wide")
STORE = "NovaCart"
GREETING = ("Hi! 👋 I'm the virtual assistant. Ask me about orders, shipping, returns, "
            "warranty and more.")
CHIPS = ["What is your return policy?", "How long does shipping take?", "How do I reset my password?"]


def setting(name, default=""):
    try:
        return str(st.secrets.get(name, os.getenv(name, default)))
    except Exception:
        return os.getenv(name, default)


API_URL = setting("API_URL").rstrip("/")

# ---------------------------------------------------------------- styling
st.markdown("""
<style>
header[data-testid="stHeader"], #MainMenu, footer {display:none !important;}
.block-container {padding:0 !important; max-width:100% !important;}
[data-testid="stAppViewContainer"] {background:#f6f8fc;}
[data-testid="stForm"] {border:0 !important; padding:0 !important;}
.st-key-chat_fab {position:fixed; right:24px; bottom:24px; z-index:1000; width:auto !important;}
.st-key-chat_fab button {width:60px; height:60px; border-radius:50%; background:#1a3c6e; border:0; box-shadow:0 8px 24px rgba(0,0,0,.28);}
.st-key-chat_fab button p {font-size:26px; color:#fff; margin:0;}
.st-key-chat_fab button:hover {background:#2f6bb5; border:0;}
.st-key-chat_panel {--np-h:min(67vh, calc(100vh - 130px)); position:fixed; right:24px; bottom:96px;
  width:clamp(420px, 36vw, 640px); max-width:calc(100vw - 32px); height:var(--np-h); overflow:hidden;
  background:#fff; border-radius:16px; box-shadow:0 12px 40px rgba(0,0,0,.28); z-index:999; padding:0 14px 10px; gap:.5rem;}
.st-key-chat_panel .st-key-chat_msgs {height:calc(var(--np-h) - 140px) !important; max-height:none !important; flex:none !important;
  overflow-y:auto !important; padding-right:4px;}
.np-head {background:#1a3c6e; color:#fff; margin:0 -14px; padding:14px 16px;}
.np-head b {font-size:1.05rem;} .np-head span {display:block; font-size:.75rem; opacity:.8;}
[data-testid="InputInstructions"] {display:none !important;}
@media (max-width:640px) {
  .st-key-chat_panel {right:8px; width:calc(100vw - 16px);}
  .st-key-chat_panel [data-testid="stHorizontalBlock"] {flex-wrap:nowrap !important;}
  .st-key-chat_panel [data-testid="stColumn"] {min-width:0 !important;}
  .st-key-chat_panel [data-testid="stForm"] button {padding-left:.2rem; padding-right:.2rem; min-width:0;}
}
.np-bar {background:#fff; border-bottom:1px solid #e4e9f2; padding:14px 28px; display:flex; align-items:center; gap:24px;}
.np-logo {font-weight:800; font-size:1.4rem; color:#1a3c6e;} .np-logo span {color:#ff7a3d;}
.np-nav {margin-left:auto; display:flex; gap:20px; font-size:.95rem; color:#5b6778;}
.np-hero {background:linear-gradient(120deg,#1a3c6e,#2f6bb5); color:#fff; padding:70px 28px;}
.np-hero h1 {font-size:clamp(2rem,5vw,3.2rem); margin:0 0 12px; max-width:16ch; line-height:1.1; color:#fff;}
.np-hero p {max-width:46ch; opacity:.9; font-size:1.1rem;}
.np-btn {display:inline-block; background:#ff7a3d; color:#fff; padding:12px 22px; border-radius:10px; font-weight:600; margin-top:8px;}
.np-perks {max-width:1100px; margin:-28px auto 0; padding:0 20px; display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:14px;}
.np-perk {background:#fff; border-radius:12px; padding:16px 18px; box-shadow:0 4px 14px rgba(26,60,110,.08); font-size:.95rem;}
.np-perk b {display:block; color:#1a3c6e; margin-bottom:2px;}
.np-sec {max-width:1100px; margin:50px auto; padding:0 20px;} .np-sec h2 {color:#1a3c6e;}
.np-grid {display:grid; grid-template-columns:repeat(auto-fill,minmax(230px,1fr)); gap:18px;}
.np-card {background:#fff; border-radius:14px; overflow:hidden; box-shadow:0 4px 14px rgba(26,60,110,.08);}
.np-thumb {height:150px; display:grid; place-items:center; font-size:3.4rem; background:linear-gradient(135deg,#e8eefb,#d3dffa);}
.np-body {padding:14px 16px 18px;} .np-body h3 {margin:0 0 4px; font-size:1.05rem; color:#1b2430;}
.np-price {color:#ff7a3d; font-weight:700;}
.np-foot {background:#13233f; color:#b9c4d8; padding:30px 20px; text-align:center; font-size:.9rem; margin-top:60px;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="np-bar"><div class="np-logo">Nova<span>Cart</span></div>
<div class="np-nav"><span>Shop</span><span>Deals</span><span>NovaCart Plus</span><span>Help</span></div></div>
<div class="np-hero"><h1>Smart gadgets for everyday life</h1>
<p>Audio, wearables and smart-home essentials with fast shipping and easy returns.</p>
<span class="np-btn">Shop now</span></div>
<div class="np-perks">
<div class="np-perk"><b>🚚 Free shipping</b>On orders over $75</div>
<div class="np-perk"><b>↩️ Easy returns</b>45-day return window</div>
<div class="np-perk"><b>🛡️ Warranty</b>1-year coverage on every product</div>
<div class="np-perk"><b>💬 Need help?</b>Chat with us, bottom right</div></div>
<div class="np-sec"><h2>Best sellers</h2><div class="np-grid">
<div class="np-card"><div class="np-thumb">🎧</div><div class="np-body"><h3>NovaBuds Pro</h3><div class="np-price">$129.99</div></div></div>
<div class="np-card"><div class="np-thumb">⌚</div><div class="np-body"><h3>NovaWatch S2</h3><div class="np-price">$249.00</div></div></div>
<div class="np-card"><div class="np-thumb">🏠</div><div class="np-body"><h3>NovaHome Hub</h3><div class="np-price">$99.00</div></div></div>
<div class="np-card"><div class="np-thumb">📷</div><div class="np-body"><h3>NovaCam 360</h3><div class="np-price">$79.00</div></div></div></div></div>
<div class="np-foot">© 2026 NovaCart (a fictional store for a capstone project)</div>
""", unsafe_allow_html=True)


def fresh_chat():
    st.session_state.msgs = [{"role": "assistant", "content": GREETING}]
    st.session_state.session_id = str(uuid.uuid4())


def clean(answer):
    return strip_citations(answer)


def render(m):
    with st.chat_message(m["role"], avatar="🧑" if m["role"] == "user" else "🤖"):
        st.markdown(m["content"].replace("$", r"\$"))


class ServiceUnavailable(Exception):
    """Raised with a customer-friendly message (never a raw network error)."""


WAKE_MAX_SECONDS = 150      # a sleeping Render instance can need ~1-2 min to boot and load the model
CHAT_READ_TIMEOUT = 60      # once the API reports ready, Groq answers in a few seconds


@st.cache_resource(ttl=900, show_spinner=False)
def wake_api():
    """Fire-and-forget ping when the page loads, so a sleeping API starts booting while the
    visitor is still reading the page (shared across sessions, repeated at most every 15 min)."""
    def _ping():
        try:
            requests.get(f"{API_URL}/api/health", timeout=120)
        except Exception:
            pass
    threading.Thread(target=_ping, daemon=True).start()
    return True


def _health():
    try:
        r = requests.get(f"{API_URL}/api/health", timeout=10)
        return r.json() if r.ok else None
    except (requests.RequestException, ValueError):
        return None


def wait_until_ready(on_status):
    """Poll /api/health until the chatbot has finished loading (or give up)."""
    deadline = time.time() + WAKE_MAX_SECONDS
    shown = False
    while time.time() < deadline:
        h = _health()
        if h and h.get("ready"):
            return
        if h and h.get("warmup") == "error":
            return          # let the chat call below trigger a fresh start and report properly
        if not shown:
            on_status("⏳ Waking up the assistant... this can take up to a minute after a quiet period.")
            shown = True
        time.sleep(3)
    raise ServiceUnavailable("The assistant is taking longer than usual to start. Please try again in a minute.")


def ask_api(question, on_status=lambda msg: None):
    if not API_URL:
        raise ServiceUnavailable("The chat service is not configured yet.")
    wait_until_ready(on_status)
    payload = {"session_id": st.session_state.session_id, "message": question}
    for attempt in range(4):
        try:
            r = requests.post(f"{API_URL}/api/chat", json=payload, timeout=(10, CHAT_READ_TIMEOUT))
        except requests.ReadTimeout:
            # The server accepted the request but is slow: do NOT resend (it may still be working).
            raise ServiceUnavailable("The assistant is taking longer than usual to reply. Please try again.")
        except requests.RequestException:      # connection refused / reset / connect timeout: safe to retry
            if attempt == 3:
                raise ServiceUnavailable("I couldn't reach the support service. Please try again in a moment.")
            time.sleep(3)
            continue
        if r.status_code == 503 and attempt < 3:          # still starting: wait and retry
            on_status("⏳ Almost ready...")
            time.sleep(min(int(r.headers.get("Retry-After", "5") or 5), 15))
            continue
        if not r.ok:
            try:
                detail = r.json().get("detail", "")
            except ValueError:
                detail = ""
            raise ServiceUnavailable(detail if isinstance(detail, str) and detail else
                                     "Sorry, something went wrong. Please try again.")
        return r.json()["answer"]
    raise ServiceUnavailable("The assistant is busy right now. Please try again in a moment.")


def reset_api_session():
    if API_URL:
        try:
            requests.post(f"{API_URL}/api/reset", json={"session_id": st.session_state.session_id}, timeout=15)
        except Exception:
            pass


def run_js(js, tag=0):
    html = f"<script>{js}</script><!--{tag}-->"
    if hasattr(st, "iframe"):
        st.iframe(html, height=1)
    else:
        import streamlit.components.v1 as components
        components.html(html, height=0)


@st.fragment
def chat_panel():
    st.markdown(f'<div class="np-head"><b>{STORE} Support</b><span>Typically replies in seconds</span></div>', unsafe_allow_html=True)
    with st.container(key="chat_msgs"):
        box = st.container()
        chips_slot = st.empty()
    with st.form("chat_form", clear_on_submit=True):
        c1, c2, c3 = st.columns([8, 1, 1])
        text = c1.text_input("Message", key="chat_text", label_visibility="collapsed", placeholder="Type your question...")
        sent = c2.form_submit_button("➤", help="Send")
        new = c3.form_submit_button("↺", help="Start a new chat")

    if new:
        reset_api_session()
        fresh_chat()
        st.rerun()

    msgs = st.session_state.msgs
    question = text.strip() if sent and text.strip() else None
    if len(msgs) == 1 and not question:
        with chips_slot.container():
            for i, q in enumerate(CHIPS):
                if st.button(q, key=f"chip{i}"):
                    question = q
    if question:
        chips_slot.empty()
        history = [(m["role"], m["content"]) for m in msgs]
        msgs.append({"role": "user", "content": question})

    with box:
        for m in msgs:
            render(m)
        if question:
            status = st.empty()
            with st.spinner("Typing..."):
                try:
                    answer = clean(ask_api(question, lambda m: status.caption(m)))
                except ServiceUnavailable as exc:
                    answer = f"⚠️ {exc}"
                except Exception:
                    answer = "Sorry, something went wrong. Please try again."
            status.empty()
            reply = {"role": "assistant", "content": answer}
            msgs.append(reply)
            render(reply)

    run_js("const p=window.parent.document.querySelector('.st-key-chat_msgs');if(p)setTimeout(()=>{p.scrollTop=p.scrollHeight},80);", tag=len(msgs))


if "msgs" not in st.session_state:
    fresh_chat()
st.session_state.setdefault("chat_open", False)

if not API_URL:
    st.warning("Customer API is not configured. Set API_URL in Streamlit secrets.")
else:
    wake_api()

with st.container(key="chat_fab"):
    if st.button("✕" if st.session_state.chat_open else "💬", key="fab_btn"):
        st.session_state.chat_open = not st.session_state.chat_open
        st.rerun()

if st.session_state.chat_open:
    with st.container(key="chat_panel"):
        chat_panel()