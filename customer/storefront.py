
"""
NovaCart Customer Streamlit Frontend.

The frontend does NOT import Chatbot/Pinecone.
It communicates with the central FastAPI backend.
"""

import os
import uuid
import time
import logging

import requests
import streamlit as st

from textutil import strip_citations


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="NovaCart | Smart gadgets for everyday life",
    page_icon="🛒",
    layout="wide"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("customer")

STORE = "NovaCart"

GREETING = (
    "Hi! 👋 I'm the virtual assistant. Ask me about orders, "
    "shipping, returns, warranty and more."
)

CHIPS = [
    "What is your return policy?",
    "How long does shipping take?",
    "How do I reset my password?"
]


# ============================================================
# API CONFIGURATION
# ============================================================

def setting(name, default=""):
    try:
        return str(st.secrets.get(name, os.getenv(name, default)))
    except Exception:
        return os.getenv(name, default)


API_URL = setting("API_URL").strip().rstrip("/")


# ============================================================
# STYLING
# ============================================================

st.markdown("""
<style>

header[data-testid="stHeader"],
#MainMenu,
footer {
    display: none !important;
}

.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

[data-testid="stAppViewContainer"] {
    background: #f6f8fc;
}

[data-testid="stForm"] {
    border: 0 !important;
    padding: 0 !important;
}

/* Floating chat button */

.st-key-chat_fab {
    position: fixed;
    right: 24px;
    bottom: 24px;
    z-index: 1000;
    width: auto !important;
}

.st-key-chat_fab button {
    width: 60px;
    height: 60px;
    border-radius: 50%;
    background: #1a3c6e;
    border: 0;
    box-shadow: 0 8px 24px rgba(0,0,0,.28);
}

.st-key-chat_fab button p {
    font-size: 26px;
    color: #fff;
    margin: 0;
}

.st-key-chat_fab button:hover {
    background: #2f6bb5;
    border: 0;
}

/* Chat panel */

.st-key-chat_panel {
    --np-h: min(67vh, calc(100vh - 130px));

    position: fixed;
    right: 24px;
    bottom: 96px;

    width: clamp(420px, 36vw, 640px);
    max-width: calc(100vw - 32px);
    height: var(--np-h);

    overflow: hidden;
    background: #fff;
    border-radius: 16px;

    box-shadow: 0 12px 40px rgba(0,0,0,.28);

    z-index: 999;
    padding: 0 14px 10px;
    gap: .5rem;
}

/* Messages */

.st-key-chat_panel .st-key-chat_msgs {
    height: calc(var(--np-h) - 140px) !important;
    max-height: none !important;
    flex: none !important;
    overflow-y: auto !important;
    padding-right: 4px;
}

.np-head {
    background: #1a3c6e;
    color: #fff;
    margin: 0 -14px;
    padding: 14px 16px;
}

.np-head b {
    font-size: 1.05rem;
}

.np-head span {
    display: block;
    font-size: .75rem;
    opacity: .8;
}

[data-testid="InputInstructions"] {
    display: none !important;
}

/* Mobile */

@media (max-width: 640px) {

    .st-key-chat_panel {
        right: 8px;
        width: calc(100vw - 16px);
    }

    .st-key-chat_panel [data-testid="stHorizontalBlock"] {
        flex-wrap: nowrap !important;
    }

    .st-key-chat_panel [data-testid="stColumn"] {
        min-width: 0 !important;
    }

    .st-key-chat_panel [data-testid="stForm"] button {
        padding-left: .2rem;
        padding-right: .2rem;
        min-width: 0;
    }
}

/* Store header */

.np-bar {
    background: #fff;
    border-bottom: 1px solid #e4e9f2;
    padding: 14px 28px;
    display: flex;
    align-items: center;
    gap: 24px;
}

.np-logo {
    font-weight: 800;
    font-size: 1.4rem;
    color: #1a3c6e;
}

.np-logo span {
    color: #ff7a3d;
}

.np-nav {
    margin-left: auto;
    display: flex;
    gap: 20px;
    font-size: .95rem;
    color: #5b6778;
}

/* Hero */

.np-hero {
    background: linear-gradient(120deg, #1a3c6e, #2f6bb5);
    color: #fff;
    padding: 70px 28px;
}

.np-hero h1 {
    font-size: clamp(2rem, 5vw, 3.2rem);
    margin: 0 0 12px;
    max-width: 16ch;
    line-height: 1.1;
    color: #fff;
}

.np-hero p {
    max-width: 46ch;
    opacity: .9;
    font-size: 1.1rem;
}

.np-btn {
    display: inline-block;
    background: #ff7a3d;
    color: #fff;
    padding: 12px 22px;
    border-radius: 10px;
    font-weight: 600;
    margin-top: 8px;
}

/* Store perks */

.np-perks {
    max-width: 1100px;
    margin: -28px auto 0;
    padding: 0 20px;

    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 14px;
}

.np-perk {
    background: #fff;
    border-radius: 12px;
    padding: 16px 18px;
    box-shadow: 0 4px 14px rgba(26,60,110,.08);
    font-size: .95rem;
}

.np-perk b {
    display: block;
    color: #1a3c6e;
    margin-bottom: 2px;
}

/* Products */

.np-sec {
    max-width: 1100px;
    margin: 50px auto;
    padding: 0 20px;
}

.np-sec h2 {
    color: #1a3c6e;
}

.np-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
    gap: 18px;
}

.np-card {
    background: #fff;
    border-radius: 14px;
    overflow: hidden;
    box-shadow: 0 4px 14px rgba(26,60,110,.08);
}

.np-thumb {
    height: 150px;
    display: grid;
    place-items: center;
    font-size: 3.4rem;
    background: linear-gradient(135deg, #e8eefb, #d3dffa);
}

.np-body {
    padding: 14px 16px 18px;
}

.np-body h3 {
    margin: 0 0 4px;
    font-size: 1.05rem;
    color: #1b2430;
}

.np-price {
    color: #ff7a3d;
    font-weight: 700;
}

/* Footer */

.np-foot {
    background: #13233f;
    color: #b9c4d8;
    padding: 30px 20px;
    text-align: center;
    font-size: .9rem;
    margin-top: 60px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# STORE WEBSITE
# ============================================================

st.markdown("""
<div class="np-bar">
    <div class="np-logo">Nova<span>Cart</span></div>

    <div class="np-nav">
        <span>Shop</span>
        <span>Deals</span>
        <span>NovaCart Plus</span>
        <span>Help</span>
    </div>
</div>

<div class="np-hero">
    <h1>Smart gadgets for everyday life</h1>

    <p>
        Audio, wearables and smart-home essentials
        with fast shipping and easy returns.
    </p>

    <span class="np-btn">Shop now</span>
</div>

<div class="np-perks">

    <div class="np-perk">
        <b>🚚 Free shipping</b>
        On orders over $75
    </div>

    <div class="np-perk">
        <b>↩️ Easy returns</b>
        45-day return window
    </div>

    <div class="np-perk">
        <b>🛡️ Warranty</b>
        1-year coverage on every product
    </div>

    <div class="np-perk">
        <b>💬 Need help?</b>
        Chat with us, bottom right
    </div>

</div>

<div class="np-sec">

    <h2>Best sellers</h2>

    <div class="np-grid">

        <div class="np-card">
            <div class="np-thumb">🎧</div>
            <div class="np-body">
                <h3>NovaBuds Pro</h3>
                <div class="np-price">$129.99</div>
            </div>
        </div>

        <div class="np-card">
            <div class="np-thumb">⌚</div>
            <div class="np-body">
                <h3>NovaWatch S2</h3>
                <div class="np-price">$249.00</div>
            </div>
        </div>

        <div class="np-card">
            <div class="np-thumb">🏠</div>
            <div class="np-body">
                <h3>NovaHome Hub</h3>
                <div class="np-price">$99.00</div>
            </div>
        </div>

        <div class="np-card">
            <div class="np-thumb">📷</div>
            <div class="np-body">
                <h3>NovaCam 360</h3>
                <div class="np-price">$79.00</div>
            </div>
        </div>

    </div>
</div>

<div class="np-foot">
    © 2026 NovaCart (a fictional store for a capstone project)
</div>
""", unsafe_allow_html=True)


# ============================================================
# CHAT SESSION
# ============================================================

def fresh_chat():

    st.session_state.msgs = [
        {
            "role": "assistant",
            "content": GREETING
        }
    ]

    st.session_state.session_id = str(uuid.uuid4())


def clean(answer):
    return strip_citations(answer)


def render(message):

    with st.chat_message(
        message["role"],
        avatar="🧑" if message["role"] == "user" else "🤖"
    ):

        st.markdown(
            message["content"].replace("$", r"**\$**")
        )


# ============================================================
# API REQUEST
# ============================================================

def ask_api(question, history):

    logger.warning("CUSTOMER: ask_api called")

    logger.warning("CUSTOMER: API_URL = %s", API_URL)

    if not API_URL:
        raise RuntimeError("API_URL is not configured.")

    payload = {
        "session_id": st.session_state.session_id,
        "message": question
    }

    started = time.time()

    logger.warning("CUSTOMER: Sending request to FastAPI")

    try:

        response = requests.post(
            f"{API_URL}/api/chat",
            json=payload,
            timeout=120
        )

        logger.warning(
            "CUSTOMER: FastAPI returned HTTP %s in %.2f seconds",
            response.status_code,
            time.time() - started
        )

        if not response.ok:

            try:
                detail = response.json().get(
                    "detail",
                    "API request failed"
                )
            except Exception:
                detail = "API request failed"

            raise RuntimeError(detail)

        data = response.json()

        if "answer" not in data:
            raise RuntimeError(
                "FastAPI response does not contain an answer."
            )

        return data["answer"]

    except requests.exceptions.Timeout:
        logger.exception("CUSTOMER: FastAPI request timed out")

        raise RuntimeError(
            "The support server took too long to respond."
        )

    except requests.exceptions.ConnectionError:
        logger.exception("CUSTOMER: Cannot connect to FastAPI")

        raise RuntimeError(
            "Cannot connect to the support server."
        )

    except Exception:
        logger.exception("CUSTOMER: API request failed")
        raise


# ============================================================
# RESET API SESSION
# ============================================================

def reset_api_session():

    if not API_URL:
        return

    try:

        logger.info("CUSTOMER: Resetting API session")

        requests.post(
            f"{API_URL}/api/reset",
            json={
                "session_id": st.session_state.session_id
            },
            timeout=15
        )

    except Exception:

        logger.exception(
            "CUSTOMER: API session reset failed"
        )


# ============================================================
# SCROLL HELPER
# ============================================================

def run_js(js, tag=0):

    html = f"<script>{js}</script><!--{tag}-->"

    if hasattr(st, "iframe"):
        st.iframe(html, height=1)
    else:
        import streamlit.components.v1 as components
        components.html(html, height=0)


# ============================================================
# CHAT PANEL
# ============================================================

@st.fragment
def chat_panel():

    logger.info("CUSTOMER: Chat panel rendered")

    st.markdown(
        f"""
        <div class="np-head">
            <b>{STORE} Support</b>
            <span>Typically replies in seconds</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Message container
    with st.container(key="chat_msgs"):

        box = st.container()
        chips_slot = st.empty()

    # Input form
    with st.form(
        "chat_form",
        clear_on_submit=True
    ):

        c1, c2, c3 = st.columns([8, 1, 1])

        text = c1.text_input(
            "Message",
            key="chat_text",
            label_visibility="collapsed",
            placeholder="Type your question..."
        )

        sent = c2.form_submit_button(
            "➤",
            help="Send"
        )

        new = c3.form_submit_button(
            "↺",
            help="Start a new chat"
        )

    logger.info(
        "CUSTOMER: Form rendered. Sent=%s",
        sent
    )

    # New chat
    if new:

        logger.info("CUSTOMER: New chat clicked")

        reset_api_session()
        fresh_chat()

        st.rerun()

    msgs = st.session_state.msgs

    question = None

    # Send button
    if sent:

        logger.warning("CUSTOMER: Send button clicked")

        if text and text.strip():

            question = text.strip()

        else:

            logger.warning(
                "CUSTOMER: Empty message submitted"
            )

    # Suggested question chips
    if len(msgs) == 1 and question is None:

        with chips_slot.container():

            for i, suggested_question in enumerate(CHIPS):

                if st.button(
                    suggested_question,
                    key=f"chip{i}"
                ):

                    logger.warning(
                        "CUSTOMER: Suggested question clicked"
                    )

                    question = suggested_question
                    break

    # Process question
    if question:

        logger.warning(
            "CUSTOMER: Processing question: %s",
            question
        )

        chips_slot.empty()

        history = [
            (message["role"], message["content"])
            for message in msgs
        ]

        msgs.append({
            "role": "user",
            "content": question
        })

    # Display messages and answer
    with box:

        for message in msgs:
            render(message)

        if question:

            with st.spinner("Typing..."):

                try:

                    logger.warning(
                        "CUSTOMER: Calling ask_api"
                    )

                    answer = clean(
                        ask_api(question, history)
                    )

                    logger.warning(
                        "CUSTOMER: Answer received successfully"
                    )

                except Exception as exc:

                    logger.exception(
                        "CUSTOMER: Chat processing failed"
                    )

                    answer = (
                        "Sorry, I couldn't reach the support "
                        f"service right now. ({exc})"
                    )

            reply = {
                "role": "assistant",
                "content": answer
            }

            msgs.append(reply)

            render(reply)

    # Scroll to latest message
    run_js(
        """
        const p = window.parent.document.querySelector(
            '.st-key-chat_msgs'
        );

        if (p) {
            setTimeout(() => {
                p.scrollTop = p.scrollHeight;
            }, 80);
        }
        """,
        tag=len(msgs)
    )


# ============================================================
# INITIALIZE SESSION
# ============================================================

if "msgs" not in st.session_state:
    fresh_chat()

st.session_state.setdefault(
    "chat_open",
    False
)


# ============================================================
# API CONFIGURATION WARNING
# ============================================================

if not API_URL:

    st.warning(
        "Customer API is not configured. "
        "Set API_URL in Streamlit secrets."
    )


# ============================================================
# FLOATING CHAT BUTTON
# ============================================================

with st.container(key="chat_fab"):

    if st.button(
        "✕" if st.session_state.chat_open else "💬",
        key="fab_btn"
    ):

        st.session_state.chat_open = (
            not st.session_state.chat_open
        )

        st.rerun()


# ============================================================
# SHOW CHAT PANEL
# ============================================================

if st.session_state.chat_open:

    with st.container(key="chat_panel"):
        chat_panel()
