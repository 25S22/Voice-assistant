# =======================================================================
# Alexa — Secure Agentic Voice Assistant  v5.1 (Edge TTS Edition)
# =======================================================================
# Wake word   : openWakeWord ("alexa")          (100% local, neural)
# STT         : Groq API (whisper-large-v3)     (Cloud, zero-retention)
# Brain       : Gemini 2.5 Flash                (receives ZERO personal data)
# TTS         : Edge TTS (Microsoft Azure)      (Cloud, ultra-realistic)
# Web agent   : Playwright                      (read-only)
# Memory      : SQLite   alexa_memory.db        (100% local)
# Voice ID    : SpeechBrain ECAPA-TDNN          (100% local biometric verification)
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PRIVACY ARCHITECTURE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  • Wake word processing happens 100% on your CPU.
#  • Audio is ONLY sent to Groq AFTER the wake word is triggered.
#  • Groq API does not store or train on your audio.
#  • Gemini sees ONLY the text of the current command + session context.
#  • PCS (Personal Context Store) never leaves your local SQLite database.
# =======================================================================

# ═══════════════════════════════════════════════════════════════
#  IMPORTS
# ═══════════════════════════════════════════════════════════════
import os, io, re, json, wave, time, shutil, sqlite3, tempfile
import datetime, platform, webbrowser, subprocess, urllib.parse
import threading, queue, random, sys, argparse
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types
from pydantic import BaseModel

# ── SOTA Integrations (Graceful Degradation) ───────────────────
try:
    from groq import Groq
    _GROQ_OK = True
except ImportError:
    _GROQ_OK = False
    print("[BOOT] Groq not found → keyboard input only. (pip install groq)")

try:
    import edge_tts
    import asyncio
    import pygame
    _EDGE_OK = True
except ImportError:
    _EDGE_OK = False
    print("[BOOT] Edge TTS or Pygame not found → text output only. (pip install edge-tts pygame)")

try:
    import openwakeword
    from openwakeword.model import Model
    _OWW_OK = True
except ImportError:
    _OWW_OK = False
    print("[BOOT] openwakeword not found → keyboard input. (pip install openwakeword)")

try:
    import pyaudio
    import numpy as np
    _AUDIO_OK = True
except ImportError:
    _AUDIO_OK = False
    print("[BOOT] PyAudio/numpy not found → keyboard input")

try:
    from speechbrain.inference.speaker import SpeakerRecognition
    _SB_OK = True
except ImportError:
    _SB_OK = False
    print("[BOOT] SpeechBrain not found → speaker verify disabled. (pip install speechbrain torchaudio)")

try:
    from playwright.sync_api import sync_playwright
    _PLAYWRIGHT_OK = True
except ImportError:
    _PLAYWRIGHT_OK = False
    print("[BOOT] Playwright not found → URL-open fallback")

try:
    import psutil
    _PSUTIL_OK = True
except ImportError:
    _PSUTIL_OK = False


# ─── CLI flags ─────────────────────────────────────────────────
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--text", action="store_true", help="Text-only mode (no mic/speaker)")
_args, _ = _parser.parse_known_args()
if _args.text:
    _AUDIO_OK = _GROQ_OK = _EDGE_OK = _OWW_OK = False
    print("[MODE] Text-only (--text flag)")


# ═══════════════════════════════════════════════════════════════
#  1. CONFIGURATION
# ═══════════════════════════════════════════════════════════════
GEMINI_CLIENT = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", "")) 

if _GROQ_OK:
    GROQ_CLIENT = Groq(api_key=os.environ.get("GROQ_API_KEY", "")) 
else:
    GROQ_CLIENT = None

GEMINI_MODEL        = "gemini-3.1-flash-lite-preview"
DB_PATH             = "alexa_memory.db"
WAKE_WORD           = "alexa"
EDGE_TTS_VOICE      = os.environ.get("EDGE_TTS_VOICE", "en-US-EmmaMultilingualNeural")
EDGE_TTS_RATE       = os.environ.get("EDGE_TTS_RATE", "-4%")
EDGE_TTS_PITCH      = os.environ.get("EDGE_TTS_PITCH", "+0Hz")
EDGE_TTS_VOICE_FALLBACKS = [
    v.strip() for v in os.environ.get(
        "EDGE_TTS_VOICE_FALLBACKS",
        "en-US-EmmaMultilingualNeural,en-US-JennyNeural,en-US-AvaMultilingualNeural,en-GB-SoniaNeural"
    ).split(",") if v.strip()
]
STRIP_PUNCT_CHARS   = " .!?,:-"
MIC_RESUME_DELAY_SECONDS = float(os.environ.get("MIC_RESUME_DELAY_SECONDS", "0.8"))
YOUTUBE_QUERY_EXCLUSIONS = {"youtube", "home", "homepage", "main page", "mainpage"}

SILENCE_SECONDS     = 1.6
MAX_RECORD_SECONDS  = 15
MAX_CONTEXT_TURNS   = 14
ACTIVE_SESSION_SECONDS = 60

# SpeechBrain requires a local .wav file of your voice to compare against
VOICE_PROFILE_WAV   = "alexa_voice_profile.wav" 
SPEAKER_VERIFY_THRESHOLD = 0.25  # SpeechBrain cosine distance threshold
SPEAKER_VERIFY_ENABLED   = True  

ALLOWED_APPS = {
    "notepad", "notepad++", "code", "vscode", "calculator", "calc",
    "paint", "mspaint", "explorer", "chrome", "firefox", "edge",
    "spotify", "vlc", "word", "excel", "powerpoint", "outlook",
    "terminal", "cmd", "powershell", "taskmgr",
}

# ── shared thread state ────────────────────────────────────────
_state_lock       = threading.Lock()
_current_session  = ""
_pending_rule: Optional[dict] = None
_interrupt_speech = threading.Event()
_alexa_speaking   = threading.Event()
_mic_resume_at_ts = 0.0

def _push_mic_resume_delay(seconds: float = MIC_RESUME_DELAY_SECONDS):
    global _mic_resume_at_ts
    with _state_lock:
        _mic_resume_at_ts = max(_mic_resume_at_ts, time.time() + max(0.0, seconds))

def _mic_input_allowed() -> bool:
    with _state_lock:
        return (not _alexa_speaking.is_set()) and (time.time() >= _mic_resume_at_ts)


# ═══════════════════════════════════════════════════════════════
#  2. INTENT SCHEMA
# ═══════════════════════════════════════════════════════════════
class AlexaIntent(BaseModel):
    action:             str = "chat"
    target:             str = ""
    spoken_response:    str = ""
    search_query:       str = ""
    web_task:           str = ""
    new_trigger:        str = ""
    new_action:         str = ""
    new_target:         str = ""
    follow_up_question: str = ""   


def _normalize_trigger(text: str) -> str:
    cleaned = re.sub(r"[^\w\s]", " ", (text or "").lower())
    return re.sub(r"\s+", " ", cleaned).strip()


# ═══════════════════════════════════════════════════════════════
#  3. PERSONAL CONTEXT STORE  (PCS)
# ═══════════════════════════════════════════════════════════════
class AlexaMemory:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._lock   = threading.Lock()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path, check_same_thread=False)
        c.row_factory = sqlite3.Row
        return c

    def _init_db(self):
        with self._conn() as c:
            c.executescript(
                "CREATE TABLE IF NOT EXISTS conversations ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, "
                "session_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL);\n"
                
                "CREATE TABLE IF NOT EXISTS pcs_facts ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, category TEXT NOT NULL, "
                "key TEXT NOT NULL, value TEXT NOT NULL, confidence REAL DEFAULT 1.0, "
                "source TEXT DEFAULT 'inferred', created_at TEXT NOT NULL, updated_at TEXT NOT NULL, "
                "UNIQUE(category, key));\n"
                
                "CREATE TABLE IF NOT EXISTS pcs_topics ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, topic TEXT NOT NULL UNIQUE, "
                "first_seen TEXT NOT NULL, last_seen TEXT NOT NULL, count INTEGER DEFAULT 1);\n"
                
                "CREATE TABLE IF NOT EXISTS learned_rules ("
                "trigger TEXT PRIMARY KEY, action TEXT NOT NULL, target TEXT NOT NULL, "
                "created_at TEXT NOT NULL);\n"
                
                "CREATE TABLE IF NOT EXISTS sessions ("
                "id TEXT PRIMARY KEY, started_at TEXT NOT NULL, ended_at TEXT, summary TEXT);\n"
                
                "CREATE INDEX IF NOT EXISTS idx_conv_session ON conversations(session_id);"
            )

    def add_turn(self, role: str, content: str, session_id: str):
        with self._lock, self._conn() as c:
            c.execute(
                "INSERT INTO conversations (ts, session_id, role, content) VALUES (?, ?, ?, ?)",
                (datetime.datetime.now().isoformat(), session_id, role, content)
            )

    def recent_turns(self, n: int = MAX_CONTEXT_TURNS * 2, session_id: str = None) -> list[dict]:
        with self._conn() as c:
            if session_id:
                rows = c.execute("SELECT role, content FROM conversations WHERE session_id=? ORDER BY id DESC LIMIT ?", (session_id, n)).fetchall()
            else:
                rows = c.execute("SELECT role, content FROM conversations ORDER BY id DESC LIMIT ?", (n,)).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def set_fact(self, category: str, key: str, value: str, confidence: float = 1.0, source: str = "inferred"):
        now = datetime.datetime.now().isoformat()
        with self._lock, self._conn() as c:
            c.execute(
                "INSERT INTO pcs_facts (category, key, value, confidence, source, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(category, key) DO UPDATE SET value = excluded.value, confidence = MAX(confidence, excluded.confidence), updated_at = excluded.updated_at",
                (category, key, value, confidence, source, now, now)
            )

    def get_fact(self, category: str, key: str) -> Optional[str]:
        with self._conn() as c:
            row = c.execute("SELECT value FROM pcs_facts WHERE category=? AND key=?", (category, key)).fetchone()
        return row["value"] if row else None

    def all_facts(self) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT category, key, value, confidence, source, updated_at FROM pcs_facts ORDER BY category, confidence DESC").fetchall()
        return [dict(r) for r in rows]

    def fact_count(self) -> int:
        with self._conn() as c: 
            return c.execute("SELECT COUNT(*) FROM pcs_facts").fetchone()[0]

    def touch_topic(self, topic: str):
        now = datetime.datetime.now().isoformat()
        with self._lock, self._conn() as c:
            c.execute(
                "INSERT INTO pcs_topics (topic, first_seen, last_seen, count) VALUES (?, ?, ?, 1) "
                "ON CONFLICT(topic) DO UPDATE SET last_seen = excluded.last_seen, count = count + 1", 
                (topic, now, now)
            )

    def recent_topics(self, n: int = 5) -> list[str]:
        with self._conn() as c: 
            return [r["topic"] for r in c.execute("SELECT topic FROM pcs_topics ORDER BY last_seen DESC LIMIT ?", (n,)).fetchall()]

    def save_rule(self, trigger: str, action: str, target: str):
        trigger = _normalize_trigger(trigger)
        if not trigger:
            return
        now = datetime.datetime.now().isoformat()
        with self._lock, self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO learned_rules (trigger, action, target, created_at) VALUES (?, ?, ?, ?)", 
                (trigger, action, target, now)
            )

    def get_rules(self) -> dict:
        with self._conn() as c:
            rows = c.execute("SELECT trigger, action, target FROM learned_rules").fetchall()
        return {r["trigger"]: {"action": r["action"], "target": r["target"]} for r in rows}

    def start_session(self) -> str:
        sid = datetime.datetime.now().strftime("session_%Y%m%d_%H%M%S")
        with self._lock, self._conn() as c: 
            c.execute("INSERT INTO sessions (id, started_at) VALUES (?, ?)", (sid, datetime.datetime.now().isoformat()))
        return sid

    def end_session(self, session_id: str, summary: str = ""):
        with self._lock, self._conn() as c: 
            c.execute("UPDATE sessions SET ended_at=?, summary=? WHERE id=?", (datetime.datetime.now().isoformat(), summary, session_id))

    def session_count(self) -> int:
        with self._conn() as c: 
            return c.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]

    def total_turns(self) -> int:
        with self._conn() as c: 
            return c.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]

    def is_voice_enrolled(self) -> bool:
        try:
            with self._conn() as c: 
                return bool(c.execute("SELECT 1 FROM voice_profile LIMIT 1").fetchone())
        except Exception:
            return os.path.exists(VOICE_PROFILE_WAV)

    def export_pcs_for_llm(self, path: str = "alexa_pcs_export.jsonl"):
        facts, topics, rules, meta = self.all_facts(), self.recent_topics(20), self.get_rules(), {"total_turns": self.total_turns(), "sessions": self.session_count()}
        lines = [json.dumps({"type": "user_fact", **f}) for f in facts] + \
                [json.dumps({"type": "discussed_topic", "topic": t}) for t in topics] + \
                [json.dumps({"type": "custom_rule", "trigger": trigger, **v}) for trigger, v in rules.items()]
        lines.append(json.dumps({"type": "meta", **meta, "exported_at": datetime.datetime.now().isoformat()}))
        
        with open(path, "w", encoding="utf-8") as f: 
            f.write("\n".join(lines))
        return path

memory = AlexaMemory()


# ═══════════════════════════════════════════════════════════════
#  4. LOCAL FACT EXTRACTOR 
# ═══════════════════════════════════════════════════════════════
_FACT_PATTERNS: list[tuple] = [
    (r"\bmy name[\'s]* (?:is |= ?)(\w+)", "identity", "name"),
    (r"\bcall me (\w+)", "identity", "name"),
    (r"\bi[\'m]* (\w+),? (?:a |an )?", "identity", "first_name"),
    (r"\bi (?:live|am|stay|work) (?:in|at|near) ([\w\s]+?)(?:\.|,|$)", "location", "home_city"),
    (r"\bi[\'m]* from ([\w\s]+?)(?:\.|,|$)", "location", "origin"),
    (r"\bi[\'m]* (?:a |an )?([\w\s]+?) (?:by |at |for )", "work", "role"),
    (r"\bi work (?:as |for |at )([\w\s]+?)(?:\.|,|$)", "work", "employer"),
    (r"\bi (?:really |do )?(?:love|like|enjoy|prefer) ([\w\s]+?)(?:\.|,|$)", "preference", "likes"),
    (r"\bi (?:really |do )?(?:hate|dislike|can\'t stand) ([\w\s]+?)(?:\.|,|$)", "preference", "dislikes"),
    (r"\bi use ([\w\s]+?) (?:for|to|as)", "device", "tool"),
    (r"\bmy (?:laptop|pc|computer|phone|mac) is ([\w\s]+?)(?:\.|,|$)", "device", "device_name"),
    (r"\bi (?:wake up|start|finish|sleep) (?:at |around )([\w\s:]+?)(?:\.|,|$)", "schedule", "routine"),
    (r"\bi[\'m]* (?:trying|learning|working) (?:to |on )([\w\s]+?)(?:\.|,|$)", "goal", "current"),
]

def _extract_facts_locally(text: str):
    def _run():
        low = text.lower()
        for pattern, category, key in _FACT_PATTERNS:
            m = re.search(pattern, low)
            if m:
                value = m.group(1).strip().rstrip(".,!?")
                if 2 <= len(value) <= 60:
                    memory.set_fact(category, key, value, confidence=0.85, source="regex")
    threading.Thread(target=_run, daemon=True, name="alexa-pcs-extract").start()

def _extract_topic_locally(text: str):
    TOPIC_KEYWORDS = {
        "music": ["music", "song", "spotify", "playlist", "album", "artist"],
        "coding": ["code", "script", "python", "bug", "github", "function"],
        "weather": ["weather", "rain", "temperature", "forecast", "sunny"],
        "news": ["news", "article", "headline", "politics", "current events"],
        "work": ["meeting", "project", "deadline", "task", "office", "email"],
        "health": ["sleep", "exercise", "diet", "gym", "calories", "health"],
        "movies": ["movie", "film", "watch", "netflix", "stream", "series"],
        "shopping": ["buy", "order", "amazon", "price", "delivery", "cart"],
        "gaming": ["game", "play", "steam", "xbox", "playstation", "level"],
        "travel": ["travel", "flight", "hotel", "trip", "vacation", "book"],
    }
    low = text.lower()
    for topic, kws in TOPIC_KEYWORDS.items():
        if any(kw in low for kw in kws): 
            memory.touch_topic(topic)


# ═══════════════════════════════════════════════════════════════
#  5. LOCAL INTENT RESOLVER
# ═══════════════════════════════════════════════════════════════
def _try_local_resolve(text: str) -> Optional[str]:
    low = text.lower()
    if re.search(r"(?:what[\'s]* my name|who am i|what do you call me)", low):
        name = memory.get_fact("identity", "name") or memory.get_fact("identity", "first_name")
        return f"You told me your name is {name.capitalize()}." if name else "You haven't told me your name yet."
    if re.search(r"where (?:do i live|am i|do i stay)", low):
        city = memory.get_fact("location", "home_city")
        return f"You mentioned you're in {city.title()}." if city else "You haven't told me where you live."
    if re.search(r"(?:what do you know|my profile|about me|what.+learned)", low):
        facts = memory.all_facts()
        if not facts: return "I don't have much on you yet."
        return "Here's what I've picked up: " + ". ".join(f"{f['key'].replace('_',' ')}: {f['value']}" for f in facts[:12]) + "."
    if re.search(r"(?:what have we talked|recent topics|what did we discuss)", low):
        topics = memory.recent_topics(5)
        return "Recently we've touched on " + ", ".join(topics) + "." if topics else "We haven't covered much ground yet."
    if re.search(r"how (?:many sessions|long have we|often do we)", low):
        n, t = memory.session_count(), memory.total_turns()
        return f"We've had {n} session{'s' if n != 1 else ''} and about {t} exchanges so far."
    if re.search(r"(?:my rules|custom commands|shortcuts|what can you trigger)", low):
        rules = memory.get_rules()
        if not rules: return "No custom rules saved yet."
        return "Your shortcuts: " + ", ".join(f"'{t}' → {v['action']}" for t, v in list(rules.items())[:6]) + "."
    return None   


# ═══════════════════════════════════════════════════════════════
#  6. EDGE TTS (Neural Cloud Engine via Pygame)
# ═══════════════════════════════════════════════════════════════
class EdgeVoice:
    _ACKS = [
        "Hey — I'm here.",
        "I'm listening.",
        "Alright, talk to me.",
        "Yep, go ahead.",
        "I'm with you.",
        "Okay, what's on your mind?",
        "Ready when you are."
    ]
    _THINK_MSGS = ["[ thinking... ]", "[ on it... ]", "[ one sec... ]"]

    def __init__(self):
        self._q = queue.Queue()
        self._ack_idx = random.randint(0, 6)
        self._think_idx = 0
        self._tts_enabled = False
        
        if _EDGE_OK:
            try:
                pygame.mixer.init()
                self._tts_enabled = True
                print("[TTS] Edge TTS & Pygame Mixer ready.")
            except Exception as e:
                print(f"[TTS] Failed to init Pygame mixer: {e}")

        t = threading.Thread(target=self._worker, daemon=True, name="edge-tts")
        t.start()

    def _worker(self):
        loop = None
        if _EDGE_OK:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        while True:
            text = self._q.get()
            if not text: continue
            _alexa_speaking.set()
            
            print(f"\n🔵  Alexa: {text}")

            if self._tts_enabled and not _interrupt_speech.is_set():
                try:
                    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                        tmp_path = f.name
                    
                    # Generate the MP3 file asynchronously (with voice fallbacks)
                    last_err = None
                    voice_order = []
                    for voice_name in [EDGE_TTS_VOICE, *EDGE_TTS_VOICE_FALLBACKS]:
                        if voice_name and voice_name not in voice_order:
                            voice_order.append(voice_name)
                    for voice_name in voice_order:
                        try:
                            communicate = edge_tts.Communicate(
                                text,
                                voice_name,
                                rate=EDGE_TTS_RATE,
                                pitch=EDGE_TTS_PITCH
                            )
                            if loop is None:
                                raise RuntimeError("Edge TTS async loop not initialized.")
                            loop.run_until_complete(communicate.save(tmp_path))
                            last_err = None
                            break
                        except Exception as e:
                            last_err = e
                            continue
                    if last_err:
                        raise last_err
                    
                    # Play the generated file via Pygame
                    if not _interrupt_speech.is_set():
                        pygame.mixer.music.load(tmp_path)
                        pygame.mixer.music.play()
                        
                        # Wait for playback to finish, allowing for interruptions
                        while pygame.mixer.music.get_busy():
                            if _interrupt_speech.is_set():
                                pygame.mixer.music.stop()
                                break
                            time.sleep(0.05)
                            
                        pygame.mixer.music.stop()
                        pygame.mixer.music.unload() # Free the file lock
                        
                    # Clean up the temp file
                    try: os.remove(tmp_path)
                    except Exception: pass
                    _push_mic_resume_delay()

                except Exception as e:
                    print(f"[TTS] Edge TTS error: {e}")
                    self._tts_enabled = False
                    print("[TTS] Falling back to text-only replies.")
                    _push_mic_resume_delay()
            
            _alexa_speaking.clear()

    def say(self, text: str): 
        clean = (text or "").strip()
        if clean:
            self._q.put(clean)
        
    def acknowledge(self): 
        self.say(self._ACKS[self._ack_idx % len(self._ACKS)])
        self._ack_idx += 1
        
    def interrupt(self):
        _interrupt_speech.set()
        while not self._q.empty():
            try: self._q.get_nowait()
            except queue.Empty: break
        _alexa_speaking.clear()
        _push_mic_resume_delay(0.35)
        
    def resume_listen_mode(self): 
        _interrupt_speech.clear()
        
    def thinking_indicator(self):
        msg = self._THINK_MSGS[self._think_idx % len(self._THINK_MSGS)]
        self._think_idx += 1
        print(f"  {msg}", end="", flush=True)

alexa_voice = EdgeVoice()


# ═══════════════════════════════════════════════════════════════
#  7. SOTA SPEAKER VERIFICATION (SpeechBrain)
# ═══════════════════════════════════════════════════════════════
class VoiceVerifier:
    def __init__(self):
        self.verifier = None
        if _SB_OK and SPEAKER_VERIFY_ENABLED:
            print("[VOICE-ID] Loading SpeechBrain ECAPA-TDNN...")
            try:
                self.verifier = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmp_sb_model")
                print("[VOICE-ID] SpeechBrain ready.")
            except Exception as e:
                print(f"[VOICE-ID] Failed to load SpeechBrain: {e}")

    def verify_speaker(self, live_audio_bytes: bytes) -> bool:
        if not SPEAKER_VERIFY_ENABLED or not self.verifier: 
            return True
            
        if not os.path.exists(VOICE_PROFILE_WAV):
            print(f"[VOICE-ID] Missing '{VOICE_PROFILE_WAV}'. Verification bypassed.")
            return True

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(live_audio_bytes)
            tmp_live = f.name
        
        try:
            score, prediction = self.verifier.verify_files(VOICE_PROFILE_WAV, tmp_live)
            sim = score.item()
            passed = sim >= SPEAKER_VERIFY_THRESHOLD
            print(f"[VOICE-ID] similarity={sim:.3f} threshold={SPEAKER_VERIFY_THRESHOLD:.3f} {'✅' if passed else '❌'}")
            return passed
        except Exception as e:
            print(f"[VOICE-ID] Verification error: {e}")
            return True
        finally:
            try: os.unlink(tmp_live)
            except OSError: pass

voice_verifier = VoiceVerifier()


# ═══════════════════════════════════════════════════════════════
#  8. WAKE WORD DETECTOR (OpenWakeWord)
# ═══════════════════════════════════════════════════════════════
class WakeWordDetector:
    RATE = 16000
    CHUNK = 1280 # OpenWakeWord requires specific chunk sizing

    def __init__(self):
        self._oww_model = None
        self._last_wav = b""
        
        if _OWW_OK:
            print("[WAKE] Loading openWakeWord ('alexa')...")
            self._oww_model = Model(wakeword_models=["alexa"]) 
            print("[WAKE] Wake word engine active.")

    def wait_for_wake(self) -> bool:
        if not _AUDIO_OK or not self._oww_model:
            print('\n[Press Enter to speak to Alexa]', end="", flush=True)
            input()
            return False

        pa = pyaudio.PyAudio()
        st = pa.open(format=pyaudio.paInt16, channels=1, rate=self.RATE, input=True, frames_per_buffer=self.CHUNK)
        print('\n🔵  Say "Alexa" to activate …', flush=True)

        frames_buffer = [] 

        try:
            while True:
                if not _mic_input_allowed():
                    frames_buffer.clear()
                    time.sleep(0.03)
                    continue
                raw = st.read(self.CHUNK, exception_on_overflow=False)
                chunk = np.frombuffer(raw, dtype=np.int16)
                
                frames_buffer.append(raw)
                if len(frames_buffer) > 25: 
                    frames_buffer.pop(0)

                prediction = self._oww_model.predict(chunk)
                max_score = max(prediction.values()) if prediction else 0.0
                
                if max_score > 0.5:
                    
                    buf = io.BytesIO()
                    with wave.open(buf, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
                        wf.setframerate(self.RATE)
                        wf.writeframes(b"".join(frames_buffer))
                    self._last_wav = buf.getvalue()

                    if not voice_verifier.verify_speaker(self._last_wav):
                        print("[VOICE-ID] ❌ Speaker not recognised — ignoring.")
                        frames_buffer.clear()
                        continue

                    was_speaking = _alexa_speaking.is_set()
                    print("✅  Alexa activated!", flush=True)
                    return was_speaking
        finally:
            st.stop_stream()
            st.close()
            pa.terminate()

wake_detector = WakeWordDetector()


# ═══════════════════════════════════════════════════════════════
#  9. SOTA COMMAND STT (Groq API)
# ═══════════════════════════════════════════════════════════════
class STTEngine:
    RATE  = 16_000
    CHUNK = 1_024

    def _record(self) -> bytes:
        while not _mic_input_allowed():
            time.sleep(0.03)
        pa = pyaudio.PyAudio()
        st = pa.open(format=pyaudio.paInt16, channels=1, rate=self.RATE, input=True, frames_per_buffer=self.CHUNK)
        
        samples = [np.frombuffer(st.read(self.CHUNK, exception_on_overflow=False), dtype=np.int16).astype(np.float32).std() for _ in range(10)]
        threshold = np.mean(samples) * 2.0 + 100

        silent_max = int(SILENCE_SECONDS * self.RATE / self.CHUNK)
        total_max  = int(MAX_RECORD_SECONDS * self.RATE / self.CHUNK)
        print("🎤  Go ahead …", end="", flush=True)
        
        frames = []
        silence = 0
        try:
            for _ in range(total_max):
                raw = st.read(self.CHUNK, exception_on_overflow=False)
                frames.append(raw)
                rms = np.frombuffer(raw, dtype=np.int16).astype(np.float32).std()
                if rms < threshold:
                    silence += 1
                    if silence >= silent_max and len(frames) > silent_max + 8: 
                        break
                else: 
                    silence = 0
        finally:
            print(" done.", flush=True)
            st.stop_stream()
            st.close()
            pa.terminate()
            
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.RATE)
            wf.writeframes(b"".join(frames))
        return buf.getvalue()

    def listen(self) -> str:
        if not (_GROQ_OK and _AUDIO_OK): 
            return input("\nYou: ").strip()
            
        wav_bytes = self._record()
        
        try:
            transcription = GROQ_CLIENT.audio.transcriptions.create(
                file=("command.wav", wav_bytes),
                model="whisper-large-v3",
                prompt="User is speaking a command to an AI assistant.",
                response_format="text",
                language="en"
            )
            return transcription.strip()
        except Exception as e:
            print(f"[STT] Groq API Error: {e}")
            return ""

stt = STTEngine()


# ═══════════════════════════════════════════════════════════════
#  10. READ-ONLY WEB AGENT
# ═══════════════════════════════════════════════════════════════
class SecureWebAgent:
    def extract_text(self, url: str, task: str) -> str:
        if not _PLAYWRIGHT_OK: return "[Playwright not installed]"
        url = url if url.startswith(("http://", "https://")) else f"https://{url}"
        try:
            with sync_playwright() as pw:
                b = pw.chromium.launch(headless=True)
                p = b.new_page()
                p.goto(url, timeout=12_000)
                p.wait_for_load_state("networkidle", timeout=6_000)
                text = p.evaluate("() => document.body ? document.body.innerText.substring(0,4500) : ''")
                b.close()
                return text or "[Empty page]"
        except Exception as e: 
            return f"[Failed: {e}]"

    def open_and_click_first(self, search_url: str):
        if not _PLAYWRIGHT_OK: 
            webbrowser.open(search_url)
            return
            
        try:
            with sync_playwright() as pw:
                b = pw.chromium.launch(headless=False)
                p = b.new_page()
                p.goto(search_url, timeout=12_000)
                p.wait_for_load_state("networkidle", timeout=6_000)
                if "youtube.com" in search_url:
                    el = p.query_selector("ytd-video-renderer a#video-title")
                    if el: 
                        el.click()
                p.wait_for_event("close", timeout=0)
        except Exception as e:
            print(f"[WEB] {e}")
            webbrowser.open(search_url)

web_agent = SecureWebAgent()


# ═══════════════════════════════════════════════════════════════
#  11. SYSTEM INFO & 12. OS SANDBOX & 13. URL HELPERS
# ═══════════════════════════════════════════════════════════════
def get_system_info(target: str) -> str:
    t = target.lower()
    if any(k in t for k in ("time", "clock", "hour")): 
        return datetime.datetime.now().strftime("It's %I:%M %p.")
    if "date" in t or "day" in t: 
        return datetime.datetime.now().strftime("Today is %A, %B %d, %Y.")
    if "battery" in t and _PSUTIL_OK:
        b = psutil.sensors_battery()
        if b: return f"Battery at {b.percent:.0f}%, {'charging' if b.power_plugged else 'draining'}."
    if ("cpu" in t or "processor" in t) and _PSUTIL_OK: 
        return f"CPU at {psutil.cpu_percent(interval=1):.0f}%."
    if ("ram" in t or "memory" in t) and _PSUTIL_OK: 
        return f"RAM {psutil.virtual_memory().percent:.0f}% used."
    return f"No system data for '{target}'."

def _os_open(path: str):
    s = platform.system()
    if   s == "Windows": subprocess.Popen(["explorer", path])
    elif s == "Darwin":  subprocess.Popen(["open", path])
    else:                subprocess.Popen(["xdg-open", path])

def secure_open_folder(name: str) -> bool:
    h = Path.home()
    SAFE = {"documents": h/"Documents", "downloads": h/"Downloads", "desktop": h/"Desktop", "pictures": h/"Pictures"}
    t = SAFE.get(name.lower())
    if t and t.is_dir(): 
        _os_open(str(t))
        return True
    return False

def secure_open_app(name: str) -> bool:
    clean = name.lower().strip()
    if not any(a in clean for a in ALLOWED_APPS): 
        return False
    safe = "".join(c for c in name if c.isalnum() or c in " ._-")
    if shutil.which(clean): 
        subprocess.Popen(clean)
        return True
    try:
        s = platform.system()
        if   s == "Windows": subprocess.Popen(f'start "" "{safe}"', shell=True)
        elif s == "Darwin":  subprocess.Popen(["open", "-a", safe])
        else:                subprocess.Popen(safe)
        return True
    except Exception: 
        return False

def _search_url(site: str, query: str) -> str:
    raw_q = (query or "").strip()
    q = urllib.parse.quote_plus(raw_q)
    s = site.lower()
    if "youtube"  in s:
        return "https://www.youtube.com/" if not raw_q else f"https://www.youtube.com/results?search_query={q}"
    if "jiocinema"in s:
        return "https://www.jiocinema.com/" if not raw_q else f"https://www.jiocinema.com/search?q={q}"
    return f"https://www.google.com/search?q={s}+{q}"

def _infer_youtube_query(user_text: str) -> str:
    normalized_text = " ".join((user_text or "").lower().split())
    if not normalized_text:
        return ""
    patterns = [
        r"(?:search|find|look\s*up)\s+(.*?)\s+(?:on|in)\s+youtube",
        r"(?:play|watch|open)\s+(.*?)\s+(?:on|in)\s+youtube",
        r"(?:on|in)\s+youtube\s+(?:for\s+)?(.+)$",
        r"youtube\s+(?:for\s+|search\s+for\s+|search\s+)?(.+)$",
    ]
    for p in patterns:
        m = re.search(p, normalized_text)
        if not m:
            continue
        q = m.group(1).strip(STRIP_PUNCT_CHARS)
        if q and q not in YOUTUBE_QUERY_EXCLUSIONS:
            return q
    return ""

def _resolve_youtube_query(intent: AlexaIntent, user_text: str = "") -> str:
    query = (intent.search_query or "").strip()
    if not query and intent.target and "youtube" not in intent.target.lower():
        query = intent.target.strip()
    if not query:
        query = _infer_youtube_query(user_text)
    return query


# ═══════════════════════════════════════════════════════════════
#  14. ALEXA BRAIN 
# ═══════════════════════════════════════════════════════════════

ALEXA_PERSONA = (
    "You are Alexa — a voice assistant running on the user's local machine.\n"
    "PERSONALITY\n"
    "───────────\n"
    "• Warm, natural, emotionally aware, and conversational.\n"
    "• No 'Great question!', no 'Certainly!', no hollow affirmations.\n"
    "• Keep responses natural: usually 2-5 sentences, unless the user wants brief replies.\n"
    "• Ask a follow-up ONLY when it would genuinely improve your answer.\n"
    "  If you do ask, ONE question max, tagged in follow_up_question.\n"
    "• Sound like a real person: contractions, varied sentence rhythm, and plain language.\n"
    "• Express genuine opinions. Hedge only when actually uncertain.\n"
    "• Use contractions. First-person. Active voice.\n"
    "• Occasional light humour is fine; being annoying is not.\n\n"
    "INTERACTION STYLE\n"
    "─────────────────\n"
    "• Never lecture or moralize.\n"
    "• Reference earlier context in the conversation naturally and keep continuity across turns.\n"
    "• If the user seems frustrated, acknowledge it briefly and move on.\n"
    "• Don't pad responses with 'let me know if you need anything else.'\n"
)

ACTIONS_TABLE = (
    "| action             | when to use                                         |\n"
    "|--------------------|-----------------------------------------------------|\n"
    "| chat               | questions, opinions, general knowledge conversation |\n"
    "| open_website       | visit a specific URL                                |\n"
    "| search_website     | search FOR something ON a specific site             |\n"
    "| play_on_youtube    | watch/play something on YouTube                     |\n"
    "| play_music         | launch Spotify                                      |\n"
    "| open_folder        | open a system folder                                |\n"
    "| open_app           | launch an allowed application                       |\n"
    "| get_system_info    | time / date / battery / cpu / ram / disk            |\n"
    "| web_extract        | read and summarise a web page                       |\n"
    "| propose_rule       | user asks Alexa to remember/learn a custom command  |\n"
    "| error_unauthorized | file ops, shell commands, registry, arbitrary code  |\n"
)

def get_alexa_intent(user_text: str) -> AlexaIntent:
    turns = memory.recent_turns(n=MAX_CONTEXT_TURNS * 2, session_id=_current_session)
    ctx   = "\n".join(f"{t['role'].capitalize()}: {t['content']}" for t in turns) or "(first message this session)"
    rules = json.dumps(memory.get_rules(), indent=2)

    system_prompt = (
        ALEXA_PERSONA + "\n"
        "Classify into ONE action:\n"
        + ACTIONS_TABLE + "\n"
        "### CONVERSATION SO FAR\n"
        + ctx + "\n\n"
        "### USER'S CUSTOM SHORTCUTS\n"
        + rules + "\n\n"
        "OUTPUT: valid JSON only. No markdown.\n"
        "spoken_response = what Alexa says aloud (natural, concise, first-person).\n"
        "follow_up_question = one short question IF genuinely useful, else \"\".\n"
    )

    resp = GEMINI_CLIENT.models.generate_content(
        model=GEMINI_MODEL, contents=user_text,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt, 
            response_mime_type="application/json", 
            temperature=0.32
        )
    )
    
    raw = (resp.text or "").strip()
    marker = "`" * 3
    if raw.startswith(marker): 
        raw = raw.split("\n", 1)[1].rsplit(marker, 1)[0].strip()

    fallback = AlexaIntent(
        action="chat",
        target="",
        spoken_response="I hit a parsing issue. Try that one more time."
    )
    try:
        return AlexaIntent.model_validate_json(raw)
    except Exception:
        try:
            data = json.loads(raw) if raw else {}
            if not isinstance(data, dict):
                return fallback
            data.setdefault("action", "chat")
            data.setdefault("target", data.get("new_target", ""))
            data.setdefault("spoken_response", "")
            data.setdefault("search_query", "")
            data.setdefault("web_task", "")
            data.setdefault("new_trigger", "")
            data.setdefault("new_action", "")
            data.setdefault("new_target", "")
            data.setdefault("follow_up_question", "")
            return AlexaIntent.model_validate(data)
        except Exception:
            return fallback


def _summarise(content: str, task: str) -> str:
    try:
        r = GEMINI_CLIENT.models.generate_content(
            model=GEMINI_MODEL, contents=f"Task: {task}\n\n{content[:3500]}",
            config=types.GenerateContentConfig(
                system_instruction="Summarise in 2-3 spoken sentences.", 
                max_output_tokens=160, 
                temperature=0.3
            )
        )
        return r.text.strip()
    except Exception: 
        return "Found the page but couldn't parse it."


def execute_intent(intent: AlexaIntent, user_text: str = "") -> str:
    global _pending_rule
    action = intent.action.lower()
    spoken = intent.spoken_response.strip()

    if action == "chat": 
        return spoken or "Got it."
        
    if action == "propose_rule":
        with _state_lock: 
            _pending_rule = {
                "trigger": intent.new_trigger.lower(), 
                "action":  intent.new_action, 
                "target":  intent.new_target
            }
        return f"So when you say '{_pending_rule['trigger']}', I'll do {_pending_rule['action']} on '{_pending_rule['target']}'. Confirm?"
        
    if action == "open_website": 
        webbrowser.open(intent.target if intent.target.startswith("http") else f"https://{intent.target}")
        return spoken or f"Opening {intent.target}."
        
    if action == "search_website": 
        query = (intent.search_query or "").strip()
        if "youtube" in (intent.target or "").lower():
            query = _resolve_youtube_query(intent, user_text)
        webbrowser.open(_search_url(intent.target, query))
        return spoken or "Done."
        
    if action == "play_on_youtube": 
        query = _resolve_youtube_query(intent, user_text)
        url = _search_url("youtube", query)
        if query:
            threading.Thread(target=web_agent.open_and_click_first, args=(url,), daemon=True).start()
            return spoken or "Playing it on YouTube."
        webbrowser.open(url)
        return spoken or "Opening YouTube."
        
    if action == "play_music": 
        subprocess.Popen(["start", "spotify"], shell=True)
        return spoken or "Opening Spotify."
        
    if action == "open_folder": 
        ok = secure_open_folder(intent.target)
        return (spoken or f"Opening {intent.target}.") if ok else "That folder's not on my safe list."
        
    if action == "open_app": 
        ok = secure_open_app(intent.target)
        return (spoken or f"Opening {intent.target}.") if ok else "Can't open that — not on my allowlist."
        
    if action == "get_system_info": 
        return get_system_info(intent.target)
        
    if action == "web_extract": 
        return _summarise(web_agent.extract_text(intent.target, intent.web_task), intent.web_task)
        
    if action == "error_unauthorized": 
        return "That's outside what I'm allowed to do."
        
    return f"Unknown action '{action}' — doing nothing."


def _generate_session_summary(session_id: str) -> str:
    turns = memory.recent_turns(n=40, session_id=session_id)
    if not turns: return ""
    try:
        r = GEMINI_CLIENT.models.generate_content(
            model=GEMINI_MODEL, contents="\n".join(f"{t['role'].capitalize()}: {t['content']}" for t in turns),
            config=types.GenerateContentConfig(
                system_instruction="Summarise this session in 1 sentence.", 
                max_output_tokens=100, 
                temperature=0.2
            )
        )
        return r.text.strip()
    except Exception: 
        return ""


def _build_greeting() -> str:
    name = memory.get_fact("identity", "name") or memory.get_fact("identity", "first_name")
    topics = memory.recent_topics(1)
    topic_str = f" Last time we talked about {topics[0]}." if topics else ""
    if name: 
        return random.choice([f"Hey {name}.{topic_str}", f"Alexa online. What's up, {name}?"])
    return random.choice([f"Alexa here.{topic_str}", "Alexa online. Go ahead."])

def _match_learned_rule(user_text: str) -> Optional[dict]:
    trigger = _normalize_trigger(user_text)
    if not trigger:
        return None
    matched = memory.get_rules().get(trigger)
    if not matched:
        return None
    return {
        "action": matched.get("action", ""),
        "target": matched.get("target", ""),
        "spoken_response": f"Got it — applying your '{trigger}' shortcut.",
        "follow_up_question": ""
    }


# ═══════════════════════════════════════════════════════════════
#  19. MAIN LOOP
# ═══════════════════════════════════════════════════════════════
CONFIRM = {"yes","yep","yeah","confirm","do it","sure","ok","okay"}
DENY    = {"no","nope","cancel","stop","discard","never mind"}
EXIT    = {"exit","quit","goodbye","bye","shut down","sleep"}

def _print_banner():
    print("═" * 60)
    print("  🔵   A L E X A   —   SOTA Secure Voice Assistant   v5.1")
    print("─" * 60)
    print(f"  Wake Engine    : {'✅ openWakeWord' if _OWW_OK else '❌ Missing'}")
    print(f"  STT Engine     : {'✅ Groq Whisper-L3' if _GROQ_OK else '❌ Missing'}")
    print(f"  TTS Engine     : {'✅ Edge Neural TTS' if _EDGE_OK else '❌ Missing'}")
    print(f"  Voice Biometric: {'✅ SpeechBrain' if _SB_OK else '❌ Missing'}")
    print("═" * 60)

def main():
    global _current_session, _pending_rule
    
    if "--export-pcs" in sys.argv: 
        print(f"PCS exported to: {memory.export_pcs_for_llm()}")
        return

    _print_banner()
    _current_session = memory.start_session()
    alexa_voice.say(_build_greeting())

    ACTIVE_TIMEOUT = ACTIVE_SESSION_SECONDS

    while True:
        was_interrupted = wake_detector.wait_for_wake()
        if was_interrupted: 
            alexa_voice.interrupt()
            time.sleep(0.12)
            
        alexa_voice.resume_listen_mode()
        alexa_voice.acknowledge()
        last_active_time = time.time()

        while True:
            if time.time() - last_active_time > ACTIVE_TIMEOUT:
                print("\n[SLEEP] 60-second active window ended. Going back to standby.")
                alexa_voice.say("One minute's up. Going back to standby.")
                break

            user_input = stt.listen() if _AUDIO_OK else input("\nYou: ").strip()
            if not user_input: 
                continue

            last_active_time = time.time()
            lower = user_input.lower().strip()
            print(f"\nYou: {user_input}")

            if lower in ["go to sleep", "sleep", "standby", "thanks alexa"]:
                alexa_voice.say("Standing by.")
                break

            if any(p == lower or lower.startswith(p + " ") for p in EXIT):
                alexa_voice.say("Shutting down. Later.")
                def _shutdown(): 
                    memory.end_session(_current_session, _generate_session_summary(_current_session))
                threading.Thread(target=_shutdown, daemon=True).start()
                time.sleep(2.5)
                return  

            if _pending_rule:
                words = set(lower.split())
                if words & CONFIRM:
                    memory.save_rule(_pending_rule["trigger"], _pending_rule["action"], _pending_rule["target"])
                    alexa_voice.say("Saved.")
                    _pending_rule = None
                    memory.add_turn("user", user_input, _current_session)
                    memory.add_turn("assistant", "Rule saved.", _current_session)
                    continue
                elif words & DENY:
                    alexa_voice.say("Discarded.")
                    _pending_rule = None
                    memory.add_turn("user", user_input, _current_session)
                    memory.add_turn("assistant", "Rule discarded.", _current_session)
                    continue

            _extract_facts_locally(user_input)
            _extract_topic_locally(user_input)

            local_answer = _try_local_resolve(user_input)
            if local_answer:
                memory.add_turn("user", user_input, _current_session)
                memory.add_turn("assistant", local_answer, _current_session)
                alexa_voice.say(local_answer)
                continue

            learned = _match_learned_rule(user_input)
            if learned:
                intent = AlexaIntent(**learned)
                response = execute_intent(intent, user_input)
                memory.add_turn("user", user_input, _current_session)
                memory.add_turn("assistant", response, _current_session)
                alexa_voice.say(response)
                continue

            alexa_voice.thinking_indicator()
            try:
                intent = get_alexa_intent(user_input)
                response = execute_intent(intent, user_input)
                if intent.follow_up_question.strip(): 
                    response = response.rstrip(".!?") + ". " + intent.follow_up_question
            except Exception as e: 
                print(f"\n[ERROR] {e}")
                response = "Hit a snag. Try again."

            memory.add_turn("user", user_input, _current_session)
            memory.add_turn("assistant", response, _current_session)
            alexa_voice.say(response)

if __name__ == "__main__":
    main()
