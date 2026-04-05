from __future__ import annotations

import hashlib
import json
import random
import re
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple


_WORD_RE = re.compile(r"[a-zA-Z']+")
_SPACE_RE = re.compile(r"\s+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_PUNCT_TAIL_RE = re.compile(r"[.!?]+$")
_FILLER_OPENERS_RE = re.compile(
    r"^(?:great question[,! ]*|certainly[,! ]*|absolutely[,! ]*|of course[,! ]*|sure[,! ]*)",
    flags=re.IGNORECASE,
)


def _normalize_spaces(text: str) -> str:
    return _SPACE_RE.sub(" ", (text or "").strip())


def _normalize_key_text(text: str) -> str:
    lowered = (text or "").lower().strip()
    lowered = re.sub(r"[^a-z0-9\s']", " ", lowered)
    return _normalize_spaces(lowered)


def _tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text or "")]


def _sentences(text: str) -> List[str]:
    clean = _normalize_spaces(text)
    if not clean:
        return []
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(clean) if p.strip()]
    return parts if parts else [clean]


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class UserSignals:
    text: str
    token_count: int
    has_question: bool
    urgency_score: float
    frustration_score: float
    affection_score: float
    humor_score: float
    uncertainty_score: float
    technicality_score: float
    explicit_briefness: bool
    explicit_depth: bool
    asks_for_speed: bool
    asks_for_natural_voice: bool
    asks_for_follow_up: bool
    asks_to_skip_follow_up: bool
    sentiment_hint: str
    dominant_mode: str


@dataclass
class StyleDirective:
    target_sentence_min: int = 2
    target_sentence_max: int = 4
    pace: str = "steady"
    tone: str = "warm"
    use_acknowledgement: bool = False
    acknowledgement: str = ""
    ask_follow_up: bool = False
    follow_up_mode: str = "optional"
    avoid_hype: bool = True
    avoid_padding: bool = True
    add_softeners: bool = False
    use_contractions: bool = True
    keep_direct: bool = False
    short_response: bool = False
    depth_response: bool = False
    interruption_friendly: bool = True


@dataclass
class SessionPulse:
    session_id: str
    turn_count: int = 0
    frustration_momentum: float = 0.0
    warmth_momentum: float = 0.0
    technical_momentum: float = 0.0
    last_user_text: str = ""
    last_assistant_text: str = ""
    last_update_ts: float = field(default_factory=time.time)
    recent_topics: Deque[str] = field(default_factory=lambda: deque(maxlen=12))
    recent_keywords: Deque[str] = field(default_factory=lambda: deque(maxlen=20))
    recent_modes: Deque[str] = field(default_factory=lambda: deque(maxlen=10))


@dataclass
class IntentCacheEntry:
    key: str
    payload: Dict[str, Any]
    created_at: float


class _LRUIntentCache:
    def __init__(self, max_size: int = 256):
        self.max_size = max(16, max_size)
        self._data: "OrderedDict[str, IntentCacheEntry]" = OrderedDict()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        entry = self._data.get(key)
        if not entry:
            return None
        self._data.move_to_end(key)
        return dict(entry.payload)

    def set(self, key: str, payload: Dict[str, Any]) -> None:
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = IntentCacheEntry(key=key, payload=dict(payload), created_at=time.time())
        if len(self._data) > self.max_size:
            self._data.popitem(last=False)


class ConversationQualityEngine:
    def __init__(self) -> None:
        self._intent_cache = _LRUIntentCache(max_size=240)
        self._session_state: Dict[str, SessionPulse] = {}
        self._contraction_map = {
            "do not": "don't",
            "does not": "doesn't",
            "did not": "didn't",
            "cannot": "can't",
            "can not": "can't",
            "will not": "won't",
            "would not": "wouldn't",
            "should not": "shouldn't",
            "could not": "couldn't",
            "i am": "I'm",
            "i have": "I've",
            "i will": "I'll",
            "i would": "I'd",
            "you are": "you're",
            "you will": "you'll",
            "you have": "you've",
            "we are": "we're",
            "we have": "we've",
            "we will": "we'll",
            "that is": "that's",
            "there is": "there's",
            "it is": "it's",
            "let us": "let's",
        }
        self._frustration_terms = {
            "annoying",
            "frustrating",
            "broken",
            "stupid",
            "bad",
            "terrible",
            "worst",
            "hate",
            "lag",
            "slow",
            "not working",
            "still listening",
            "robotic",
            "useless",
            "waste",
            "wrong",
        }
        self._affection_terms = {
            "thanks",
            "thank you",
            "nice",
            "awesome",
            "great",
            "love",
            "helpful",
            "appreciate",
            "cool",
            "perfect",
            "beautiful",
            "excellent",
        }
        self._humor_terms = {
            "haha",
            "lol",
            "lmao",
            "funny",
            "joke",
            "kidding",
            "tease",
            "sarcastic",
            "banter",
            "roast",
        }
        self._uncertainty_terms = {
            "maybe",
            "not sure",
            "i think",
            "probably",
            "possibly",
            "kinda",
            "sort of",
            "idk",
            "unsure",
            "wonder",
            "guess",
        }
        self._urgency_terms = {
            "now",
            "quickly",
            "asap",
            "urgent",
            "immediately",
            "hurry",
            "right away",
            "fast",
            "quick",
            "speed up",
            "faster",
            "lag",
            "delay",
        }
        self._technical_terms = {
            "api",
            "latency",
            "cache",
            "thread",
            "loop",
            "prompt",
            "json",
            "database",
            "sql",
            "voice model",
            "neural",
            "tts",
            "stt",
            "interrupt",
            "queue",
            "session",
            "integration",
            "inference",
            "pipeline",
            "streaming",
            "buffer",
            "chunk",
            "token",
        }
        self._brief_terms = {
            "short",
            "brief",
            "in one line",
            "just answer",
            "quick answer",
            "no details",
            "concise",
            "tldr",
        }
        self._depth_terms = {
            "detailed",
            "deep",
            "in depth",
            "full explanation",
            "step by step",
            "comprehensive",
            "thorough",
            "complete",
            "elaborate",
        }
        self._follow_up_terms = {
            "ask me",
            "question back",
            "follow up",
            "clarify with me",
            "interactive",
            "conversational",
        }
        self._skip_follow_up_terms = {
            "don't ask",
            "no questions",
            "no follow up",
            "just answer",
            "skip follow up",
        }
        self._natural_voice_terms = {
            "human",
            "natural",
            "not robotic",
            "less robotic",
            "interactive",
            "conversational",
            "more natural",
            "voice quality",
            "carry conversation",
        }
        self._direct_openers = [
            "Got it.",
            "Okay.",
            "Sure.",
            "Alright.",
            "Makes sense.",
            "I hear you.",
        ]
        self._empathetic_openers = [
            "You're right.",
            "I hear you.",
            "Fair point.",
            "Totally valid.",
            "That makes sense.",
            "You're not wrong.",
            "I get why that feels rough.",
            "Thanks for calling that out.",
            "That's a solid catch.",
            "You're spot on.",
        ]
        self._frustration_softeners = [
            "Let's clean that up.",
            "I'll tighten this up.",
            "We can fix that quickly.",
            "I'll keep this focused and practical.",
            "Let's make it feel smoother.",
        ]
        self._follow_up_templates = [
            "Want me to keep replies short while we tune this?",
            "Should I optimize for speed first or voice quality first?",
            "Do you want a more calm tone or a more energetic one?",
            "Want me to keep this to one suggestion per turn?",
            "Should I prioritize fewer words or more detail?",
            "Want follow-up questions every turn or only when needed?",
        ]
        self._smalltalk_fast_responses: Dict[str, List[str]] = {
            "greeting": [
                "Hey — I'm here.",
                "Hi. Ready when you are.",
                "Hey, what's up?",
                "Hello. Let's do this.",
                "Hey there.",
                "Hi, good to see you.",
                "Hey, I'm listening.",
                "Hi — what do you need?",
                "Hey. How can I help?",
                "Hello. What's on your mind?",
            ],
            "thanks": [
                "Anytime.",
                "You're welcome.",
                "Glad that helped.",
                "Happy to help.",
                "No problem.",
                "You got it.",
                "Always.",
                "Of course.",
                "Any time.",
                "Happy I could help.",
            ],
            "status_ok": [
                "I'm running smoothly.",
                "All good on my side.",
                "Yep, I'm online and ready.",
                "I'm up and responsive.",
                "Everything looks stable right now.",
                "I'm good to go.",
                "I'm here and ready.",
                "I'm running fine.",
                "All systems okay.",
                "I'm active and listening.",
            ],
            "farewell": [
                "Got it. I'll be here when you need me.",
                "Sounds good. Catch you in a bit.",
                "Okay — see you soon.",
                "Alright, talk soon.",
                "Sure thing. I'm on standby.",
                "Cool, I'll stay ready.",
                "No problem. Ping me anytime.",
                "See you.",
                "Talk soon.",
                "Standing by.",
            ],
            "affirmation": [
                "Yep.",
                "Absolutely.",
                "Yes.",
                "For sure.",
                "Definitely.",
                "Exactly.",
                "That's right.",
                "100%.",
                "Yeah.",
                "Correct.",
            ],
            "negative_ack": [
                "Okay, skipping that.",
                "Got it, I won't do that.",
                "Understood. I'll leave it.",
                "No problem, we can avoid that.",
                "Fair enough, I'll skip it.",
                "Alright, not doing that.",
                "Makes sense. We'll leave it.",
                "Done — I won't include it.",
                "Okay, thanks for clarifying.",
                "Got it.",
            ],
        }
        self._stopword_set = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "for",
            "of",
            "to",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "on",
            "in",
            "at",
            "with",
            "as",
            "that",
            "this",
            "it",
            "its",
            "i",
            "you",
            "we",
            "they",
            "he",
            "she",
            "them",
            "my",
            "your",
            "our",
            "their",
            "me",
            "us",
            "do",
            "does",
            "did",
            "can",
            "could",
            "will",
            "would",
            "should",
            "have",
            "has",
            "had",
            "if",
            "then",
            "there",
            "here",
            "from",
            "by",
            "about",
            "just",
            "only",
            "not",
            "no",
            "yes",
            "okay",
            "ok",
            "hey",
            "hi",
            "hello",
        }

    def _contains_any_phrase(self, lowered_text: str, phrase_set: Iterable[str]) -> bool:
        for phrase in phrase_set:
            if phrase in lowered_text:
                return True
        return False

    def _score_phrase_hits(self, lowered_text: str, phrase_set: Iterable[str], weight: float = 1.0) -> float:
        hits = 0
        for phrase in phrase_set:
            if phrase in lowered_text:
                hits += 1
        if hits == 0:
            return 0.0
        return hits * weight

    def detect_user_signals(self, text: str) -> UserSignals:
        clean = _normalize_spaces(text)
        lowered = clean.lower()
        tokens = _tokenize(clean)
        token_count = len(tokens)
        has_question = "?" in clean or lowered.startswith(("what", "why", "how", "when", "where", "who", "which", "can ", "could "))

        frustration_raw = self._score_phrase_hits(lowered, self._frustration_terms, weight=0.17)
        affection_raw = self._score_phrase_hits(lowered, self._affection_terms, weight=0.12)
        humor_raw = self._score_phrase_hits(lowered, self._humor_terms, weight=0.15)
        uncertainty_raw = self._score_phrase_hits(lowered, self._uncertainty_terms, weight=0.14)
        urgency_raw = self._score_phrase_hits(lowered, self._urgency_terms, weight=0.2)
        technical_raw = self._score_phrase_hits(lowered, self._technical_terms, weight=0.08)

        frustration_score = _clamp(frustration_raw + (0.18 if "!" in clean and frustration_raw > 0 else 0.0), 0.0, 1.0)
        affection_score = _clamp(affection_raw, 0.0, 1.0)
        humor_score = _clamp(humor_raw, 0.0, 1.0)
        uncertainty_score = _clamp(uncertainty_raw, 0.0, 1.0)
        urgency_score = _clamp(urgency_raw + (0.12 if "now" in lowered else 0.0), 0.0, 1.0)
        technicality_score = _clamp(technical_raw + (0.14 if token_count > 18 else 0.0), 0.0, 1.0)

        explicit_briefness = self._contains_any_phrase(lowered, self._brief_terms)
        explicit_depth = self._contains_any_phrase(lowered, self._depth_terms)
        asks_for_speed = urgency_score > 0.28 or ("lag" in lowered) or ("slow" in lowered) or ("faster" in lowered)
        asks_for_natural_voice = self._contains_any_phrase(lowered, self._natural_voice_terms)
        asks_for_follow_up = self._contains_any_phrase(lowered, self._follow_up_terms)
        asks_to_skip_follow_up = self._contains_any_phrase(lowered, self._skip_follow_up_terms)

        sentiment_hint = "neutral"
        if frustration_score >= 0.3 and frustration_score > affection_score:
            sentiment_hint = "frustrated"
        elif affection_score >= 0.24 and affection_score > frustration_score:
            sentiment_hint = "positive"
        elif uncertainty_score >= 0.25:
            sentiment_hint = "uncertain"

        dominant_mode = "chat"
        if technicality_score >= 0.4:
            dominant_mode = "technical"
        elif asks_for_speed:
            dominant_mode = "speed"
        elif asks_for_natural_voice:
            dominant_mode = "voice_quality"
        elif has_question:
            dominant_mode = "question"

        return UserSignals(
            text=clean,
            token_count=token_count,
            has_question=has_question,
            urgency_score=urgency_score,
            frustration_score=frustration_score,
            affection_score=affection_score,
            humor_score=humor_score,
            uncertainty_score=uncertainty_score,
            technicality_score=technicality_score,
            explicit_briefness=explicit_briefness,
            explicit_depth=explicit_depth,
            asks_for_speed=asks_for_speed,
            asks_for_natural_voice=asks_for_natural_voice,
            asks_for_follow_up=asks_for_follow_up,
            asks_to_skip_follow_up=asks_to_skip_follow_up,
            sentiment_hint=sentiment_hint,
            dominant_mode=dominant_mode,
        )

    def _get_or_create_pulse(self, session_id: str) -> SessionPulse:
        key = session_id or "_global"
        pulse = self._session_state.get(key)
        if pulse is None:
            pulse = SessionPulse(session_id=key)
            self._session_state[key] = pulse
        return pulse

    def update_session_pulse(
        self,
        session_id: str,
        user_text: str,
        assistant_text: str = "",
        topic_hints: Optional[Sequence[str]] = None,
    ) -> SessionPulse:
        pulse = self._get_or_create_pulse(session_id)
        signals = self.detect_user_signals(user_text)
        pulse.turn_count += 1
        pulse.frustration_momentum = _clamp((pulse.frustration_momentum * 0.75) + (signals.frustration_score * 0.45), 0.0, 1.0)
        pulse.warmth_momentum = _clamp((pulse.warmth_momentum * 0.78) + (signals.affection_score * 0.42), 0.0, 1.0)
        pulse.technical_momentum = _clamp((pulse.technical_momentum * 0.8) + (signals.technicality_score * 0.5), 0.0, 1.0)
        pulse.last_user_text = user_text or ""
        if assistant_text:
            pulse.last_assistant_text = assistant_text
        pulse.last_update_ts = time.time()
        pulse.recent_modes.append(signals.dominant_mode)

        token_pool = [tok for tok in _tokenize(user_text) if len(tok) > 3 and tok not in self._stopword_set]
        for tok in token_pool[:4]:
            pulse.recent_keywords.append(tok)
        if topic_hints:
            for t in topic_hints:
                t_clean = _normalize_key_text(t)
                if t_clean:
                    pulse.recent_topics.append(t_clean)
        return pulse

    def style_for_turn(self, signals: UserSignals, pulse: Optional[SessionPulse] = None) -> StyleDirective:
        directive = StyleDirective()
        if signals.explicit_briefness:
            directive.target_sentence_min = 1
            directive.target_sentence_max = 2
            directive.short_response = True
            directive.keep_direct = True
        if signals.explicit_depth:
            directive.target_sentence_min = 3
            directive.target_sentence_max = 6
            directive.depth_response = True
        if signals.asks_for_speed or signals.urgency_score >= 0.36:
            directive.pace = "quick"
            directive.keep_direct = True
            directive.target_sentence_max = min(directive.target_sentence_max, 3)
        if signals.asks_for_natural_voice:
            directive.tone = "natural"
            directive.use_contractions = True
            directive.add_softeners = True
        if signals.frustration_score >= 0.3:
            directive.use_acknowledgement = True
            directive.acknowledgement = random.choice(self._empathetic_openers)
            directive.keep_direct = True
        if signals.uncertainty_score >= 0.28 and not signals.explicit_briefness:
            directive.ask_follow_up = True
            directive.follow_up_mode = "clarify"
        if signals.asks_to_skip_follow_up:
            directive.ask_follow_up = False
        if signals.asks_for_follow_up:
            directive.ask_follow_up = True
            directive.follow_up_mode = "engage"

        if pulse:
            if pulse.frustration_momentum >= 0.42:
                directive.use_acknowledgement = True
                if not directive.acknowledgement:
                    directive.acknowledgement = random.choice(self._empathetic_openers)
                directive.keep_direct = True
                directive.target_sentence_max = min(directive.target_sentence_max, 3)
            if pulse.technical_momentum >= 0.38 and not signals.explicit_briefness:
                directive.depth_response = True
                directive.target_sentence_min = max(directive.target_sentence_min, 3)
            if pulse.warmth_momentum >= 0.36 and signals.frustration_score < 0.25:
                directive.tone = "friendly"
        return directive

    def choose_ack(self, signals: UserSignals, pulse: Optional[SessionPulse]) -> str:
        if signals.frustration_score >= 0.28 or (pulse and pulse.frustration_momentum >= 0.4):
            return random.choice(self._empathetic_openers)
        if signals.urgency_score >= 0.34:
            return random.choice(self._direct_openers)
        if signals.affection_score >= 0.24:
            return random.choice(["Appreciate that.", "Nice.", "Glad to hear it."])
        return ""

    def _extract_keywords(self, text: str, limit: int = 10) -> List[str]:
        seen = set()
        out: List[str] = []
        for tok in _tokenize(text):
            if len(tok) < 4:
                continue
            if tok in self._stopword_set:
                continue
            if tok in seen:
                continue
            seen.add(tok)
            out.append(tok)
            if len(out) >= limit:
                break
        return out

    def _score_context_turn(self, user_text: str, role: str, content: str, idx_from_end: int) -> float:
        user_keywords = set(self._extract_keywords(user_text, limit=12))
        turn_keywords = set(self._extract_keywords(content, limit=18))
        overlap = len(user_keywords & turn_keywords)
        recency_bonus = max(0.0, 1.2 - (idx_from_end * 0.12))
        role_bonus = 0.15 if role == "assistant" else 0.2
        question_bonus = 0.14 if "?" in content else 0.0
        token_bias = min(0.22, len(turn_keywords) * 0.015)
        return overlap * 0.45 + recency_bonus + role_bonus + question_bonus + token_bias

    def compact_context(self, turns: Sequence[Dict[str, Any]], user_text: str, max_chars: int = 2200) -> str:
        if not turns:
            return "(first message this session)"
        indexed: List[Tuple[float, Dict[str, Any]]] = []
        total = len(turns)
        for i, t in enumerate(turns):
            role = str(t.get("role", "user"))
            content = _normalize_spaces(str(t.get("content", "")))
            if not content:
                continue
            idx_from_end = total - i - 1
            score = self._score_context_turn(user_text, role, content, idx_from_end)
            indexed.append((score, {"role": role, "content": content, "idx": i}))
        indexed.sort(key=lambda x: x[0], reverse=True)
        selected: List[Dict[str, Any]] = []
        used = 0
        for _, t in indexed:
            row = f"{t['role'].capitalize()}: {t['content']}"
            row_len = len(row) + 1
            if used + row_len > max_chars:
                continue
            selected.append(t)
            used += row_len
            if used >= int(max_chars * 0.9):
                break
        if not selected:
            t = turns[-1]
            return f"{str(t.get('role', 'user')).capitalize()}: {_normalize_spaces(str(t.get('content', '')))}"
        selected.sort(key=lambda x: x["idx"])
        return "\n".join(f"{t['role'].capitalize()}: {t['content']}" for t in selected)

    def _intent_cache_key(self, user_text: str, compact_ctx: str, rules_json: str) -> str:
        key_payload = {
            "u": _normalize_key_text(user_text),
            "c": _normalize_key_text(compact_ctx)[:1200],
            "r": _normalize_key_text(rules_json)[:1000],
        }
        encoded = json.dumps(key_payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def get_cached_intent(self, user_text: str, compact_ctx: str, rules_json: str) -> Optional[Dict[str, Any]]:
        key = self._intent_cache_key(user_text, compact_ctx, rules_json)
        return self._intent_cache.get(key)

    def set_cached_intent(self, user_text: str, compact_ctx: str, rules_json: str, payload: Dict[str, Any]) -> None:
        key = self._intent_cache_key(user_text, compact_ctx, rules_json)
        if not payload:
            return
        if str(payload.get("action", "")).lower() in {"chat", "get_system_info", "open_website", "search_website"}:
            self._intent_cache.set(key, payload)

    def _looks_like_greeting(self, lowered_text: str) -> bool:
        return bool(re.fullmatch(r"(hey+|hi+|hello+|yo+|sup+|good (morning|afternoon|evening))([.! ]*)", lowered_text))

    def _looks_like_thanks(self, lowered_text: str) -> bool:
        patterns = (
            r"thanks[.! ]*",
            r"thank you[.! ]*",
            r"appreciate it[.! ]*",
            r"thx[.! ]*",
        )
        return any(re.fullmatch(p, lowered_text) for p in patterns)

    def _looks_like_status_check(self, lowered_text: str) -> bool:
        pats = (
            r"are you (there|online|alive|up|working)[? ]*",
            r"you there[? ]*",
            r"status[? ]*",
            r"are you okay[? ]*",
            r"all good[? ]*",
        )
        return any(re.fullmatch(p, lowered_text) for p in pats)

    def _looks_like_farewell(self, lowered_text: str) -> bool:
        pats = (
            r"bye[.! ]*",
            r"goodbye[.! ]*",
            r"talk (later|soon)[.! ]*",
            r"see you[.! ]*",
            r"catch you later[.! ]*",
            r"i'?m done[.! ]*",
        )
        return any(re.fullmatch(p, lowered_text) for p in pats)

    def _looks_like_simple_affirmation(self, lowered_text: str) -> bool:
        return bool(re.fullmatch(r"(yes|yeah|yep|sure|ok|okay|affirmative|correct)[.! ]*", lowered_text))

    def _looks_like_simple_negation(self, lowered_text: str) -> bool:
        return bool(re.fullmatch(r"(no|nope|nah|negative|not now|skip it)[.! ]*", lowered_text))

    def fast_intent(self, user_text: str, session_id: str = "") -> Optional[Dict[str, Any]]:
        clean = _normalize_spaces(user_text)
        if not clean:
            return None
        lowered = clean.lower()
        if len(lowered) > 80:
            return None
        pulse = self._get_or_create_pulse(session_id or "_global")
        if self._looks_like_greeting(lowered):
            return {"action": "chat", "spoken_response": random.choice(self._smalltalk_fast_responses["greeting"]), "follow_up_question": ""}
        if self._looks_like_thanks(lowered):
            return {"action": "chat", "spoken_response": random.choice(self._smalltalk_fast_responses["thanks"]), "follow_up_question": ""}
        if self._looks_like_status_check(lowered):
            return {"action": "chat", "spoken_response": random.choice(self._smalltalk_fast_responses["status_ok"]), "follow_up_question": ""}
        if self._looks_like_farewell(lowered):
            return {"action": "chat", "spoken_response": random.choice(self._smalltalk_fast_responses["farewell"]), "follow_up_question": ""}
        if self._looks_like_simple_affirmation(lowered):
            mode = "affirmation"
            if pulse.frustration_momentum >= 0.45:
                return {"action": "chat", "spoken_response": "Good call — let's keep going.", "follow_up_question": ""}
            return {"action": "chat", "spoken_response": random.choice(self._smalltalk_fast_responses[mode]), "follow_up_question": ""}
        if self._looks_like_simple_negation(lowered):
            return {"action": "chat", "spoken_response": random.choice(self._smalltalk_fast_responses["negative_ack"]), "follow_up_question": ""}
        return None

    def build_prompt_overlay(
        self,
        user_text: str,
        turns: Sequence[Dict[str, Any]],
        session_id: str = "",
    ) -> str:
        signals = self.detect_user_signals(user_text)
        pulse = self._get_or_create_pulse(session_id or "_global")
        directive = self.style_for_turn(signals, pulse)
        keywords = self._extract_keywords(user_text, limit=7)
        continuity = ", ".join(list(pulse.recent_topics)[-3:] + list(pulse.recent_keywords)[-4:])
        continuity = continuity[:200]
        block = [
            "### DYNAMIC VOICE/STYLE DIRECTIVE",
            f"- user_mode: {signals.dominant_mode}",
            f"- sentiment_hint: {signals.sentiment_hint}",
            f"- urgency_score: {signals.urgency_score:.2f}",
            f"- frustration_score: {signals.frustration_score:.2f}",
            f"- technicality_score: {signals.technicality_score:.2f}",
            f"- target_sentence_min: {directive.target_sentence_min}",
            f"- target_sentence_max: {directive.target_sentence_max}",
            f"- tone: {directive.tone}",
            f"- pace: {directive.pace}",
            f"- short_response: {str(directive.short_response).lower()}",
            f"- depth_response: {str(directive.depth_response).lower()}",
            f"- ask_follow_up: {str(directive.ask_follow_up).lower()}",
            f"- keep_direct: {str(directive.keep_direct).lower()}",
            f"- avoid_hype: {str(directive.avoid_hype).lower()}",
            f"- avoid_padding: {str(directive.avoid_padding).lower()}",
            f"- use_contractions: {str(directive.use_contractions).lower()}",
            f"- user_keywords: {', '.join(keywords) if keywords else '(none)'}",
            f"- continuity_hints: {continuity if continuity else '(none)'}",
            "- do_not_repeat_user_verbatim: true",
            "- natural_variation: true",
        ]
        if signals.asks_for_natural_voice:
            block.append("- user_requested_human_like_voice: true")
        if signals.asks_for_speed:
            block.append("- prioritize_latency_style: true")
        if turns:
            last_assistant = ""
            for t in reversed(turns):
                if str(t.get("role", "")) == "assistant":
                    last_assistant = _normalize_spaces(str(t.get("content", "")))
                    if last_assistant:
                        break
            if last_assistant:
                block.append(f"- avoid_recent_assistant_phrase: {last_assistant[:160]}")
        return "\n".join(block) + "\n"

    def _apply_contractions(self, text: str) -> str:
        out = text
        for long, short in self._contraction_map.items():
            out = re.sub(rf"\b{re.escape(long)}\b", short, out, flags=re.IGNORECASE)
        return out

    def _trim_repetition(self, text: str) -> str:
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if not lines:
            return ""
        dedup: List[str] = []
        seen = set()
        for line in lines:
            key = _normalize_key_text(line)
            if key in seen:
                continue
            seen.add(key)
            dedup.append(line)
        return " ".join(dedup).strip()

    def _remove_hollow_openers(self, text: str) -> str:
        t = _normalize_spaces(text)
        if not t:
            return t
        cleaned = _FILLER_OPENERS_RE.sub("", t).strip()
        if cleaned:
            return cleaned[0].upper() + cleaned[1:] if len(cleaned) > 1 else cleaned.upper()
        return t

    def _bound_sentence_count(self, text: str, min_sent: int, max_sent: int) -> str:
        sents = _sentences(text)
        if not sents:
            return ""
        if len(sents) > max_sent:
            sents = sents[:max_sent]
        if len(sents) < min_sent and len(sents) == 1:
            first = sents[0]
            if not _PUNCT_TAIL_RE.search(first):
                first += "."
            if min_sent >= 2 and len(first) < 120:
                sents = [first, "I can expand if you want more detail."]
        return " ".join(sents).strip()

    def _ensure_terminal_punct(self, text: str) -> str:
        t = text.strip()
        if not t:
            return t
        if _PUNCT_TAIL_RE.search(t):
            return t
        return t + "."

    def _apply_empathy_prefix(self, text: str, directive: StyleDirective, signals: UserSignals) -> str:
        if not directive.use_acknowledgement:
            return text
        prefix = directive.acknowledgement or self.choose_ack(signals, None)
        if not prefix:
            return text
        if text.lower().startswith(prefix.lower()):
            return text
        return f"{prefix} {text}".strip()

    def _compress_padding(self, text: str) -> str:
        t = text
        removals = [
            "Let me know if you need anything else.",
            "Let me know if you need anything more.",
            "Please let me know if you have any questions.",
            "I hope that helps.",
            "Hope that helps.",
            "Feel free to ask anything else.",
            "If you want, I can also",
            "I would be happy to",
        ]
        for chunk in removals:
            t = re.sub(re.escape(chunk), "", t, flags=re.IGNORECASE)
        return _normalize_spaces(t)

    def _inject_natural_variation(self, text: str, directive: StyleDirective, signals: UserSignals) -> str:
        t = text
        if directive.keep_direct and signals.urgency_score >= 0.3:
            t = t.replace("I can", "I can quickly")
        if directive.tone in {"natural", "friendly"} and signals.frustration_score < 0.3:
            t = t.replace("I will ", "I'll ")
        if directive.add_softeners and signals.frustration_score >= 0.22:
            softener = random.choice(self._frustration_softeners)
            if softener.lower() not in t.lower():
                t = f"{softener} {t}"
        return _normalize_spaces(t)

    def refine_response(
        self,
        text: str,
        user_text: str,
        session_id: str = "",
        force_short: bool = False,
    ) -> str:
        clean = _normalize_spaces(text)
        if not clean:
            return clean
        signals = self.detect_user_signals(user_text)
        pulse = self._get_or_create_pulse(session_id or "_global")
        directive = self.style_for_turn(signals, pulse)
        if force_short:
            directive.target_sentence_min = 1
            directive.target_sentence_max = 2
            directive.short_response = True
            directive.keep_direct = True
        out = clean
        out = self._trim_repetition(out)
        out = self._remove_hollow_openers(out)
        if directive.use_contractions:
            out = self._apply_contractions(out)
        out = self._inject_natural_variation(out, directive, signals)
        if directive.avoid_padding:
            out = self._compress_padding(out)
        out = self._bound_sentence_count(out, directive.target_sentence_min, directive.target_sentence_max)
        out = self._apply_empathy_prefix(out, directive, signals)
        out = self._ensure_terminal_punct(out)
        return out

    def decide_follow_up(
        self,
        response_text: str,
        user_text: str,
        existing_follow_up: str = "",
        session_id: str = "",
    ) -> str:
        if existing_follow_up and existing_follow_up.strip():
            return _normalize_spaces(existing_follow_up)
        signals = self.detect_user_signals(user_text)
        pulse = self._get_or_create_pulse(session_id or "_global")
        directive = self.style_for_turn(signals, pulse)
        if signals.asks_to_skip_follow_up:
            return ""
        if not directive.ask_follow_up:
            return ""
        if len(_tokenize(response_text)) < 8:
            return ""
        if signals.explicit_briefness and not signals.asks_for_follow_up:
            return ""
        if signals.frustration_score >= 0.36 and signals.asks_for_follow_up is False:
            return ""
        return random.choice(self._follow_up_templates)

    def to_tts_text(self, text: str, user_text: str = "", session_id: str = "") -> str:
        clean = _normalize_spaces(text)
        if not clean:
            return clean
        signals = self.detect_user_signals(user_text)
        pulse = self._get_or_create_pulse(session_id or "_global")
        directive = self.style_for_turn(signals, pulse)
        out = clean
        if directive.short_response and len(out) > 240:
            out = " ".join(_sentences(out)[:2])
        out = out.replace(" - ", ", ")
        out = out.replace("—", ", ")
        out = out.replace("...", ".")
        out = re.sub(r"\s+,", ",", out)
        out = re.sub(r"\s+\.", ".", out)
        out = _normalize_spaces(out)
        out = self._ensure_terminal_punct(out)
        return out

    def postprocess_intent_payload(
        self,
        payload: Dict[str, Any],
        user_text: str,
        session_id: str = "",
    ) -> Dict[str, Any]:
        data = dict(payload or {})
        action = str(data.get("action", "chat")).strip().lower() or "chat"
        data["action"] = action
        spoken = _normalize_spaces(str(data.get("spoken_response", "")))
        if action == "chat":
            if not spoken:
                spoken = "Got it."
            spoken = self.refine_response(spoken, user_text, session_id=session_id)
            data["spoken_response"] = spoken
            follow = self.decide_follow_up(spoken, user_text, str(data.get("follow_up_question", "")), session_id=session_id)
            data["follow_up_question"] = follow
        else:
            if spoken:
                data["spoken_response"] = self.refine_response(spoken, user_text, session_id=session_id, force_short=True)
            else:
                data["spoken_response"] = spoken
            if str(data.get("follow_up_question", "")).strip():
                data["follow_up_question"] = self.decide_follow_up(
                    str(data.get("spoken_response", "")),
                    user_text,
                    str(data.get("follow_up_question", "")),
                    session_id=session_id,
                )
        if action == "search_website":
            target = _normalize_spaces(str(data.get("target", "")))
            if target and not data.get("search_query"):
                data["search_query"] = ""
        if action == "open_website":
            target = str(data.get("target", "")).strip()
            if target and "." not in target and "http" not in target and " " not in target:
                data["target"] = f"{target}.com"
        return data

    def build_turn_summary(self, turns: Sequence[Dict[str, Any]], max_chars: int = 800) -> str:
        if not turns:
            return ""
        recent = turns[-10:]
        user_topics: List[str] = []
        assistant_actions: List[str] = []
        for t in recent:
            role = str(t.get("role", ""))
            content = _normalize_spaces(str(t.get("content", "")))
            if not content:
                continue
            keywords = self._extract_keywords(content, limit=3)
            if role == "user" and keywords:
                user_topics.extend(keywords[:2])
            elif role == "assistant" and keywords:
                assistant_actions.extend(keywords[:2])
        user_topics = list(dict.fromkeys(user_topics))[:8]
        assistant_actions = list(dict.fromkeys(assistant_actions))[:8]
        summary = f"user_topics={', '.join(user_topics) if user_topics else '(none)'}; assistant_focus={', '.join(assistant_actions) if assistant_actions else '(none)'}"
        return summary[:max_chars]

    def infer_topic_hints(self, text: str) -> List[str]:
        tokens = self._extract_keywords(text, limit=10)
        hints: List[str] = []
        for tok in tokens:
            if tok in self._technical_terms:
                hints.append("technical")
            elif tok in {"music", "song", "spotify", "playlist"}:
                hints.append("music")
            elif tok in {"youtube", "video", "watch"}:
                hints.append("video")
            elif tok in {"time", "date", "battery", "cpu", "ram"}:
                hints.append("system")
            elif tok in {"voice", "speak", "response", "conversation"}:
                hints.append("conversation_quality")
            else:
                hints.append(tok)
        uniq = list(dict.fromkeys(hints))
        return uniq[:6]

    def should_show_thinking_indicator(self, user_text: str) -> bool:
        clean = _normalize_spaces(user_text)
        if not clean:
            return False
        if len(clean) <= 16 and not ("?" in clean):
            return False
        lowered = clean.lower()
        if self._looks_like_greeting(lowered) or self._looks_like_thanks(lowered):
            return False
        if self._looks_like_status_check(lowered):
            return False
        return True

    def fast_response_text(self, user_text: str, session_id: str = "") -> Optional[str]:
        fast = self.fast_intent(user_text, session_id=session_id)
        if not fast:
            return None
        text = str(fast.get("spoken_response", "")).strip()
        if not text:
            return None
        return self.to_tts_text(text, user_text=user_text, session_id=session_id)

    def update_after_assistant_reply(
        self,
        session_id: str,
        user_text: str,
        assistant_text: str,
    ) -> None:
        topic_hints = self.infer_topic_hints(user_text)
        self.update_session_pulse(
            session_id=session_id or "_global",
            user_text=user_text,
            assistant_text=assistant_text,
            topic_hints=topic_hints,
        )

    def adaptive_temperature(self, user_text: str, default_temperature: float = 0.32) -> float:
        signals = self.detect_user_signals(user_text)
        temp = default_temperature
        if signals.asks_for_speed or signals.explicit_briefness:
            temp -= 0.06
        if signals.asks_for_natural_voice and not signals.asks_for_speed:
            temp += 0.04
        if signals.technicality_score >= 0.4:
            temp -= 0.03
        if signals.humor_score >= 0.25:
            temp += 0.03
        return _clamp(temp, 0.15, 0.5)

    def adaptive_max_tokens(self, user_text: str, default_max_tokens: int = 260) -> int:
        signals = self.detect_user_signals(user_text)
        out = default_max_tokens
        if signals.explicit_briefness or signals.asks_for_speed:
            out = int(out * 0.7)
        if signals.explicit_depth:
            out = int(out * 1.35)
        if signals.technicality_score >= 0.45:
            out = int(out * 1.15)
        return max(120, min(480, out))

    def build_latency_hints(self, user_text: str) -> str:
        signals = self.detect_user_signals(user_text)
        hints = [
            "### LATENCY GUIDANCE",
            "- prefer direct answer first, then one concise extension if needed",
            "- avoid enumerating unnecessary options",
            "- avoid repetitive restatements",
            "- do not add generic closers",
        ]
        if signals.asks_for_speed or signals.explicit_briefness:
            hints.append("- user_requested_speed=true")
            hints.append("- keep answer compact and decisive")
        if signals.asks_for_natural_voice:
            hints.append("- user_requested_human_like_quality=true")
            hints.append("- maintain conversational rhythm with varied sentence lengths")
        return "\n".join(hints) + "\n"

    def normalize_llm_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(data or {})
        out.setdefault("action", "chat")
        out.setdefault("target", "")
        out.setdefault("spoken_response", "")
        out.setdefault("search_query", "")
        out.setdefault("web_task", "")
        out.setdefault("new_trigger", "")
        out.setdefault("new_action", "")
        out.setdefault("new_target", "")
        out.setdefault("follow_up_question", "")
        for k, v in list(out.items()):
            if isinstance(v, str):
                out[k] = _normalize_spaces(v)
        return out

    def refine_nonchat_action_reply(self, action: str, spoken_text: str, user_text: str, session_id: str = "") -> str:
        action_norm = (action or "").strip().lower()
        text = _normalize_spaces(spoken_text)
        if not text:
            if action_norm == "open_website":
                text = "Opening it."
            elif action_norm == "search_website":
                text = "Searching now."
            elif action_norm == "play_on_youtube":
                text = "Playing that on YouTube."
            elif action_norm == "open_app":
                text = "Opening it."
            elif action_norm == "open_folder":
                text = "Opening that folder."
            elif action_norm == "get_system_info":
                text = "Here's what I found."
            elif action_norm == "web_extract":
                text = "I checked that page."
            else:
                text = "Done."
        return self.refine_response(text, user_text, session_id=session_id, force_short=True)

    def should_use_cache(self, user_text: str) -> bool:
        signals = self.detect_user_signals(user_text)
        if signals.asks_for_speed:
            return True
        if signals.token_count <= 14 and signals.technicality_score < 0.4:
            return True
        return False

    def should_force_llm(self, user_text: str) -> bool:
        signals = self.detect_user_signals(user_text)
        if signals.explicit_depth:
            return True
        if signals.technicality_score >= 0.55:
            return True
        if signals.token_count > 32:
            return True
        return False

