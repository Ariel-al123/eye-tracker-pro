"""Small local next-word helper with contextual and per-user phrase learning."""

from __future__ import annotations

import re
from collections import Counter


WORD_BANKS = {
    "en": "I you we they he she it this that the a an and or but if then because so with to from for of in on at by about my your our their can could would should will do does did am is are was were be been being have has had need want like love feel think know see hear say tell help call please thank yes no not more less now later today tomorrow here there where when what who why how okay good bad tired hungry thirsty hot cold sick pain stop start wait go come stay turn open close left right up down water food medicine doctor bathroom bed chair phone family home work name safe sorry ready finished again speak slowly repeat understand difficult easy happy sad afraid comfortable uncomfortable " .split(),
    "es": "yo tú usted nosotros ellos él ella esto eso el la un una y o pero si entonces porque así con a de para por en sobre mi tu nuestro su puedo podría quiero quisiera necesito me gusta amo siento pienso sé veo escucho digo ayuda llama por favor gracias sí no más menos ahora después hoy mañana aquí allá dónde cuándo qué quién cómo bien mal cansado cansada hambre sed calor frío enfermo enferma dolor parar empezar esperar ir venir quedarse girar abrir cerrar izquierda derecha arriba abajo agua comida medicina doctor doctora baño cama silla teléfono familia casa trabajo nombre seguro segura perdón listo lista otra vez hablar despacio repetir entiendo difícil fácil feliz triste miedo cómodo cómoda incómodo incómoda " .split(),
    "fr": "je tu vous nous ils elle ceci cela le la un une et ou mais si alors parce que donc avec à de pour par dans sur mon ton notre leur peux pourrais veux voudrais besoin me aime adore sens pense sais vois entends dis aide appelle s'il vous plaît merci oui non plus moins maintenant après aujourd'hui demain ici là où quand quoi qui comment bien mal fatigué fatiguée faim soif chaud froide malade douleur arrêter commencer attendre aller venir rester tourner ouvrir fermer gauche droite haut bas eau nourriture médicament médecin toilette lit chaise téléphone famille maison travail nom sûr sûre désolé désolée prêt prête encore parler lentement répéter comprends difficile facile heureux heureuse triste peur confortable inconfortable " .split(),
    "zh": list("我你我们他们她它这那的一个和或者但是如果因为所以与到从为在上我的你的我们的他们的可以想要喜欢爱感觉认为知道看听说帮助请谢谢是不要更多少现在以后今天明天这里那里哪里什么时候什么谁为什么怎么好坏累饿渴热冷生病疼痛停止开始等待去来留下转开关左边右边上面下面水食物药医生厕所床椅子电话家人家工作名字安全抱歉准备好了再一次说慢一点重复我明白困难简单开心难过害怕舒服不舒服"),
}

CONTEXT = {
    "en": {
        "i": "am need want would can have feel think like don't", "i am": "okay tired hungry thirsty hot cold sick in pain ready", "i need": "help water food medicine the bathroom to call someone", "i want": "water food to go home more help to sit down", "can you": "help me call someone bring water wait please", "please": "help me wait stop call come open close", "thank": "you so much", "my": "name family head arm leg back chest phone", "feel": "okay sick pain better worse hot cold tired", "call": "my family a doctor for help", "i feel": "okay sick pain tired hungry thirsty afraid happy sad", "where is": "the bathroom my family the doctor my room", "it hurts": "here there everywhere when I move",
    },
    "es": {
        "yo": "quiero necesito puedo estoy siento tengo prefiero", "yo necesito": "ayuda agua comida medicina ir al baño llamar a alguien", "yo quiero": "agua comida ir a casa más ayuda sentarme", "necesito": "ayuda agua comida medicina ir llamar descansar", "quiero": "agua comer ir a casa ayuda descansar sentarme", "por favor": "ayuda espera detente llama ven abre cierra", "me siento": "bien mal cansado cansada enfermo enferma feliz triste con dolor", "tengo": "hambre sed frío calor dolor sueño miedo", "mi": "familia nombre cabeza brazo pierna espalda teléfono", "llama": "a mi familia al doctor para pedir ayuda", "dónde está": "el baño mi familia el doctor mi habitación", "me duele": "aquí allí mucho cuando me muevo",
    },
    "fr": {
        "je": "suis veux voudrais peux pense sens besoin", "j'ai": "faim soif froid chaud mal besoin peur", "j'ai besoin": "d'aide d'eau de nourriture de médicaments d'aller aux toilettes", "je veux": "de l'eau manger rentrer chez moi de l'aide m'asseoir", "je me": "sens suis", "je me sens": "bien mal fatigué fatiguée malade heureux heureuse triste", "s'il vous plaît": "aidez-moi attendez arrêtez appelez venez ouvrez fermez", "mon": "nom téléphone bras dos ventre médecin", "ma": "famille maison tête jambe chambre", "appelez": "ma famille un médecin les secours", "où est": "la toilette ma famille le médecin ma chambre", "j'ai mal": "ici là partout quand je bouge",
    },
    "zh": {
        "我": "要 想 可以 需要 是 感觉", "我需要": "帮助 水 食物 药 去厕所 休息", "我想": "喝水 吃东西 回家 坐下 休息", "请": "帮助 等一下 停下来 打电话 过来", "我感觉": "很好 不舒服 疼 累 饿 渴 害怕 开心", "我的": "家人 名字 头 手 腿 电话", "帮我": "叫医生 打电话 拿水 等一下", "哪里有": "厕所 水 医生 家人",
    },
}

TOKEN_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)


class SuggestionEngine:
    def __init__(self, learned: dict[str, dict[str, int]] | None = None):
        self.learned = {
            str(context): Counter({str(word): int(count) for word, count in words.items()})
            for context, words in (learned or {}).items()
            if isinstance(words, dict)
        }
        self._completed_tokens: list[str] = []

    def _tokens(self, text: str) -> list[str]:
        return [token.lower() for token in TOKEN_RE.findall(text)]

    def suggest(self, text: str, language: str = "en", limit: int = 5) -> list[str]:
        lang = language if language in WORD_BANKS else "en"
        if lang == "zh":
            compact = text.rstrip()
            context_char = compact[-1:] if compact else ""
            previous_char = compact[-2:] if len(compact) > 1 else ""
            context_map = CONTEXT["zh"]
            learned_words: list[str] = []
            for key in (previous_char, context_char):
                counter = self.learned.get(f"zh:{key}")
                if counter:
                    learned_words.extend(word for word, _ in counter.most_common())
            candidates = learned_words + context_map.get(previous_char, "").split() + context_map.get(context_char, "").split() + WORD_BANKS["zh"]
            return list(dict.fromkeys(candidates))[:max(1, limit)]
        raw = text.rstrip()
        has_partial_word = bool(raw) and not text[-1:].isspace() and bool(TOKEN_RE.search(raw))
        tokens = self._tokens(raw)
        prefix = tokens[-1] if has_partial_word and tokens else ""
        context_tokens = tokens[:-1] if prefix else tokens
        context_map = CONTEXT.get(lang, CONTEXT["en"])
        context_words: list[str] = []
        for width in (2, 1):
            if len(context_tokens) >= width:
                key = " ".join(context_tokens[-width:])
                context_words.extend(context_map.get(key, "").split())

        learned_words: list[str] = []
        for width in (2, 1):
            if len(context_tokens) >= width:
                key = " ".join(context_tokens[-width:])
                counter = self.learned.get(f"{lang}:{key}")
                if counter:
                    learned_words.extend(word for word, _ in counter.most_common())

        ordered = learned_words + context_words + WORD_BANKS[lang]
        result: list[str] = []
        seen: set[str] = set()
        for candidate in ordered:
            normalized = candidate.lower()
            if normalized in seen or (prefix and not normalized.startswith(prefix)):
                continue
            seen.add(normalized)
            result.append(candidate)
            if len(result) >= max(1, limit):
                break
        return result

    def learn_selection(self, prior_text: str, selected: str, language: str = "en") -> None:
        trimmed = prior_text.rstrip()
        tokens = self._tokens(trimmed)
        if trimmed and not prior_text[-1:].isspace() and tokens:
            tokens = tokens[:-1]
        context = " ".join(tokens[-2:]) if len(tokens) >= 2 else (tokens[-1] if tokens else "")
        first_word = self._tokens(selected)
        if not context or not first_word:
            return
        key = f"{language}:{context}"
        self.learned.setdefault(key, Counter())[first_word[0]] += 1

    def learn_completed_text(self, text: str, language: str = "en") -> None:
        # Learn from words the user completes with Space, without relearning older words.
        if not text.endswith((" ", "\n", "\t")):
            return
        tokens = self._tokens(text)
        if tokens[:len(self._completed_tokens)] != self._completed_tokens:
            self._completed_tokens = []
        for index in range(len(self._completed_tokens), len(tokens)):
            if index == 0:
                continue
            context = " ".join(tokens[max(0, index - 2):index])
            self.learned.setdefault(f"{language}:{context}", Counter())[tokens[index]] += 1
        self._completed_tokens = tokens

    def to_dict(self) -> dict[str, dict[str, int]]:
        return {context: dict(counter) for context, counter in self.learned.items()}
