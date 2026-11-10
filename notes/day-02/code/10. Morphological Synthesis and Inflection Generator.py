"""
=============================================================================
Course: NLP & Computer Vision (PGCP-AI) - Day 2 / Session 3
Topic:  Experiment 10: Morphological Synthesis (Inflected Form Generation)
File:   10. Morphological Synthesis and Inflection Generator.py
=============================================================================

WHAT IS MORPHOLOGICAL SYNTHESIS?
--------------------------------
Morphological Synthesis (generation) is the exact inverse of Morphological Analysis:
  - Morphological Analysis : Word Form -> Lemma / Root + Grammatical Features
      Example: "rides" -> root="ride", POS=Noun, Number=Plural
               "running" -> root="run", POS=Verb, Aspect=Progressive
  - Morphological Synthesis: Root Word + Grammatical Features -> Inflected Word
      Example: root="ride" + [Number=Plural] -> "rides"
               root="go" + [Tense=Past] -> "went"
               root="big" + [Degree=Superlative] -> "biggest"

WHY IS IT NEEDED IN NLP?
------------------------
1. Natural Language Generation (NLG): Producing grammatically sound responses.
2. Machine Translation (MT): Conjugating verbs into the correct target tense/person.
3. Dialogue Systems & Chatbots: Adapting verbs and pronouns to user context.

GRAMMATICAL FEATURES HANDLED BY THIS SYSTEM:
--------------------------------------------
1. Noun Inflections      : Singular vs. Plural (including regular & irregular rules)
2. Verb Conjugations     : Present (3rd person), Past, Past Participle, Present Participle
3. Adjective Comparisons : Positive, Comparative, Superlative
4. Indian Language Demo  : Hindi Verb Paradigm (padh -> padhta hai, padhenge, padhi)
=============================================================================
"""

class MorphologicalSynthesizer:
    """
    Rule-based system for generating inflected word forms from base root words
    and specified grammatical features.
    """

    def __init__(self):
        # Irregular English Noun Plurals
        self.irregular_nouns = {
            "child": "children",
            "man": "men",
            "woman": "women",
            "tooth": "teeth",
            "foot": "feet",
            "mouse": "mice",
            "person": "people",
            "goose": "geese",
            "ox": "oxen",
            "focus": "foci",
            "cactus": "cacti",
            "datum": "data"
        }

        # Irregular English Verbs: {root: (past, past_participle, 3rd_person, present_participle)}
        self.irregular_verbs = {
            "go": ("went", "gone", "goes", "going"),
            "see": ("saw", "seen", "sees", "seeing"),
            "take": ("took", "taken", "takes", "taking"),
            "buy": ("bought", "bought", "buys", "buying"),
            "eat": ("ate", "eaten", "eats", "eating"),
            "ride": ("rode", "ridden", "rides", "riding"),
            "write": ("wrote", "written", "writes", "writing"),
            "have": ("had", "had", "has", "having"),
            "be": ("was/were", "been", "is", "being"),
            "make": ("made", "made", "makes", "making"),
            "run": ("ran", "run", "runs", "running")
        }

        # Irregular Adjectives: {root: (comparative, superlative)}
        self.irregular_adjectives = {
            "good": ("better", "best"),
            "bad": ("worse", "worst"),
            "far": ("farther", "farthest"),
            "little": ("less", "least"),
            "many": ("more", "most"),
            "much": ("more", "most")
        }

    # -----------------------------------------------------------------------
    # 1. Noun Inflection (Singular -> Plural)
    # -----------------------------------------------------------------------
    def inflect_noun(self, root: str, number: str = "plural") -> str:
        """
        Synthesize noun based on target number ('singular' or 'plural').
        Applies English morphophonemic spelling rules.
        """
        root_lower = root.lower()
        if number.lower() in ["singular", "sg"]:
            return root

        # Check irregular exceptions first
        if root_lower in self.irregular_nouns:
            return self.irregular_nouns[root_lower]

        # Rule 1: Sibilants (s, x, z, ch, sh) add '-es'
        if root_lower.endswith(('s', 'x', 'z', 'ch', 'sh')):
            return root + "es"

        # Rule 2: Consonant + y -> '-ies' (e.g., party -> parties)
        vowels = "aeiou"
        if root_lower.endswith('y') and len(root_lower) > 1 and root_lower[-2] not in vowels:
            return root[:-1] + "ies"

        # Rule 3: Words ending in 'f' or 'fe' -> '-ves' (e.g., leaf -> leaves, life -> lives)
        if root_lower.endswith('fe'):
            return root[:-2] + "ves"
        if root_lower.endswith('f') and not root_lower.endswith(('ff', 'oof')):
            return root[:-1] + "ves"

        # Default Rule: Append '-s'
        return root + "s"

    # -----------------------------------------------------------------------
    # 2. Verb Inflection (Tenses and Aspects)
    # -----------------------------------------------------------------------
    def inflect_verb(self, root: str, feature: str = "past") -> str:
        """
        Synthesize verb form based on target grammatical feature:
          - 'present_3sg'      : 3rd person singular present (e.g., writes, runs)
          - 'past'             : Simple past tense (e.g., wrote, walked)
          - 'past_participle'  : Perfect aspect form (e.g., written, walked)
          - 'present_participle': Continuous / progressive form (e.g., writing, running)
        """
        root_lower = root.lower()
        vowels = "aeiou"

        # Check irregular verbs lookup table
        if root_lower in self.irregular_verbs:
            past, p_part, pres_3sg, pres_part = self.irregular_verbs[root_lower]
            mapping = {
                "past": past,
                "past_participle": p_part,
                "present_3sg": pres_3sg,
                "present_participle": pres_part
            }
            return mapping.get(feature, root)

        # Helper function for consonant doubling in regular verbs
        def should_double_consonant(word: str) -> bool:
            vowels = "aeiou"
            if len(word) < 3:
                return False
            # Multisyllabic words usually do not double unless stressed at the end (e.g., refer, commit)
            vowel_count = sum(1 for ch in word if ch in vowels)
            if vowel_count > 1:
                multisyllabic_doublers = {"refer", "occur", "commit", "begin", "prefer"}
                return word in multisyllabic_doublers
            # Monosyllabic CVC (e.g., stop, run, plan, hop)
            return (word[-1] not in vowels and 
                    word[-2] in vowels and 
                    word[-3] not in vowels and 
                    word[-1] not in "wxy")

        # Regular Verb Inflection Rules:
        if feature == "present_3sg":
            if root_lower.endswith(('s', 'x', 'z', 'ch', 'sh', 'o')):
                return root + "es"
            if root_lower.endswith('y') and len(root_lower) > 1 and root_lower[-2] not in vowels:
                return root[:-1] + "ies"
            return root + "s"

        elif feature in ["past", "past_participle"]:
            if root_lower.endswith('e'):
                return root + "d"
            if root_lower.endswith('y') and len(root_lower) > 1 and root_lower[-2] not in vowels:
                return root[:-1] + "ied"
            if should_double_consonant(root_lower):
                return root + root[-1] + "ed"
            return root + "ed"

        elif feature == "present_participle":
            if root_lower.endswith('ie'):
                return root[:-2] + "ying"
            if root_lower.endswith('e') and not root_lower.endswith('ee'):
                return root[:-1] + "ing"
            if should_double_consonant(root_lower):
                return root + root[-1] + "ing"
            return root + "ing"

        return root

    # -----------------------------------------------------------------------
    # 3. Adjective Comparison Inflection
    # -----------------------------------------------------------------------
    def inflect_adjective(self, root: str, degree: str = "comparative") -> str:
        """
        Synthesize adjective comparison:
          - 'comparative': bigger, faster, more beautiful
          - 'superlative': biggest, fastest, most beautiful
        """
        root_lower = root.lower()
        if degree == "positive":
            return root

        # Check irregular adjectives
        if root_lower in self.irregular_adjectives:
            comp, supl = self.irregular_adjectives[root_lower]
            return comp if degree == "comparative" else supl

        # Multi-syllable heuristic (words with > 2 syllables or ending in '-ful', '-ous', '-able')
        if len(root_lower) > 6 or root_lower.endswith(('ful', 'ous', 'able', 'ive', 'ing')):
            prefix = "more " if degree == "comparative" else "most "
            return prefix + root

        # Short words rule
        vowels = "aeiou"
        if degree == "comparative":
            if root_lower.endswith('e'):
                return root + "r"
            if root_lower.endswith('y') and root_lower[-2] not in vowels:
                return root[:-1] + "ier"
            if (len(root_lower) >= 3 and 
                root_lower[-1] not in vowels and 
                root_lower[-2] in vowels and 
                root_lower[-3] not in vowels):
                return root + root[-1] + "er"
            return root + "er"

        elif degree == "superlative":
            if root_lower.endswith('e'):
                return root + "st"
            if root_lower.endswith('y') and root_lower[-2] not in vowels:
                return root[:-1] + "iest"
            if (len(root_lower) >= 3 and 
                root_lower[-1] not in vowels and 
                root_lower[-2] in vowels and 
                root_lower[-3] not in vowels):
                return root + root[-1] + "est"
            return root + "est"

        return root

    # -----------------------------------------------------------------------
    # 4. Indian Language Demo: Hindi Verb Morphological Synthesis
    # -----------------------------------------------------------------------
    def inflect_hindi_verb(self, root: str, gender: str = "m", number: str = "sg", tense: str = "present") -> str:
        """
        Synthesizes Hindi verb forms from root verb (dhatu).
        Features: gender (m/f), number (sg/pl), tense (present/past/future).
        Example: padh (root) -> padhta hai / padhti hai / padhenge
        """
        is_m = (gender.lower() == "m")
        is_sg = (number.lower() in ["sg", "singular"])

        if tense == "present":
            # Present Habitual: root + taa/te/tee + hai/hain
            suffix = "ta" if (is_m and is_sg) else ("te" if is_m else "ti")
            aux = "hai" if is_sg else "hain"
            return f"{root}{suffix} {aux}"

        elif tense == "past":
            # Past Perfective: if root ends in vowel (e.g., khaa -> khaayaa, khaaye, khaayee)
            if root.endswith(('a', 'aa', 'ee', 'o')):
                suffix = "yaa" if (is_m and is_sg) else ("ye" if is_m else ("yee" if is_sg else "yeen"))
            else:
                suffix = "aa" if (is_m and is_sg) else ("e" if is_m else ("ee" if is_sg else "een"))
            return f"{root}{suffix}"

        elif tense == "future":
            # Future: root + eyga/eynge/eygi/eyngi
            if is_m:
                suffix = "ega" if is_sg else "enge"
            else:
                suffix = "egi" if is_sg else "engi"
            return f"{root}{suffix}"

        return root


# ===========================================================================
# Demonstration and Testing
# ===========================================================================
if __name__ == "__main__":
    synthesizer = MorphologicalSynthesizer()

    print("=" * 75)
    print("EXPERIMENT 10: MORPHOLOGICAL SYNTHESIS (INFLECTION GENERATION)")
    print("=" * 75)
    print("A system to generate inflected word forms from root words based on features.\n")

    # -----------------------------------------------------------------------
    # Test 1: Noun Inflection (Singular -> Plural)
    # -----------------------------------------------------------------------
    print("=" * 75)
    print("1. NOUN INFLECTION SYSTEM (Singular to Plural)")
    print("=" * 75)
    test_nouns = ["park", "coaster", "bus", "wish", "city", "leaf", "child", "mouse", "foot"]
    print(f"{'Root Noun':<15} | {'Input Feature':<15} | {'Synthesized Word Form'}")
    print("-" * 55)
    for noun in test_nouns:
        plural_form = synthesizer.inflect_noun(noun, number="plural")
        print(f"{noun:<15} | {'Number=Plural':<15} | {plural_form}")

    # -----------------------------------------------------------------------
    # Test 2: Verb Inflection (Tense / Aspect Paradigm)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("2. VERB INFLECTION PARADIGM (Roots to Inflected Tense Forms)")
    print("=" * 75)
    test_verbs = ["visit", "watch", "study", "stop", "go", "ride", "see"]
    features = [
        ("present_3sg", "Present (3rd Sg)"),
        ("past", "Simple Past"),
        ("present_participle", "Continuous (-ing)"),
        ("past_participle", "Past Participle")
    ]

    print(f"{'Root Verb':<10} | {'Present (3sg)':<14} | {'Simple Past':<14} | {'Continuous':<14} | {'Past Participle'}")
    print("-" * 75)
    for verb in test_verbs:
        p3sg = synthesizer.inflect_verb(verb, "present_3sg")
        past = synthesizer.inflect_verb(verb, "past")
        prog = synthesizer.inflect_verb(verb, "present_participle")
        ppar = synthesizer.inflect_verb(verb, "past_participle")
        print(f"{verb:<10} | {p3sg:<14} | {past:<14} | {prog:<14} | {ppar}")

    # -----------------------------------------------------------------------
    # Test 3: Adjective Comparison System
    # -----------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("3. ADJECTIVE COMPARISON SYNTHESIS")
    print("=" * 75)
    test_adjs = ["big", "fast", "happy", "exciting", "magical", "good", "bad"]
    print(f"{'Positive (Root)':<18} | {'Comparative':<20} | {'Superlative'}")
    print("-" * 65)
    for adj in test_adjs:
        comp = synthesizer.inflect_adjective(adj, "comparative")
        supl = synthesizer.inflect_adjective(adj, "superlative")
        print(f"{adj:<18} | {comp:<20} | {supl}")

    # -----------------------------------------------------------------------
    # Test 4: Hindi Morphological Synthesis Paradigm (padh, khaa, likh)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("4. HINDI VERB MORPHOLOGICAL SYNTHESIS (Root + Gender/Number/Tense)")
    print("=" * 75)
    test_hindi = [
        ("padh", "m", "sg", "present"),
        ("padh", "f", "sg", "present"),
        ("padh", "m", "pl", "future"),
        ("khaa", "m", "sg", "past"),
        ("khaa", "f", "pl", "past"),
        ("likh", "m", "sg", "future")
    ]

    print(f"{'Dhatu (Root)':<14} | {'Grammatical Features':<28} | {'Generated Inflected Form'}")
    print("-" * 70)
    for root, g, n, t in test_hindi:
        generated = synthesizer.inflect_hindi_verb(root, gender=g, number=n, tense=t)
        feat_desc = f"Gender={g.upper()}, Num={n.upper()}, Tense={t.capitalize()}"
        print(f"{root:<14} | {feat_desc:<28} | {generated}")

    print("=" * 75)
    print("[Summary] Morphological synthesis generates accurate inflected forms from grammatical features.")
    print("=" * 75)
