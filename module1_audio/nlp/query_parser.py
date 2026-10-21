# nlp/query_parser.py
# ─────────────────────────────────────────────────────────────
# Module 1 – Hearing & Understanding
# Rule-based intent, object, and attribute extraction.
# ─────────────────────────────────────────────────────────────

import spacy
from nlp.keyword_extractor import extract_keywords

# Attribute dictionaries
COLORS = {
    "black", "white", "red", "blue", "green", "yellow", 
    "orange", "pink", "purple", "brown", "grey", "gray"
}
SIZES = {"small", "big", "large", "tiny", "huge"}
POSITIONS = {"left", "right", "front", "behind", "near", "far"}


def get_intent(text_lower: str) -> str:
    """Determine the user's intent using simple rule-based keyword matching."""
    if any(word in text_lower for word in ["where", "find", "locate", "search"]):
        return "find_object"
    
    if any(word in text_lower for word in ["describe", "color", "size"]):
        return "describe_object"
        
    if any(word in text_lower for word in ["what is", "identify", "what's"]):
        return "identify_object"
        
    if any(word in text_lower for word in ["tell me", "about"]):
        return "general_question"
        
    return "unknown"


def parse_query(nlp_model: spacy.Language, text: str) -> dict:
    """
    Parse the query to extract intent, object, attributes, and keywords.
    """
    # 1. Extract base keywords and the main object (noun)
    kw_result = extract_keywords(nlp_model, text)
    keywords = kw_result["keywords"]
    main_object = kw_result["main_keyword"]
    
    # Ensure the main object is not mistakenly an attribute (like "left", which spaCy might tag as a noun)
    # Also ignore generic attribute property names ("color", "size", "position")
    ignore_as_object = POSITIONS | COLORS | SIZES | {"color", "size", "position", "shape"}
    if main_object in ignore_as_object:
        for kw in reversed(keywords):
            if kw not in ignore_as_object:
                main_object = kw
                break
    
    # 2. Determine intent
    text_lower = text.lower()
    intent = get_intent(text_lower)
    
    # 3. Extract attributes (colour, size, position)
    attributes = {}
    
    # We can use spaCy tokens to ensure we match whole words and not substrings
    doc = nlp_model(text_lower)
    for token in doc:
        word = token.text
        if word in COLORS:
            attributes["color"] = word
        elif word in SIZES:
            attributes["size"] = word
        elif word in POSITIONS:
            attributes["position"] = word
            
    return {
        "text": text,
        "language": "en",
        "intent": intent,
        "object": main_object,
        "attributes": attributes,
        "keywords": keywords
    }
