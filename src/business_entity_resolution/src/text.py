"""Name / address normalisation. Pure functions, no external lookups.

Every raw record becomes a dict of derived string fields used by blocking and features:
  name_clean   : lowercased, accent-free, transliterated, l33t-fixed tokens (legal words canonicalised)
  name_core    : name_clean without legal-form and filler tokens
  name_sorted  : sorted unique name_core tokens
  name_concat  : name_core without spaces (compares against domain-style names)
  legal        : sorted canonical legal-form tokens
  is_domain    : 1 if the raw name looked like a web domain
  addr_clean   : normalised address (street types / ordinals canonicalised, state removed)
  addr_nums    : space-joined numeric tokens (leading zeros stripped)
  house        : first number of the street component
  street       : alphabetic tokens of the street component
  city         : alphabetic tokens of non-street, non-state components
  state        : canonical state/region code
  postcode     : 6-digit PIN code if present
"""
import re
import unicodedata

from anyascii import anyascii

# ---------------------------------------------------------------- dictionaries
LEGAL = {
    "inc": "inc", "incorporated": "inc", "llc": "llc", "pllc": "pllc", "llp": "llp", "lp": "lp",
    "corp": "corp", "corporation": "corp", "co": "co", "company": "co", "ltd": "ltd", "limited": "ltd",
    "pvt": "pvt", "private": "pvt", "pc": "pc", "plc": "plc", "opc": "opc", "pa": "pa",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa", "sci": "sci", "snc": "snc", "eurl": "eurl",
    "scop": "scop", "selarl": "selarl", "cie": "cie", "gmbh": "gmbh",
    "pra": "pvt", "prai": "pvt", "li": "ltd", "lim": "ltd", "pvtltd": "pvt",   # abbreviated Indic forms
}
FILLER = {"the", "and", "of", "a", "an", "dba", "de", "la", "le", "les", "du", "des", "et", "l", "d"}

STREET = {
    "street": "st", "st": "st", "str": "st", "saint": "st", "sreet": "st", "stree": "st",
    "road": "rd", "rd": "rd", "avenue": "ave", "ave": "ave", "av": "ave", "drive": "dr", "dr": "dr",
    "lane": "ln", "ln": "ln", "boulevard": "blvd", "blvd": "blvd", "bd": "blvd", "bld": "blvd",
    "court": "ct", "ct": "ct", "place": "pl", "pl": "pl", "circle": "cir", "cir": "cir",
    "highway": "hwy", "hwy": "hwy", "parkway": "pkwy", "pkwy": "pkwy", "terrace": "ter", "ter": "ter",
    "trail": "trl", "trl": "trl", "square": "sq", "sq": "sq", "way": "way", "pike": "pike",
    "north": "n", "south": "s", "east": "e", "west": "w", "n": "n", "s": "s", "e": "e", "w": "w",
    "apartment": "apt", "apt": "apt", "suite": "ste", "ste": "ste", "unit": "unit", "floor": "fl", "fl": "fl",
    "number": "no", "no": "no", "nos": "no", "door": "no", "plot": "plot", "near": "near", "opp": "opp",
    "opposite": "opp", "behind": "behind", "nagar": "nagar", "ngr": "nagar", "marg": "marg", "mg": "marg",
    "colony": "colony", "sector": "sector", "sec": "sector", "phase": "phase", "ph": "phase",
    "rue": "rue", "r": "rue", "allee": "allee", "all": "allee", "chemin": "chemin", "ch": "chemin",
    "impasse": "impasse", "imp": "impasse", "route": "rte", "rte": "rte", "quai": "quai", "cours": "cours",
    "faubourg": "fbg", "fbg": "fbg", "residence": "res", "res": "res", "lieu": "lieu", "dit": "dit",
}
STREET_TYPES = {"st", "rd", "ave", "dr", "ln", "blvd", "ct", "pl", "cir", "hwy", "pkwy", "ter", "trl", "sq",
                "way", "pike", "rue", "allee", "chemin", "impasse", "rte", "quai", "cours", "fbg", "marg"}
CITY_NOISE = {"city", "county", "township", "ownship", "cdp", "region", "district", "dist", "corporation",
              "hq", "town", "village", "municipality", "mandal", "tehsil", "taluk", "division"}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca", "colorado": "co",
    "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la",
    "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny", "north carolina": "nc",
    "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
    "rhode island": "ri", "south carolina": "sc", "south dakota": "sd", "tennessee": "tn", "texas": "tx",
    "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa", "west virginia": "wv",
    "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc", "puerto rico": "pr",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br", "chhattisgarh": "cg",
    "chattisgarh": "cg", "ct": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp",
    "jharkhand": "jh", "karnataka": "ka", "kerala": "kl", "keralam": "kl", "madhya pradesh": "mp",
    "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl",
    "odisha": "od", "orissa": "od", "or": "od", "punjab": "pb", "rajasthan": "rj", "sikkim": "sk",
    "tamil nadu": "tn", "telangana": "tg", "ts": "tg", "tripura": "tr", "uttar pradesh": "up",
    "uttarakhand": "uk", "uttaranchal": "uk", "ut": "uk", "west bengal": "wb", "delhi": "dl",
    "new delhi": "dl", "nct of delhi": "dl", "jammu and kashmir": "jk", "jammu kashmir": "jk",
    "ladakh": "la", "chandigarh": "ch", "puducherry": "py", "pondicherry": "py",
    "andaman and nicobar islands": "an", "dadra and nagar haveli": "dn", "daman and diu": "dd",
    "lakshadweep": "ld",
}
IN_CODES = set(IN_STATES.values())
FR_REGIONS = {
    "ile de france": "idf", "hauts de france": "hdf", "nouvelle aquitaine": "naq", "pays de la loire": "pdl",
    "bretagne": "bre", "normandie": "nor", "grand est": "ges", "occitanie": "occ", "bourgogne franche comte": "bfc",
    "auvergne rhone alpes": "ara", "provence alpes cote d azur": "pac", "centre val de loire": "cvl",
    "corse": "cor",
    # departements of the regions present in the data -> region
    "nord": "hdf", "pas de calais": "hdf", "somme": "hdf", "aisne": "hdf", "oise": "hdf",
    "gironde": "naq", "landes": "naq", "dordogne": "naq", "lot et garonne": "naq", "pyrenees atlantiques": "naq",
    "charente": "naq", "charente maritime": "naq", "deux sevres": "naq", "vienne": "naq", "haute vienne": "naq",
    "correze": "naq", "creuse": "naq",
    "loire atlantique": "pdl", "maine et loire": "pdl", "mayenne": "pdl", "sarthe": "pdl", "vendee": "pdl",
    "paris": "idf",
}
STATE_MAPS = {"us": US_STATES, "india": IN_STATES, "france": FR_REGIONS}
STATE_CODES = {k: set(v.values()) for k, v in STATE_MAPS.items()}
ALL_STATES = {**FR_REGIONS, **IN_STATES, **US_STATES}

LEET = {"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b", "@": "a", "$": "s"}
DOMAIN_RE = re.compile(r"^(?:www\.)?([a-z0-9\-]+)\.(com|in|net|org|co|io|biz|fr|us|info|co\.in)\b")
JUNK_RE = re.compile(r"[<>#*~^|=_+\"`]+")
NON_ALNUM_RE = re.compile(r"[^0-9a-z ]+")
ORDINAL_RE = re.compile(r"^(\d+)(st|nd|rd|th)$")
NUM_RE = re.compile(r"\d+")
DIGIT_ALPHA_RE = re.compile(r"(?=.*[a-z])(?=.*\d)")

INDIC_DICT: dict = {}   # learned token dictionary (translit.py), injected at runtime


def set_indic_dict(d: dict):
    INDIC_DICT.clear()
    INDIC_DICT.update(d)


def is_latin(s: str) -> bool:
    return all(ord(c) < 0x250 for c in s)


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


TOKEN_SPLIT_RE = re.compile(r"([\s,.;:()\[\]\-/#]+)")
# single-word Indic state names whose learned translation is only part of the state name
STATE_WORDS = {"bengal": "west bengal", "nadu": "tamil nadu", "pradesh": "pradesh"}


def to_latin(s: str) -> str:
    """Transliterate non-Latin tokens (split on punctuation too, so 'दिल्ली,' hits the dictionary):
    learned dictionary first, anyascii fallback."""
    if is_latin(s):
        return s
    out = []
    for t in TOKEN_SPLIT_RE.split(s):
        if not t or is_latin(t):
            out.append(t)
        else:
            lat = INDIC_DICT.get(t) or anyascii(t)
            out.append(STATE_WORDS.get(lat, lat) if lat in ("bengal", "nadu") else lat)
    return "".join(out)


def base_clean(s: str) -> str:
    s = to_latin(s)
    s = strip_accents(unicodedata.normalize("NFKC", s)).lower()
    return s


def fix_leet(tok: str) -> str:
    if DIGIT_ALPHA_RE.match(tok) and sum(c.isalpha() for c in tok) >= 2:
        return "".join(LEET.get(c, c) for c in tok)
    return tok


# ---------------------------------------------------------------- names
def norm_name(raw: str) -> dict:
    s = base_clean(raw).strip()
    s = JUNK_RE.sub(" ", s).strip(" -.,")
    is_domain = 0
    m = DOMAIN_RE.match(s.replace(" ", ""))
    if m and " " not in s.strip():
        is_domain = 1
        s = m.group(1).replace("-", "")
    if " dba " in f" {s} ":
        before, _, after = f" {s} ".partition(" dba ")
        s = after.strip() or before.strip() or s
    s = s.replace("&", " and ")
    s = re.sub(r"(?<=\b[a-z])\.(?=[a-z]\b)", "", s)      # p.l.l.c -> pllc, s.a.s -> sas
    s = s.replace(".", "").replace("'", "")
    s = NON_ALNUM_RE.sub(" ", s)
    toks = [fix_leet(t) for t in s.split()]
    clean, core, legal = [], [], []
    for t in toks:
        if t in LEGAL:
            legal.append(LEGAL[t])
            clean.append(LEGAL[t])
        else:
            clean.append(t)
            if t not in FILLER:
                core.append(t)
    if not core:                       # name made only of legal/filler words
        core = [t for t in clean]
    return {
        "name_clean": " ".join(clean),
        "name_core": " ".join(core),
        "name_sorted": " ".join(sorted(set(core))),
        "name_concat": "".join(core),
        "legal": " ".join(sorted(set(legal))),
        "is_domain": is_domain,
    }


# ---------------------------------------------------------------- addresses
def _state_of(comp: str, country: str):
    """Canonical state/region code for an address component, country-aware (OR = Oregon in the US)."""
    m = STATE_MAPS.get(country)
    if m is None:                       # unseen country: accept any known full name, no bare codes
        return ALL_STATES.get(comp) if len(comp) > 3 else None
    if comp in m:
        return m[comp]
    if len(comp) == 2 and comp in STATE_CODES[country]:
        return comp
    return None


def _norm_tok(t: str) -> str:
    m = ORDINAL_RE.match(t)
    if m:
        return m.group(1).lstrip("0") or "0"
    if t.isdigit():
        return t.lstrip("0") or "0"
    return STREET.get(t, t)


def norm_address(raw: str, country: str = "") -> dict:
    country = country.lower()
    s = base_clean(raw)
    s = re.sub(r"\bnull\b", " ", s)
    s = JUNK_RE.sub(" ", s)
    s = s.replace("&", " and ").replace("'", "")
    comps = []
    for c in s.split(","):
        c = NON_ALNUM_RE.sub(" ", c.replace(".", " ").replace("-", " ").replace("/", " ")).strip()
        c = " ".join(c.split())
        if c:
            comps.append(c)
    state, postcode, street_comp = "", "", None
    kept = []
    for c in comps:
        st = _state_of(c, country)
        if st is not None:
            state = state or st
            continue
        kept.append(c)
    all_toks, nums, street, city = [], [], [], []
    for c in kept:
        toks = [_norm_tok(t) for t in c.split()]
        has_num = any(t.isdigit() for t in toks)
        has_type = any(t in STREET_TYPES for t in toks)
        for t in toks:
            if t.isdigit():
                nums.append(t)
                if len(t) == 6 and not postcode:
                    postcode = t
        if (has_num or has_type) and street_comp is None:
            street_comp = toks
        elif not has_num:
            city.extend(t for t in toks if t not in CITY_NOISE and t not in STREET_TYPES)
        all_toks.extend(toks)
    house, street_toks = "", []
    if street_comp is not None:
        for t in street_comp:
            if t.isdigit() and not house:
                house = t
            elif not t.isdigit() and t not in STREET_TYPES and t not in ("no", "plot", "apt", "ste", "unit", "fl"):
                street_toks.append(t)
    if not house and nums:
        house = nums[0]
    return {
        "addr_clean": " ".join(all_toks),
        "addr_nums": " ".join(dict.fromkeys(nums)),
        "house": house,
        "street": " ".join(street_toks),
        "city": " ".join(dict.fromkeys(city)),
        "state": state,
        "postcode": postcode,
    }


if __name__ == "__main__":
    tests = ["Raab Modern Treasury, LLC", "Llc Raab Modern Treoasubr,y", "raabmoderntreasury.com",
             "Reliable Digital Collective P.L.L.C.", "-- Holloway Peak Inc Seafood", "Indo 5ky Energy",
             "Tavozephdelta dba Perfect Technology Pvt Ltd", "<< Team Ecole", "ELDER & (PUCKETT)",
             "Payne Énterprises", "SCI Ptit Àmicale", "Marina Ecole France Sarl"]
    for t in tests:
        print(t, "->", norm_name(t))
    for a in ["2670- DUMBLE SAINT, TX, ALVIN", "PITTSFIEDL, ME, 01466 MAIN ST",
              "##1138 Jamison Ave, Roanoke City, Virginia", "32623 224st Pl, Black Diamond, Washington",
              "Gujarat, Adarsh Society, Near Gokulam Dairy, Athwalines, Surat, Plot No. 569/A",
              "3315 FREMONT ST, null, PEORIA, IL", "29 Bd. De La Libération, Saint-nazaire",
              "12 All. Jean-jacques Rousseau, Saint-nazaire, Pays de la Loire", "", "10, New Delhi, DL"]:
        print(a, "->", norm_address(a))
