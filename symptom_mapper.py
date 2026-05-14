import re
from typing import List, Optional

from symptom_semantic_mapper import SemanticMapper


REWRITE = [
    # =====================
    # GENERAL / SYSTEMIC
    # =====================
    (r"\bfeverish\b", "fever"),
    (r"\bhigh fever\b", "fever"),
    (r"\btemperature\b", "fever"),
    (r"\bchills\b", "chills"),
    (r"\bshivering\b", "chills"),
    (r"\btiredness\b", "fatigue"),
    (r"\bvery tired\b", "fatigue"),
    (r"\bexhaustion\b", "fatigue"),
    (r"\bweakness\b", "weakness"),
    (r"\bfeeling weak\b", "weakness"),
    (r"\bdizzy\b", "dizziness"),
    (r"\blightheaded\b", "dizziness"),
    (r"\bfear of weight gain\b", "fear of weight gain"),
    (r"\bskip meals\b", "skipping meals"),
    (r"\bskipping meals\b", "skipping meals"),
    # =====================
    # PAIN
    # =====================
    (r"\bsevere headache\b", "headache"),
    (r"\bhead pain\b", "headache"),
    (r"\bmigraine\b", "headache"),
    (r"\bstomachache\b", "abdominal pain"),
    (r"\bstomach ache\b", "abdominal pain"),
    (r"\bstomach pain\b", "abdominal pain"),
    (r"\btummy pain\b", "abdominal pain"),
    (r"\bbelly pain\b", "abdominal pain"),
    (r"\bchest pain\b", "chest pain"),
    (r"\bsharp chest pain\b", "sharp chest pain"),
    (r"\bjaw pain\b", "jaw pain"),
    (r"\bneck pain\b", "neck pain"),
    (r"\bback pain\b", "back pain"),
    (r"\bmuscle aches\b", "muscle pain"),
    (r"\bmuscle ache\b", "muscle pain"),
    (r"\bbody aches\b", "muscle pain"),

    (r"\bnumbness\b", "loss of sensation"),
    (r"\bface numbness\b", "facial numbness"),
    (r"\bweakness\b", "muscle weakness"),
    (r"\bone sided weakness\b", "muscle weakness"),
    (r"\bslurred speech\b", "speech difficulty"),
    (r"\bdifficulty speaking\b", "speech difficulty"),
    (r"\btrouble speaking\b", "speech difficulty"),
    (r"\bblurred vision\b", "vision loss"),
    (r"\btemporary blurred vision\b", "vision loss"),
    (r"\bvision problems\b", "vision loss"),
    (r"\bloss of balance\b", "balance disorder"),
    (r"\bdifficulty walking\b", "difficulty walking"),
    (r"\bfacial drooping\b", "facial weakness"),
    # =====================
    # RESPIRATORY / ENT
    # =====================
    (r"\bcoughing\b", "cough"),
    (r"\bpersistent dry cough\b", "cough"),
    (r"\bdry cough\b", "cough"),
    (r"\bwet cough\b", "cough"),
    (r"\bproductive cough\b", "cough"),
    (r"\bsore throat\b", "sore throat"),
    (r"\bthroat pain\b", "sore throat"),
    (r"\bthroat is sore\b", "sore throat"),
    (r"\bshortness of breath when walking\b", "shortness of breath"),
    (r"\bshort of breath\b", "shortness of breath"),
    (r"\bbreathlessness\b", "shortness of breath"),
    (r"\bdifficulty breathing\b", "difficulty breathing"),
    (r"\bdyspnea\b", "shortness of breath"),
    (r"\bdyspnea on exertion\b", "shortness of breath"),
    (r"\bwheezing\b", "wheezing"),
    (r"\brunny nose\b", "runny nose"),
    (r"\bstuffy nose\b", "nasal congestion"),
    (r"\bblocked nose\b", "nasal congestion"),
    (r"\bcongested nose\b", "nasal congestion"),
    (r"\bsneezing\b", "sneezing"),
    (r"\bear pain\b", "ear pain"),
    (r"\bhoarseness\b", "hoarseness"),
    (r"\bvoice change\b", "hoarseness"),

    # =====================
    # ORAL / SALIVARY / DENTAL
    # =====================
    (r"\bdry mouth\b", "dry mouth"),
    (r"\bbad taste\b", "bad taste"),
    (r"\bmetallic taste\b", "bad taste"),
    (r"\bmouth pain\b", "mouth pain"),
    (r"\btooth pain\b", "toothache"),
    (r"\btoothache\b", "toothache"),
    (r"\bgum pain\b", "gum pain"),
    (r"\bmouth swelling\b", "mouth swelling"),
    (r"\bface swelling\b", "facial swelling"),
    (r"\bswelling face\b", "facial swelling"),
    (r"\bswollen face\b", "facial swelling"),
    (r"\bswollen jaw\b", "jaw swelling"),
    (r"\bjaw swelling\b", "jaw swelling"),
    (r"\bswelling near jaw\b", "jaw swelling"),
    (r"\bswelling near ear\b", "facial swelling"),
    (r"\bdifficulty swallowing\b", "difficulty in swallowing"),
    (r"\btrouble swallowing\b", "difficulty in swallowing"),
    (r"\bpain when eating\b", "pain with eating"),
    (r"\bpain while eating\b", "pain with eating"),

    # =====================
    # DIGESTIVE
    # =====================
    (r"\bnauseous\b", "nausea"),
    (r"\bfeeling sick\b", "nausea"),
    (r"\bthrowing up\b", "vomiting"),
    (r"\bvomiting\b", "vomiting"),
    (r"\bdiarrhoea\b", "diarrhea"),
    (r"\bloose stools\b", "diarrhea"),
    (r"\bconstipated\b", "constipation"),
    (r"\bheartburn\b", "heartburn"),
    (r"\bacid reflux\b", "heartburn"),
    (r"\bloss of appetite\b", "decreased appetite"),
    (r"\bno appetite\b", "decreased appetite"),
    (r"\bbloating\b", "bloating"),
    (r"\babdominal swelling\b", "abdominal swelling"),

    # =====================
    # URINARY / KIDNEY
    # =====================
    (r"\bburning when urinating\b", "painful urination"),
    (r"\bburning urination\b", "painful urination"),
    (r"\bpain when urinating\b", "painful urination"),
    (r"\bfrequent urination\b", "frequent urination"),
    (r"\burinating often\b", "frequent urination"),
    (r"\bpeeing a lot\b", "frequent urination"),
    (r"\bblood in urine\b", "blood in urine"),
    (r"\bflank pain\b", "flank pain"),

    # =====================
    # DIABETES / METABOLIC
    # =====================
    (r"\bexcessive thirst\b", "thirst"),
    (r"\bvery thirsty\b", "thirst"),
    (r"\bincreased hunger\b", "hunger"),
    (r"\bextreme hunger\b", "hunger"),
    (r"\bblurred vision\b", "diminished vision"),
    (r"\bvision changes\b", "diminished vision"),
    (r"\bunexplained weight loss\b", "recent weight loss"),
    (r"\bweight loss\b", "recent weight loss"),
    (r"\bweight gain\b", "weight gain"),

    # =====================
    # SKIN
    # =====================
    (r"\brash\b", "skin rash"),
    (r"\bskin rash\b", "skin rash"),
    (r"\bitching\b", "itching of skin"),
    (r"\bitchy skin\b", "itching of skin"),
    (r"\bred skin\b", "skin redness"),
    (r"\bskin redness\b", "skin redness"),
    (r"\bskin swelling\b", "skin swelling"),
    (r"\bswelling\b", "swelling"),
    (r"\bbruising\b", "bruising"),
    (r"\bulcer\b", "ulcer"),
    (r"\bwound\b", "wound"),

    # =====================
    # EYES
    # =====================
    (r"\beye pain\b", "eye pain"),
    (r"\bred eyes\b", "redness of eye"),
    (r"\bred eye\b", "redness of eye"),
    (r"\bwatery eyes\b", "lacrimation"),
    (r"\bdouble vision\b", "double vision"),
    (r"\bloss of vision\b", "diminished vision"),
    (r"\bvision loss\b", "diminished vision"),

    # =====================
    # NEUROLOGICAL
    # =====================
    (r"\bfainting\b", "fainting"),
    (r"\bpassing out\b", "fainting"),
    (r"\bconfusion\b", "confusion"),
    (r"\bseizure\b", "seizures"),
    (r"\bseizures\b", "seizures"),
    (r"\bnumbness\b", "numbness"),
    (r"\btingling\b", "paresthesia"),
    (r"\bfocal weakness\b", "focal weakness"),
    (r"\bweakness on one side\b", "focal weakness"),
    (r"\bmemory loss\b", "memory loss"),

    # =====================
    # CARDIOVASCULAR
    # =====================
    (r"\bpalpitations\b", "palpitations"),
    (r"\bheart racing\b", "palpitations"),
    (r"\birregular heartbeat\b", "palpitations"),
    (r"\bfast heartbeat\b", "palpitations"),
    (r"\bleg swelling\b", "peripheral edema"),
    (r"\bankle swelling\b", "peripheral edema"),

    # =====================
    # MENTAL / ADHD / PSYCHIATRIC
    # =====================
    (r"\banxiety\b", "anxiety"),
    (r"\bpanic attacks\b", "panic attacks"),
    (r"\bpanic attack\b", "panic attacks"),
    (r"\bdepression\b", "depression"),
    (r"\bsadness\b", "depression"),
    (r"\binsomnia\b", "insomnia"),
    (r"\btrouble sleeping\b", "insomnia"),
    (r"\bdifficulty focusing\b", "lack of concentration"),
    (r"\btrouble focusing\b", "lack of concentration"),
    (r"\bpoor concentration\b", "lack of concentration"),
    (r"\binattention\b", "lack of concentration"),
    (r"\bdisorganization\b", "lack of concentration"),
    (r"\bpoor time management\b", "lack of concentration"),
    (r"\bfidgeting\b", "restlessness"),
    (r"\bhyperactivity\b", "restlessness"),
    (r"\bimpulsivity\b", "impulsivity"),
    (r"\bblurting answers\b", "impulsivity"),
    (r"\btrouble waiting turns\b", "impulsivity"),
    (r"\bexcessive talking\b", "hyperactivity"),
    (r"\brestlessness\b", "restlessness"),
    (r"\birritability\b", "irritable mood"),

    # =====================
    # GYNECOLOGICAL
    # =====================
    (r"\bvaginal dryness\b", "vaginal dryness"),
    (r"\bvaginal itching\b", "vaginal itching"),
    (r"\bvaginal discharge\b", "vaginal discharge"),
    (r"\bpain during intercourse\b", "pain during intercourse"),
    (r"\bpelvic pain\b", "pelvic pain"),
        # =====================
    # STROKE / TIA ADVANCED
    # =====================
    (r"\bmini stroke\b", "transient ischemic attack"),
    (r"\bstroke symptoms\b", "focal weakness"),
    (r"\bface drooping\b", "facial weakness"),
    (r"\bdrooping face\b", "facial weakness"),
    (r"\bcrooked smile\b", "facial weakness"),
    (r"\bone arm weakness\b", "arm weakness"),
    (r"\barm drift\b", "arm weakness"),
    (r"\bcan.t raise my arm\b", "arm weakness"),
    (r"\bcan.t move my arm\b", "arm weakness"),
    (r"\bsudden weakness\b", "focal weakness"),
    (r"\bsudden paralysis\b", "focal weakness"),
    (r"\bcan.t speak properly\b", "speech difficulty"),
    (r"\bwords are slurred\b", "speech difficulty"),
    (r"\bgarbled speech\b", "speech difficulty"),
    (r"\btrouble finding words\b", "speech difficulty"),
    (r"\bcannot understand speech\b", "confusion"),
    (r"\bsudden confusion\b", "confusion"),
    (r"\bvision went blurry\b", "diminished vision"),
    (r"\blost vision temporarily\b", "diminished vision"),
    (r"\bblind in one eye\b", "diminished vision"),
    (r"\btrouble seeing\b", "diminished vision"),
    (r"\bunsteady gait\b", "balance disorder"),
    (r"\bloss of coordination\b", "balance disorder"),
    (r"\bpoor coordination\b", "balance disorder"),
    (r"\bvertigo\b", "dizziness"),

    # =====================
    # HEART ATTACK / CARDIAC
    # =====================
    (r"\bpressure in my chest\b", "chest pain"),
    (r"\bheavy chest\b", "chest pain"),
    (r"\bchest heaviness\b", "chest pain"),
    (r"\btight chest\b", "chest pain"),
    (r"\bcrushing pain in chest\b", "sharp chest pain"),
    (r"\bpain radiating to jaw\b", "jaw pain"),
    (r"\bpain radiating to arm\b", "arm pain"),
    (r"\bleft arm pain\b", "arm pain"),
    (r"\bcold sweat\b", "sweating"),
    (r"\bcold sweats\b", "sweating"),
    (r"\bprofuse sweating\b", "sweating"),

    # =====================
    # COVID / FLU
    # =====================
    (r"\bloss of smell\b", "anosmia"),
    (r"\bcan.t smell\b", "anosmia"),
    (r"\bloss of taste\b", "loss of taste"),
    (r"\bcan.t taste anything\b", "loss of taste"),
    (r"\bflu like symptoms\b", "fever"),
    (r"\bbody chills\b", "chills"),

    # =====================
    # ALLERGY
    # =====================
    (r"\bhives\b", "skin rash"),
    (r"\bitchy rash\b", "itching of skin"),
    (r"\bpuffy eyes\b", "eye swelling"),
    (r"\bswollen lips\b", "lip swelling"),
    (r"\btongue swelling\b", "tongue swelling"),
    (r"\bthroat swelling\b", "throat swelling"),

    # =====================
    # GASTRO
    # =====================
    (r"\bstomach cramps\b", "abdominal pain"),
    (r"\babdominal cramps\b", "abdominal pain"),
    (r"\bcramps\b", "abdominal pain"),
    (r"\bfood poisoning\b", "vomiting"),
    (r"\bblack stool\b", "blood in stool"),
    (r"\bbloody stool\b", "blood in stool"),
    (r"\bblood in stool\b", "blood in stool"),

    # =====================
    # KIDNEY / UTI
    # =====================
    (r"\bcloudy urine\b", "abnormal appearing urine"),
    (r"\bsmelly urine\b", "abnormal appearing urine"),
    (r"\bstrong smelling urine\b", "abnormal appearing urine"),
    (r"\btrouble peeing\b", "difficulty urinating"),
    (r"\bdifficulty urinating\b", "difficulty urinating"),
    (r"\bcan.t urinate\b", "urinary retention"),

    # =====================
    # MUSCULOSKELETAL
    # =====================
    (r"\bjoint pain\b", "arthralgia"),
    (r"\bjoint stiffness\b", "stiffness"),
    (r"\bstiff joints\b", "stiffness"),
    (r"\bswollen joints\b", "joint swelling"),
    (r"\bmuscle stiffness\b", "muscle stiffness"),
    (r"\bneck stiffness\b", "neck stiffness"),

    # =====================
    # MENTAL HEALTH
    # =====================
    (r"\bfeeling hopeless\b", "depression"),
    (r"\bno motivation\b", "depression"),
    (r"\bpanic\b", "panic attacks"),
    (r"\boverthinking\b", "anxiety"),
    (r"\bconstant worry\b", "anxiety"),
    (r"\bmood swings\b", "mood changes"),
    (r"\bsocial withdrawal\b", "social isolation"),
    (r"\bhearing voices\b", "hallucinations"),
    (r"\bseeing things\b", "hallucinations"),

    # =====================
    # EATING DISORDER ADVANCED
    # =====================
    (r"\bobsessed with calories\b", "fear of weight gain"),
    (r"\bcounting calories constantly\b", "fear of weight gain"),
    (r"\brefusing to eat\b", "difficulty eating"),
    (r"\bavoiding meals\b", "skipping meals"),
    (r"\bfeeling fat\b", "body image disturbance"),
    (r"\bafraid to eat\b", "difficulty eating"),
    (r"\bbinge eating episodes\b", "binge eating"),
    (r"\beat and vomit\b", "vomiting"),

    # =====================
    # SALIVARY / DENTAL ADVANCED
    # =====================
    (r"\bpain near ear\b", "facial pain"),
    (r"\bpain below ear\b", "jaw pain"),
    (r"\bpus in mouth\b", "mouth infection"),
    (r"\bfoul taste\b", "bad taste"),
    (r"\bpain while chewing\b", "pain with eating"),
    (r"\bcheek swelling\b", "facial swelling"),

    # =====================
    # RESPIRATORY ADVANCED
    # =====================
    (r"\bcoughing blood\b", "hemoptysis"),
    (r"\bblood in cough\b", "hemoptysis"),
    (r"\bgreen mucus\b", "productive cough"),
    (r"\bphlegm\b", "productive cough"),
    (r"\bchest congestion\b", "cough"),

    # =====================
    # DERMATOLOGY
    # =====================
    (r"\bpeeling skin\b", "skin peeling"),
    (r"\bdry skin\b", "skin dryness"),
    (r"\boily skin\b", "skin changes"),
    (r"\bskin lesions\b", "skin lesion"),
    (r"\bdark spots\b", "skin discoloration"),

    # =====================
    # ENDOCRINE
    # =====================
    (r"\bheat intolerance\b", "heat intolerance"),
    (r"\bcold intolerance\b", "cold intolerance"),
    (r"\bnight sweats\b", "night sweats"),
    (r"\bincreased sweating\b", "sweating"),
    (r"\bhair loss\b", "hair loss"),

    # =====================
    # PEDIATRIC / GENERAL
    # =====================
    (r"\bcrying excessively\b", "irritable mood"),
    (r"\bnot eating\b", "decreased appetite"),
    (r"\blethargic\b", "fatigue"),
    (r"\bpoor feeding\b", "difficulty eating"),
        # =====================
    # RESPIRATORY EXTRA
    # =====================
    (r"\bchoking sensation\b", "difficulty breathing"),
    (r"\bair hunger\b", "shortness of breath"),
    (r"\bcan.t catch my breath\b", "shortness of breath"),
    (r"\btrouble catching breath\b", "shortness of breath"),
    (r"\bpainful breathing\b", "pain with breathing"),
    (r"\bbreathing hurts\b", "pain with breathing"),
    (r"\bdeep cough\b", "cough"),
    (r"\bconstant coughing\b", "cough"),
    (r"\bcough attacks\b", "cough"),
    (r"\bnight cough\b", "cough"),
    (r"\bmorning cough\b", "cough"),
    (r"\bsinus pressure\b", "sinus pain"),
    (r"\bsinus pain\b", "sinus pain"),
    (r"\bpost nasal drip\b", "runny nose"),
    (r"\bnasal drainage\b", "runny nose"),

    # =====================
    # GI / ABDOMINAL EXTRA
    # =====================
    (r"\bsharp stomach pain\b", "abdominal pain"),
    (r"\bburning stomach\b", "abdominal pain"),
    (r"\bupper abdominal pain\b", "abdominal pain"),
    (r"\blower abdominal pain\b", "abdominal pain"),
    (r"\bpain after eating\b", "abdominal pain"),
    (r"\bfeeling bloated\b", "bloating"),
    (r"\bexcess gas\b", "bloating"),
    (r"\bpassing gas\b", "flatulence"),
    (r"\bupset stomach\b", "nausea"),
    (r"\bindigestion\b", "heartburn"),
    (r"\bfood comes back up\b", "vomiting"),
    (r"\bretching\b", "vomiting"),

    # =====================
    # URINARY EXTRA
    # =====================
    (r"\bpelvic pressure\b", "pelvic pain"),
    (r"\bbladder pain\b", "pelvic pain"),
    (r"\bneed to pee often\b", "frequent urination"),
    (r"\bwaking up to pee\b", "frequent urination"),
    (r"\burgency to urinate\b", "frequent urination"),
    (r"\bleaking urine\b", "urinary incontinence"),
    (r"\bincontinence\b", "urinary incontinence"),

    # =====================
    # NEURO EXTRA
    # =====================
    (r"\bbrain fog\b", "confusion"),
    (r"\bmental fog\b", "confusion"),
    (r"\bslow thinking\b", "confusion"),
    (r"\btrouble remembering\b", "memory loss"),
    (r"\bforgetfulness\b", "memory loss"),
    (r"\bshaking\b", "tremor"),
    (r"\bhand tremor\b", "tremor"),
    (r"\bjerking movements\b", "seizures"),
    (r"\bmuscle twitching\b", "muscle twitching"),
    (r"\btwitching\b", "muscle twitching"),

    # =====================
    # CARDIOVASCULAR EXTRA
    # =====================
    (r"\bchest tightness\b", "chest pain"),
    (r"\bpounding heartbeat\b", "palpitations"),
    (r"\bskipped heartbeat\b", "palpitations"),
    (r"\bfluttering heart\b", "palpitations"),
    (r"\bhigh blood pressure\b", "hypertension"),
    (r"\blow blood pressure\b", "hypotension"),
    (r"\bblue lips\b", "cyanosis"),
    (r"\bblue fingers\b", "cyanosis"),

    # =====================
    # SKIN EXTRA
    # =====================
    (r"\bskin peeling\b", "skin peeling"),
    (r"\bflaky skin\b", "skin dryness"),
    (r"\bcracked skin\b", "skin dryness"),
    (r"\bskin burning\b", "skin pain"),
    (r"\bburning rash\b", "skin rash"),
    (r"\bpus filled bumps\b", "skin lesions"),
    (r"\bpimples\b", "skin lesions"),
    (r"\bacne\b", "skin lesions"),

    # =====================
    # EYE EXTRA
    # =====================
    (r"\bitchy eyes\b", "eye irritation"),
    (r"\bburning eyes\b", "eye irritation"),
    (r"\beye discharge\b", "eye discharge"),
    (r"\bsensitivity to light\b", "photophobia"),
    (r"\blight sensitivity\b", "photophobia"),
    (r"\bseeing flashes\b", "visual disturbance"),
    (r"\bfloaters\b", "visual disturbance"),

    # =====================
    # EAR EXTRA
    # =====================
    (r"\bringinging in ears\b", "tinnitus"),
    (r"\bbuzzing in ears\b", "tinnitus"),
    (r"\bhearing loss\b", "hearing loss"),
    (r"\bmuffled hearing\b", "hearing loss"),
    (r"\bfluid from ear\b", "ear discharge"),

    # =====================
    # MUSCULOSKELETAL EXTRA
    # =====================
    (r"\bmuscle cramps\b", "muscle pain"),
    (r"\bleg cramps\b", "muscle pain"),
    (r"\bback stiffness\b", "back pain"),
    (r"\bshoulder pain\b", "shoulder pain"),
    (r"\bknee pain\b", "knee pain"),
    (r"\bhip pain\b", "hip pain"),
    (r"\bfoot pain\b", "foot pain"),
    (r"\barm pain\b", "arm pain"),
    (r"\bhand pain\b", "hand pain"),

    # =====================
    # PSYCHIATRIC EXTRA
    # =====================
    (r"\bfeeling nervous\b", "anxiety"),
    (r"\bextreme fear\b", "anxiety"),
    (r"\bparanoia\b", "paranoia"),
    (r"\blow mood\b", "depression"),
    (r"\bcrying spells\b", "depression"),
    (r"\banger outbursts\b", "irritable mood"),
    (r"\bself harm thoughts\b", "suicidal thoughts"),
    (r"\bsuicidal thoughts\b", "suicidal thoughts"),
    (r"\bcan.t sleep\b", "insomnia"),
    (r"\bwaking up often\b", "insomnia"),
    (r"\bsleeping too much\b", "hypersomnia"),

    # =====================
    # ADHD / COGNITIVE EXTRA
    # =====================
    (r"\beasily distracted\b", "lack of concentration"),
    (r"\bforgetting tasks\b", "memory loss"),
    (r"\bforgetting appointments\b", "memory loss"),
    (r"\binterrupting people\b", "impulsivity"),
    (r"\bcan.t sit still\b", "restlessness"),
    (r"\bconstantly moving\b", "hyperactivity"),
    (r"\bdaydreaming\b", "lack of concentration"),

    # =====================
    # GYNECOLOGICAL EXTRA
    # =====================
    (r"\bheavy periods\b", "heavy menstrual bleeding"),
    (r"\bmissed period\b", "amenorrhea"),
    (r"\birregular periods\b", "irregular menstruation"),
    (r"\bmenstrual cramps\b", "pelvic pain"),
    (r"\bbreast pain\b", "breast pain"),
    (r"\bnipple discharge\b", "nipple discharge"),

    # =====================
    # INFECTIOUS
    # =====================
    (r"\bswollen glands\b", "lymph node enlargement"),
    (r"\bswollen lymph nodes\b", "lymph node enlargement"),
    (r"\benlarged lymph nodes\b", "lymph node enlargement"),
    (r"\bnight fever\b", "fever"),
    (r"\brecurrent infections\b", "frequent infections"),
    (r"\bflank pain\b", "flank pain"),
    (r"\bside pain\b", "flank pain"),
    (r"\bkidney pain\b", "flank pain"),
    (r"\bpain in my side\b", "flank pain"),
    (r"\blower back side pain\b", "flank pain"),
    # =====================
    # CANCER WARNING SIGNS
    # =====================
    (r"\bunintentional weight loss\b", "recent weight loss"),
    (r"\bextreme fatigue\b", "fatigue"),
    (r"\bpersistent pain\b", "chronic pain"),
    (r"\bnon healing wound\b", "wound"),
    (r"\bchronic cough\b", "cough"),
    (r"\bcough lasting weeks\b", "cough"),
]


_STOP_PREFIX = r"\b(including|with|having|i have|ive|i've|i am|im|i'm|i feel|feeling|patient has|the patient has)\b\s*"

BAD_SEMANTIC_MAPS = {
    "focal weakness",
    "hostile behavior",
    "temper problems",
    "drug abuse",
    "antisocial behavior",
}


def canonicalize(symptom: str) -> str:
    s = str(symptom or "").strip().lower()
    s = s.strip(" .,!?:;\"'()[]{}")
    s = re.sub(_STOP_PREFIX, "", s)

    for pat, rep in REWRITE:
        s = re.sub(pat, rep, s)

    s = re.sub(r"\b(severe|mild|moderate|high|low|extreme|persistent|occasional|acute|chronic)\b\s+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _should_try_semantic(s: str) -> bool:
    if not s:
        return False

    # Nu mapam semantic aceste simptome clare ca sa evitam zgomot
    protected = {
        "impulsivity",
        "hyperactivity",
        "restlessness",
        "lack of concentration",
        "dry mouth",
        "bad taste",
        "facial swelling",
        "jaw swelling",
        "jaw pain",
        "abdominal pain",
        "headache",
        "fever",
        "cough",
        "sore throat",
        "difficulty in swallowing",
        # respiratory — semantic mapper le distorsioneaza
        "runny nose",
        "nasal congestion",
        "sneezing",
        "wheezing",
        "shortness of breath",
        "difficulty breathing",
        # digestive
        "nausea",
        "vomiting",
        "diarrhea",
        "constipation",
        "bloating",
        # general
        "fatigue",
        "dizziness",
        "chills",
        "weakness",
        "rash",
        "itching of skin",
        # pain
        "chest pain",
        "back pain",
        "muscle pain",
        "neck pain",
        "ear pain",
        "eye pain",
    }

    if s in protected:
        return False

    if " " in s:
        return True

    if len(s) >= 8:
        return True

    return False


def canonicalize_list(
    symptoms: List[str],
    semantic: bool = False,
    mapper: Optional[SemanticMapper] = None,
    *,
    keep_original: bool = False,
) -> List[str]:
    out = []

    for s in symptoms:
        cs = canonicalize(s)
        if cs:
            out.append(cs)

    out = list(dict.fromkeys(out))

    if not semantic or not out:
        return out

    mapper = mapper or SemanticMapper()

    final = []

    for s in out:
        if not _should_try_semantic(s):
            final.append(s)
            continue

        try:
            ms, score = mapper.map_one(s)
            ms = canonicalize(ms)
        except Exception:
            final.append(s)
            continue

        if not ms or ms == s:
            final.append(s)
            continue

        if score < 0.65:
            final.append(s)
            continue

        if ms in BAD_SEMANTIC_MAPS:
            final.append(s)
            continue

        final.append(ms)

        if keep_original:
            final.append(s)

    final = [canonicalize(x) for x in final if x]
    final = [x for x in final if x]

    return list(dict.fromkeys(final))