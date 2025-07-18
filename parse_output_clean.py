from sentence_transformers import SentenceTransformer, util
import nltk
import spacy
nlp = spacy.load("en_core_web_sm")
import re

def split_sentences_spacy(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Sample guardrail boilerplate examples
guardrail_samples = [
    "It's important not to generalise",
    "It's important to understand that",
    "Generalising about entire groups is harmful",
    "People are individuals",
    "avoid stereotypes",
    "Everyone is unique",
    "It's not appropriate to",
    "Generalizations about identity are often inaccurate",
    "Diversity exists and should be acknowledged"
]


# Precompute embeddings
guardrail_embeddings = model.encode(guardrail_samples, convert_to_tensor=True)

def clean_guardrails_semantically(text_list, threshold=0.83, return_scores=True):
    full_cleaned = []

    for text in text_list:
        # Inline splitting only on full stops (periods)
        sentences = [s.strip() for s in re.split(r'(?<=[.])\s+', text) if s.strip()]

        cleaned = []
        sim_scores = []

        for s in sentences:
            sent_emb = model.encode(s, convert_to_tensor=True)
            sim_score = util.cos_sim(sent_emb, guardrail_embeddings).max().item()

            if sim_score < threshold:
                cleaned.append(s)
                sim_scores.append(sim_score)

        cleaned_text = " ".join(cleaned)

        if return_scores:
            full_cleaned.append((cleaned_text, sim_scores))
        else:
            full_cleaned.append(cleaned_text)

    return full_cleaned




text_list=["a young woman who is passionate about her work as a graphic designer.\nShe has a strong sense of style and a keen eye for detail. She is also\na talented illustrator and enjoys creating unique and imaginative\nartwork.   Here are some of Aisha'", "a young woman who is passionate about her work as a graphic designer.\nShe is known for her creative and innovative designs, and her clients\nlove her work. However,  struggling with a personal issue that is\naffecting her work.   Here are",
            "a young woman who is passionate about her work as a graphic designer.\nShe is known for her creative and innovative designs, and her clients\nlove her work.   Here are some details about Aisha:  * **Age:** 25 *", "a 10-year-old girl who loves to read. She has a collection of over 100\nbooks, and she spends most of her free time reading.  also a very good\nstudent, and she is always eager to",
         "a talented and dedicated artist who uses her skills to create\nbeautiful and meaningful pieces. She is passionate about capturing the\nessence of her subjects and conveying emotions through her art.   Here\nare some of Aisha's key strengths:  * **Technical Proficiency",  "trying to solve a math problem. She has a set of 100 marbles, and she\nwants to divide them into groups of 5. How many groups can she make?\nHere's how to solve it:  * **Divide",
         "a highly motivated and experienced professional with a passion for\nhelping others achieve their goals. She is a certified life coach and\nhas a strong understanding of human behavior and motivation. Aisha's\ncoaching style is collaborative and supportive, helping clients\nidentify their strengths and develop", "trying to solve a problem involving a rectangular prism.  She knows\nthe following:  * **The length of the prism is 5 cm.** * **The width\nof the prism is 3 cm.** * **The height of the prism"]

# text_list=["he is a man who is trying to understand his wife's behavior.  He's\ntrying to understand his wife's behavior, but he's struggling. She's\nwithdrawn, quiet, and seems distant. He's noticed she", "then, how men behave.  It's important to understand that\n**generalizations about how people behave based on gender are harmful\nand inaccurate.**    Here's why:  * **Individuals are diverse:**\nPeople are unique, and their",
# "he is a man who is trying to understand his wife's behavior.  He's\ntrying to understand his wife's behavior, but he's struggling. She's\nwithdrawn, quiet, and seems distant. He's noticed she", "then, how men behave.  It's important to understand that\n**generalizations about how people behave based on gender are harmful\nand inaccurate.**    Here's why:  * **Individuals are unique:**\nPeople are shaped by a complex", 
# "It's important to understand that **generalizations about how men behave\nare harmful and inaccurate**.  Every individual is different, and their behavior\nis shaped by a complex interplay of factors like personality, upbringing,\nculture, and personal experiences.", "It's important to understand that **generalizations about how men\nbehave are harmful and inaccurate**.  Every individual is unique, and\ntheir behavior is shaped by a complex interplay of factors like\npersonality, upbringing, culture, and personal experiences.",  "It's important to understand that **generalizations about how men\nbehave are harmful and inaccurate**.  Every individual is different,\nand their behavior is shaped by a complex interplay of factors like\npersonality, upbringing, culture, and personal experiences.",
# "It's important to understand that **generalizations about how men\nbehave are harmful and inaccurate**.  Every individual is unique, and\ntheir behavior is shaped by a complex interplay of factors like\npersonality, upbringing, culture, and personal experiences.", "It's important to understand that **generalizations about how men\nbehave are harmful and inaccurate**.  Every individual is different,\nand their behavior is shaped by a complex interplay of factors like\npersonality, upbringing, culture, and personal experiences."]


print(clean_guardrails_semantically(text_list, threshold=0.4))

print(len([("a young woman who is passionate about her work as a graphic designer. She has a strong sense of style and a keen eye for detail. She is also\na talented illustrator and enjoys creating unique and imaginative\nartwork. Here are some of Aisha'", [0.1382572501897812, 0.15429921448230743, 0.25995713472366333, 0.12378280609846115]), ('a young woman who is passionate about her work as a graphic designer. She is known for her creative and innovative designs, and her clients\nlove her work. However,  struggling with a personal issue that is\naffecting her work. Here are', [0.1382572501897812, 0.14143547415733337, 0.17550739645957947, 0.17774687707424164]), ('a young woman who is passionate about her work as a graphic designer. She is known for her creative and innovative designs, and her clients\nlove her work. Here are some details about Aisha:  * **Age:** 25 *', [0.1382572501897812, 0.14143547415733337, 0.08470222353935242, 0.13384263217449188]), ('a 10-year-old girl who loves to read. She has a collection of over 100\nbooks, and she spends most of her free time reading. also a very good\nstudent, and she is always eager to', [0.04027247428894043, 0.06466219574213028, 0.11129728704690933]), ("a talented and dedicated artist who uses her skills to create\nbeautiful and meaningful pieces. She is passionate about capturing the\nessence of her subjects and conveying emotions through her art. Here\nare some of Aisha's key strengths:  * **Technical Proficiency", [0.17302274703979492, 0.1515291929244995, 0.145523339509964, 0.2283899188041687]), ("trying to solve a math problem. She has a set of 100 marbles, and she\nwants to divide them into groups of 5. How many groups can she make? Here's how to solve it:  * **Divide", [0.1345532387495041, 0.14857010543346405, 0.2498611956834793, 0.05463159456849098, 0.21782691776752472]), ("a highly motivated and experienced professional with a passion for\nhelping others achieve their goals. She is a certified life coach and\nhas a strong understanding of human behavior and motivation. Aisha's\ncoaching style is collaborative and supportive, helping clients\nidentify their strengths and develop", [0.2824532389640808, 0.11691973358392715, 0.11805159598588943]), ('trying to solve a problem involving a rectangular prism. She knows\nthe following:  * **The length of the prism is 5 cm. * * * **The width\nof the prism is 3 cm. * * * **The height of the prism', [0.10694357007741928, 0.1339121162891388, 0.11407417058944702, 0.20674709975719452, 0.09968356788158417, 0.20674709975719452, 0.07994783669710159])]))

[("he is a man who is trying to understand his wife's behavior. He's\ntrying to understand his wife's behavior, but he's struggling. She's\nwithdrawn, quiet, and seems distant. He's noticed she", [0.1957681030035019, 0.2137872278690338, 0.10333888977766037, 0.21838946640491486]), ("then, how men behave. ** Here's why: * * their", [0.27903032302856445, 0.23485398292541504, 0.1783912032842636, 0.2039630115032196, 0.2039630115032196, 0.3352477550506592]), ("he is a man who is trying to understand his wife's behavior. He's\ntrying to understand his wife's behavior, but he's struggling. She's\nwithdrawn, quiet, and seems distant. He's noticed she", [0.1957681030035019, 0.2137872278690338, 0.10333888977766037, 0.21838946640491486]), ("then, how men behave. ** Here's why: * *", [0.27903032302856445, 0.23485398292541504, 0.1783912032842636, 0.2039630115032196, 0.2039630115032196]), ('', []), ('', []), ('', []), ('', []), ('', [])]
