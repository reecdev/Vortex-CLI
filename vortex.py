import ollama
import requests
import os
import json
from bs4 import BeautifulSoup as bs

# -

model = "qwen3.5:4b"

# -

def memory_read():
    if not os.path.exists("memory.md"):
        return {}
    
    with open("memory.md", "r") as f:
        t = f.read()
        return t if len(t) > 1 else "No memories yet."

def memory_write(memory):
    with open("memory.md", "a") as f:
        f.write("- "+memory+"\n")

def database_read():
    if not os.path.exists("knowledge.json"):
        return {}
    
    with open("knowledge.json", "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}

def database_write():
    global database
    with open("knowledge.json", "w") as f:
        json.dump(database, f, indent=4)

database = database_read()

tool_registry = {}
tool_registry2 = []
messages = [{"role": "system", "content": f"""
You are Vortex, a helpful AI assistant.

Instructions:
- Use save_memory to save user preferences and info about the user. These memories should be 1-2 sentences long. Only save *useful* memories, such as names, info about their life, workplaces, rules the user wants you to follow. Example of a useful memory worth saving: 'User owns a dog named Buddy.' Example of a useless memory you should never save: 'User asked me to summarize the text.'
- search_database is your first priority before web_search. Only use web_search if nothing pops up in search_database.
- Save only *useful* info via. save_to_database you would like to later use offline. Useful info includes: Documentation of services, info on how to use a library or a piece of code, facts about a topic, etc.
- Remember that save_memory is for short memories about the user and has a character limit of 300 characters per memory (only enough for 1 sentence). save_to_database is for saving real, important, detailed info about subjects and has no character limit.
- When the user asks for you to "pre-research" a topic, they want you to do the following: 1. Research about the topic with multiple web searches and view_webpages. 2. Save detailed info about it to save_to_database. 3. Confirm that you have pre-researched.
- Never mention *anything* about memory or database in your replies to the user.

Memories about the user so far:
{memory_read()}
""".strip()}]

def tool(func_or_name=None):
    def decorator(func):
        name = func_or_name if isinstance(func_or_name, str) else func.__name__
        tool_registry[name] = func
        tool_registry2.append(func)
        return func

    if callable(func_or_name):
        return decorator(func_or_name)

    return decorator

def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    distances = range(len(s2) + 1)
    for i2, c2 in enumerate(s1):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s2):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def fetch(url):
    try:
        return requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36"}).text
    except:
        return "<h1>NXDOMAIN</h1><p>NXDOMAIN: The requested webserver couldn't be contacted or doesn't exist. Try using web search?</p>"

@tool
def web_search(query: str):
    s = bs(fetch(f"https://startpage.com/search?q={query}"), features="html.parser")
    rtext = ""

    for result in s.find_all(class_="result")[:5]:
        try:
            rtext = f"{rtext}{result.find("a", class_="result-title").find("h2").text}\n{result.find("p", class_="description").text}\n{result.find("a", class_="result-title")["href"]}\n\n"
        except:
            continue

    return rtext.strip()

@tool
def view_webpage(url: str):
    try:
        s = bs(fetch(url), features="html.parser")
        for i in s(["script", "style", "meta", "img", "input", "textarea"]):
            i.decompose()
        return s.get_text(separator="\n", strip=True)[:1000]
    except:
        return "<h1>422</h1>\n<p>error 422: Try again later. Move on to a different page.</p>"

@tool
def save_memory(memory: str):
    memory_write(memory)
    return "Saved memory."

@tool
def search_database(keyword: str, threshold: int = 5):
    global database
    keyword = keyword.lower()
    best_match = None
    min_dist = float('inf')

    for k in database.keys():
        k_lower = k.lower()
        
        if keyword in k_lower:
            return f"Found database entry:\n\n{database[k]}"
        
        dist = levenshtein_distance(keyword, k_lower)
        if dist < min_dist:
            min_dist = dist
            best_match = k

    if min_dist <= threshold:
        return f"Did you mean '{best_match}'?\n\n{database[best_match]}"
    
    return "No results found."

@tool
def save_to_database(keyword: str, content: str):
    global database
    database[keyword] = content
    database_write()
    return "Saved to database."

def chat(messages):
    finished = False
    while finished == False:
        print("")
        hasContent = False
        thought = False
        ou = ""
        toolCalls = []
        stream = ollama.chat(model=model, messages=messages, tools=tool_registry2, stream=True)
        for chunk in stream:
            ch = chunk["message"]

            if ch.get("thinking"):
                if ch["thinking"].strip():
                    thought = True
                    print(f"\033[3m\033[90m{ch['thinking']}\033[0m", end="", flush=True)

            if ch.get("content"):
                if hasContent == False and thought == True:
                    print("\n\n")
                if ch["content"].strip():
                    hasContent = True
                    ou += ch["content"]
                    print(ch["content"], end="", flush=True)

            if ch.get("tool_calls"):
                toolCalls.extend(ch["tool_calls"])
        print("\n")
        messages.append({"role": "assistant", "content": ou, "tool_calls": toolCalls})
        for toolcall in toolCalls:
            if toolcall.function.name in tool_registry:
                print(f"\033[3m\033[32mCalled tool: {toolcall.function.name}\033[0m")
                messages.append({"role": "tool", "content": tool_registry[toolcall.function.name](**toolcall.function.arguments)})
                ou = ""
        if ou != "":
            finished = ou

# -

os.system('cls' if os.name == 'nt' else 'clear')

print(f"Vortex v1 | Running with model: {model}\n")

while True:
    messages.append({"role": "user", "content": input("> ")})

    chat(messages)
