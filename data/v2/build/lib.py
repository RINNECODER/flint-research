import sys,json,re,glob,hashlib
sys.path.insert(0,'.')  # run from data/v2/build
from render import *   # tok, EOS, build, render, enc
import pyarrow.parquet as pq
MAXTOK=6144
SYSTEM=("You are an autonomous coding agent working in the user's repository. Use your tools to read, search, edit and run code. "
        "Inspect before you change anything, make focused edits, run the tests or the program to verify, and give a short final summary.")
def P(n,t="string",**k): return {"type":t,**k}
def fn(name,desc,props,req): return {"type":"function","function":{"name":name,"description":desc,"parameters":{"type":"object","properties":props,"required":req}}}
S={"type":"string"};I={"type":"integer"}
TOOLS=[
 fn("bash","Run a shell command and return its output.",{"command":S,"timeout":I},["command"]),
 fn("read","Read a file (optionally a line range).",{"path":S,"offset":I,"limit":I},["path"]),
 fn("write","Create or overwrite a file with the given content.",{"path":S,"content":S},["path","content"]),
 fn("edit","Edit a file by replacing exact text snippets.",{"path":S,"edits":{"type":"array","items":{"type":"object","properties":{"oldText":S,"newText":S},"required":["oldText","newText"]}}},["path","edits"]),
 fn("grep","Search file contents with a regex.",{"pattern":S,"path":S},["pattern"]),
 fn("glob","Find files by glob pattern.",{"pattern":S,"path":S},["pattern"]),
 fn("web_search","Search the web.",{"query":S},["query"]),
 fn("web_fetch","Fetch a URL and answer a question about it.",{"url":S,"prompt":S},["url"]),
]
ALLOWED={t["function"]["name"] for t in TOOLS}
class Skip(Exception): pass
def norm_call(name,a):
    """map any source tool call to the unified toolset; raise Skip if unmappable"""
    if not isinstance(a,dict): raise Skip("args")
    g=a.get
    if name in("Bash","bash","run_terminal_cmd"):
        if not isinstance(g("command"),str): raise Skip("args")
        o={"command":g("command")}
        if isinstance(g("timeout"),int): o["timeout"]=g("timeout")
        return "bash",o
    if name in("Read","read","read_file"):
        p=g("file_path") or g("path")
        if not isinstance(p,str): raise Skip("args")
        o={"path":p}
        off=g("offset") if g("offset") is not None else g("start_line")
        lim=g("limit")
        if lim is None and g("end_line") is not None and g("start_line") is not None: lim=g("end_line")-g("start_line")+1
        if isinstance(off,int): o["offset"]=off
        if isinstance(lim,int) and lim<=1500: o["limit"]=lim   # huge end_line == "read to end"
        return "read",o
    if name in("Write","write","write_file"):
        p=g("file_path") or g("path"); c=g("content") if g("content") is not None else g("new_string")
        if not isinstance(p,str) or not isinstance(c,str): raise Skip("args")
        return "write",{"path":p,"content":c}
    if name in("Edit",):
        if g("replace_all"): raise Skip("replace_all")
        if not all(isinstance(g(k),str) for k in("file_path","old_string","new_string")): raise Skip("args")
        return "edit",{"path":g("file_path"),"edits":[{"oldText":g("old_string"),"newText":g("new_string")}]}
    if name=="edit":
        e=g("edits")
        if not isinstance(g("path"),str) or not isinstance(e,list) or not e: raise Skip("args")
        return "edit",{"path":g("path"),"edits":[{"oldText":x["oldText"],"newText":x["newText"]} for x in e]}
    if name in("Grep","grep_search"):
        if not isinstance(g("pattern"),str): raise Skip("args")
        o={"pattern":g("pattern")}
        if isinstance(g("path"),str): o["path"]=g("path")
        return "grep",o
    if name in("Glob","glob"):
        if not isinstance(g("pattern"),str): raise Skip("args")
        o={"pattern":g("pattern")}
        if isinstance(g("path"),str): o["path"]=g("path")
        return "glob",o
    if name in("WebSearch","web_search"):
        if not isinstance(g("query"),str): raise Skip("args")
        return "web_search",{"query":g("query")}
    if name in("WebFetch","read_url"):
        if not isinstance(g("url"),str): raise Skip("args")
        o={"url":g("url")}
        if isinstance(g("prompt"),str): o["prompt"]=g("prompt")
        return "web_fetch",o
    raise Skip("tool:"+name)
def asst(reasoning,content,calls):
    m={"role":"assistant","content":content or ""}
    if reasoning: m["reasoning_content"]=reasoning
    if calls: m["tool_calls"]=[{"type":"function","function":{"name":n,"arguments":a}} for n,a in calls]
    return m
def ntok(msgs,tools=TOOLS): return len(enc(render(msgs,tools)))+sum(m["role"]=="assistant" for m in msgs)
def window(msgs,tools=TOOLS,maxtok=MAXTOK,min_asst=1):
    """msgs: [system,user,...]. Longest prefix ending at an assistant msg with <=maxtok tokens.
    returns (prefix_msgs, is_full) or None."""
    idx=[i for i,m in enumerate(msgs) if m["role"]=="assistant"]
    if not idx: return None
    lo,hi,best=0,len(idx)-1,-1
    while lo<=hi:
        mid=(lo+hi)//2
        if ntok(msgs[:idx[mid]+1],tools)<=maxtok: best=mid;lo=mid+1
        else: hi=mid-1
    if best+1<min_asst or best<0: return None
    return msgs[:idx[best]+1], best==len(idx)-1 and idx[best]==len(msgs)-1
# ---- decontamination (same as data/v1/DATA.md: word-level lowercase 13-gram vs HumanEval+/MBPP+ prompt+solution)
W=lambda s:re.findall(r"\w+",s.lower())
grams=lambda w,n:{" ".join(w[i:i+n]) for i in range(len(w)-n+1)}
_B=None
def bench():
    global _B
    if _B is None:
        B=set(); R="../raw/ep/"
        for r in pq.read_table(glob.glob(R+"humanevalplus/*.parquet")[0]).to_pylist(): B|=grams(W(r["prompt"]+"\n"+r["canonical_solution"]),13)
        for r in pq.read_table(glob.glob(R+"mbppplus/*.parquet")[0]).to_pylist(): B|=grams(W(r["prompt"]+"\n"+r["code"]),13)
        _B=B
    return _B
def contaminated(msgs):
    t="\n".join((m.get("content") or "")+" "+(m.get("reasoning_content") or "")+" "+json.dumps(m.get("tool_calls",""),ensure_ascii=False) for m in msgs)
    return bool(grams(W(t),13)&bench())
def finish(src,msgs,tools=TOOLS):
    ids,mask=build(msgs,tools); assert len(ids)<=MAXTOK+8 and len(ids)==len(mask)
    return {"src":src,"ids":ids,"mask":mask}
