import json, jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer
M=__import__("os").path.expanduser("~/.cache/huggingface/hub/models--dima6312--Flint-40B-A2.5B-abliterated-Rec-4bit/snapshots/c0d2388f441399e43a0bdeccccf14f66043fedad/")
tok=Tokenizer.from_file(M+"tokenizer.json")
EOS=tok.token_to_id("<|end_of_text|>")
env=ImmutableSandboxedEnvironment(trim_blocks=True,lstrip_blocks=True)
env.filters["tojson"]=lambda x,ensure_ascii=False,indent=None,separators=None,sort_keys=False: json.dumps(x,ensure_ascii=ensure_ascii,indent=indent,separators=separators,sort_keys=sort_keys)
tpl=env.from_string(open(M+"chat_template.jinja").read())
def render(msgs,tools=None,gen=False):
    return tpl.render(messages=msgs,tools=tools,add_generation_prompt=gen)
enc=lambda s: tok.encode(s,add_special_tokens=False).ids
def build(msgs,tools=None):
    """returns ids, mask. Assistant content (after <|assistant|>) + EOS trained."""
    ids=[];mask=[];pos=0
    for i,m in enumerate(msgs):
        if m["role"]!="assistant": continue
        pre=render(msgs[:i],tools,True)          # ends with <|assistant|>
        upto=render(msgs[:i+1],tools)
        assert upto.startswith(pre) and pre.startswith(render(msgs[:i],tools)) , i
        assert pre[pos:] is not None
        p=enc(pre[pos:]); ids+=p; mask+=[0]*len(p)
        a=enc(upto[len(pre):])+[EOS]; ids+=a; mask+=[1]*len(a)
        pos=len(upto)
    full=render(msgs,tools)
    t=full[pos:]
    if t: p=enc(t); ids+=p; mask+=[0]*len(p)
    return ids,mask
