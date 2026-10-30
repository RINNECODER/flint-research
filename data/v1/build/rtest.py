import sys; sys.path.insert(0,__import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from render import *
msgs=[{"role":"system","content":"sys"},{"role":"user","content":"hi"},
{"role":"assistant","reasoning_content":"think a","content":"I'll run","tool_calls":[{"type":"function","function":{"name":"run_python","arguments":{"code":"print(1)","stdin":""}}}]},
{"role":"tool","content":"1\n"},{"role":"assistant","content":"done"}]
tools=[{"type":"function","function":{"name":"run_python","description":"d","parameters":{"type":"object","properties":{"code":{"type":"string"}}}}}]
print(repr(render(msgs,tools)))
ids,mask=build(msgs,tools)
print(repr(tok.decode([i for i,m in zip(ids,mask) if m],skip_special_tokens=False)))
print(repr(tok.decode(ids,skip_special_tokens=False)))
