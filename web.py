"""
web.py — крошечный веб-интерфейс к RAG на стандартной библиотеке (без Flask).

Зачем: показать продуктовую сторону RAG, а не только консоль — потоковый ответ
(SSE), панель источников с цитатами [#id], индикатор уверенности и список
query-трансформаций (HyDE/подзапросы). Один файл, нулевые зависимости.

Запуск:  python app.py web --port 8000  ->  http://127.0.0.1:8000
"""
from __future__ import annotations

import json
from urllib.parse import urlparse, parse_qs
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ragbench.retrieval import retrieve, MODES
from ragbench.answer import confidence, stream_answer

PAGE = """<!doctype html><html lang=ru><meta charset=utf-8>
<title>RAG-стенд</title>
<style>
 body{font:15px/1.5 system-ui,Segoe UI,Arial;margin:0;background:#0f1117;color:#e6e6e6}
 .wrap{max-width:880px;margin:0 auto;padding:24px}
 h1{font-size:20px;margin:0 0 4px} .sub{color:#8a93a6;font-size:13px;margin-bottom:18px}
 .row{display:flex;gap:8px;margin-bottom:14px}
 input,select,button{font:inherit;background:#1a1e27;color:#e6e6e6;border:1px solid #2c3240;border-radius:8px;padding:10px}
 input{flex:1} button{cursor:pointer;background:#2d6cdf;border-color:#2d6cdf}
 button:disabled{opacity:.5}
 #answer{background:#161a22;border:1px solid #2c3240;border-radius:10px;padding:16px;min-height:48px;white-space:pre-wrap}
 .conf{height:6px;background:#2c3240;border-radius:4px;overflow:hidden;margin:12px 0 4px}
 .conf>i{display:block;height:100%;background:#3fbf6f}
 .meta{color:#8a93a6;font-size:13px;margin:4px 0 14px}
 .src{background:#161a22;border:1px solid #2c3240;border-left:3px solid #2d6cdf;border-radius:8px;padding:10px 12px;margin:8px 0}
 .src b{color:#9db4ff} .tag{color:#6f7889;font-size:12px}
 .note{color:#d8a657;font-size:13px} .hyde{color:#8a93a6;font-size:13px;font-style:italic}
</style>
<div class=wrap>
 <h1>RAG-стенд по регламенту компании</h1>
 <div class=sub id=prov>…</div>
 <div class=row>
   <input id=q placeholder="Напр.: сколько суточных при поездке в Москву?" autofocus>
   <select id=mode></select>
   <button id=go>Спросить</button>
 </div>
 <div class=conf><i id=confbar style=width:0></i></div>
 <div class=meta id=meta></div>
 <div id=answer></div>
 <div id=sources></div>
</div>
<script>
const $=s=>document.querySelector(s);
fetch('/api/info').then(r=>r.json()).then(d=>{
  $('#prov').textContent=`провайдер: ${d.provider} · эмбеддер: ${d.embedder} · LLM: ${d.llm}`;
  d.modes.forEach(m=>{const o=document.createElement('option');o.value=o.textContent=m;if(m=='hybrid')o.selected=1;$('#mode').append(o)});
});
let es;
function ask(){
  const q=$('#q').value.trim(); if(!q)return;
  if(es)es.close();
  $('#answer').textContent=''; $('#sources').innerHTML=''; $('#meta').textContent='…'; $('#confbar').style.width=0;
  $('#go').disabled=true;
  es=new EventSource('/api/ask?mode='+$('#mode').value+'&q='+encodeURIComponent(q));
  es.addEventListener('meta',e=>{
    const m=JSON.parse(e.data);
    $('#confbar').style.width=Math.round(m.confidence*100)+'%';
    $('#meta').textContent=`уверенность ${m.confidence.toFixed(2)} (${m.conf_label}) · режим ${m.mode}`;
    if(m.expansions&&m.expansions.length) $('#meta').innerHTML+=` · <span class=hyde>расширение: ${m.expansions.map(x=>x.slice(0,80)).join(' | ')}</span>`;
    if(m.notes&&m.notes.length) $('#meta').innerHTML+=` · <span class=note>${m.notes.join('; ')}</span>`;
    m.sources.forEach((s,i)=>{
      const d=document.createElement('div');d.className='src';
      d.innerHTML=`<b>${i+1}. [#${s.id}]</b> <span class=tag>${s.section||''} · ${s.source} · score ${s.score.toFixed(3)}</span><br>${s.preview}`;
      $('#sources').append(d);
    });
  });
  es.addEventListener('token',e=>{$('#answer').textContent+=JSON.parse(e.data).t});
  es.addEventListener('done',e=>{es.close();$('#go').disabled=false});
  es.onerror=()=>{es.close();$('#go').disabled=false};
}
$('#go').onclick=ask; $('#q').addEventListener('keydown',e=>{if(e.key=='Enter')ask()});
</script>"""


def make_handler(idx, llm, cfg, info):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):           # тише в консоли
            pass

        def _send(self, code, ctype, body=b""):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.end_headers()
            if body:
                self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/":
                return self._send(200, "text/html; charset=utf-8", PAGE.encode("utf-8"))
            if u.path == "/api/info":
                body = json.dumps({"provider": info["resolved"], "embedder": info["embedder"],
                                   "llm": info["llm"], "modes": MODES}).encode("utf-8")
                return self._send(200, "application/json", body)
            if u.path == "/api/ask":
                return self._sse_ask(parse_qs(u.query))
            if u.path == "/favicon.ico":
                return self._send(204, "text/plain")
            self._send(404, "text/plain", b"not found")

        def _sse(self, event, data):
            payload = f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
            self.wfile.write(payload.encode("utf-8"))
            self.wfile.flush()

        def _sse_ask(self, params):
            q = (params.get("q") or [""])[0].strip()
            mode = (params.get("mode") or ["hybrid"])[0]
            if not q:
                return self._send(400, "text/plain", b"no query")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            rr = retrieve(idx, q, mode=mode, k=cfg.top_k, llm=llm, pool=cfg.candidate_pool)
            conf, label = confidence(idx, q, rr.hits)
            self._sse("meta", {
                "mode": mode, "confidence": conf, "conf_label": label,
                "notes": rr.notes, "expansions": rr.expansions,
                "sources": [{"id": h.id, "section": h.chunk.section, "source": h.chunk.source,
                             "score": h.score, "preview": h.chunk.text.replace("\n", " ")[:200]}
                            for h in rr.hits],
            })
            try:
                for tok in stream_answer(idx, q, rr, llm=llm, cfg=cfg):
                    self._sse("token", {"t": tok})
            except Exception as e:
                self._sse("token", {"t": f"\n[ошибка генерации: {e}]"})
            self._sse("done", {})
    return H


def serve(idx, llm, cfg, info, port=8000):
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(idx, llm, cfg, info))
    print(f"RAG-стенд: http://127.0.0.1:{port}  (провайдер: {info['resolved']}, Ctrl+C — стоп)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nстоп.")
