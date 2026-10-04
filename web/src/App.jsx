import { useEffect, useRef, useState } from "react";

// Talks to the local llama-server through the Vite proxy. Streaming is done
// with the OpenAI-compatible /v1/chat/completions endpoint.
async function streamChat(messages, onDelta, signal) {
  const response = await fetch("/v1/chat/completions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages, stream: true, max_tokens: 512 }),
    signal,
  });
  if (!response.ok || !response.body) {
    throw new Error("request failed: " + response.status);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed.startsWith("data:")) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === "[DONE]") return;
      try {
        const parsed = JSON.parse(payload);
        const delta = parsed.choices?.[0]?.delta?.content;
        if (delta) onDelta(delta);
      } catch {
        // ignore malformed keep-alive chunks
      }
    }
  }
}

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const abortRef = useRef(null);
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function send() {
    const text = input.trim();
    if (!text || busy) return;
    const history = [...messages, { role: "user", content: text }];
    setMessages([...history, { role: "assistant", content: "" }]);
    setInput("");
    setBusy(true);
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      await streamChat(history, (delta) => {
        setMessages((current) => {
          const next = current.slice();
          next[next.length - 1] = {
            role: "assistant",
            content: next[next.length - 1].content + delta,
          };
          return next;
        });
      }, controller.signal);
    } catch (error) {
      if (error.name !== "AbortError") {
        setMessages((current) => {
          const next = current.slice();
          next[next.length - 1] = {
            role: "assistant",
            content: "Error: " + error.message,
          };
          return next;
        });
      }
    } finally {
      setBusy(false);
      abortRef.current = null;
    }
  }

  function stop() {
    abortRef.current?.abort();
  }

  function reset() {
    stop();
    setMessages([]);
  }

  function onKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      send();
    }
  }

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <span className="logo">◆</span>
          <span className="name">Astrax</span>
        </div>
        <button className="ghost" onClick={reset}>New chat</button>
      </header>

      <main className="messages">
        {messages.length === 0 && (
          <div className="empty">
            <div className="empty-logo">◆</div>
            <h1>Astrax</h1>
            <p>Ask anything. Runs locally on your machine.</p>
          </div>
        )}
        {messages.map((message, index) => (
          <div key={index} className={"row " + message.role}>
            <div className={"bubble " + message.role}>{message.content || "…"}</div>
          </div>
        ))}
        <div ref={bottomRef} />
      </main>

      <footer className="composer">
        <textarea
          value={input}
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Message Astrax…"
          rows={1}
        />
        {busy ? (
          <button className="send stop" onClick={stop}>Stop</button>
        ) : (
          <button className="send" onClick={send} disabled={!input.trim()}>Send</button>
        )}
      </footer>
    </div>
  );
}
