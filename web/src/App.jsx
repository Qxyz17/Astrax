import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

// The assistant introduces itself as Astrax. The backend is a local model; the
// identity is set here so the whole product reads as one system.
const SYSTEM_PROMPT = {
  role: "system",
  content:
    "You are Astrax, a helpful assistant running locally on the user's machine. " +
    "Answer clearly and concisely. Use Markdown for structure, lists, and code. " +
    "If a question is unclear, ask for clarification instead of guessing. " +
    "When the user writes in Chinese, answer in Chinese.",
};

// Talks to the local llama-server through the Vite proxy. Streaming is done
// with the OpenAI-compatible /v1/chat/completions endpoint.
async function streamChat(messages, onDelta, signal) {
  const response = await fetch("/v1/chat/completions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      messages: [SYSTEM_PROMPT, ...messages],
      stream: true,
      max_tokens: 1024,
      temperature: 0.7,
      repeat_penalty: 1.1,
    }),
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
  const textareaRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  function autoGrow() {
    const element = textareaRef.current;
    if (!element) return;
    element.style.height = "auto";
    element.style.height = Math.min(element.scrollHeight, 160) + "px";
  }

  async function send() {
    const text = input.trim();
    if (!text || busy) return;
    const history = [...messages, { role: "user", content: text }];
    setMessages([...history, { role: "assistant", content: "" }]);
    setInput("");
    requestAnimationFrame(autoGrow);
    setBusy(true);
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      await streamChat(history, (delta) => {
        setMessages((current) => {
          const next = current.slice();
          const last = next.length - 1;
          next[last] = { role: "assistant", content: next[last].content + delta };
          return next;
        });
      }, controller.signal);
    } catch (error) {
      if (error.name !== "AbortError") {
        setMessages((current) => {
          const next = current.slice();
          next[next.length - 1] = {
            role: "assistant",
            content: "**Error:** " + error.message,
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
    setInput("");
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
            {message.role === "assistant" ? (
              <div className="bubble assistant markdown">
                {message.content ? (
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {message.content}
                  </ReactMarkdown>
                ) : (
                  <span className="typing">…</span>
                )}
              </div>
            ) : (
              <div className="bubble user">{message.content}</div>
            )}
          </div>
        ))}
        <div ref={bottomRef} />
      </main>

      <footer className="composer">
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(event) => { setInput(event.target.value); autoGrow(); }}
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
