import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

// The assistant introduces itself as Astrax. The backend is a local model; the
// identity is set here so the whole product reads as one system.
const SYSTEM_PROMPT = {
  role: "system",
  content: [
    "You are Astrax, a local assistant.",
    "",
    "Identity:",
    "- Your name is Astrax. If asked who you are, say you are Astrax.",
    "- You run locally on the user's own machine. You are not a cloud service.",
    "",
    "Capabilities you may state:",
    "- Hold a multi-turn conversation and remember earlier messages in the chat.",
    "- Answer questions, explain concepts, summarize, translate, and write code.",
    "- Read a web page when the user provides a URL: the page text is fetched",
    "  and given to you as context, so answer from that text.",
    "- Reply in the user's language (Chinese or English).",
    "",
    "Rules:",
    "- When web page content is provided in the conversation, treat it as the",
    "  source of truth for questions about that page. Never invent facts about a",
    "  page you were not given. If asked what something is built with, only claim",
    "  it after seeing it in the provided text; otherwise say you do not know.",
    "- If a question is unclear, ask for clarification instead of guessing.",
    "- Be concise. Use Markdown for lists, code blocks, and tables.",
    "- Do not claim abilities you do not have (no internet browsing on your own,",
    "  no tool access beyond what is provided in the conversation).",
  ].join("\n"),
};

// Talks to the local llama-server through the Vite proxy. Streaming is done
// with the OpenAI-compatible /v1/chat/completions endpoint.
// Finds http(s) URLs in a message.
function extractUrls(text) {
  const matches = text.match(/https?:\/\/[^\s]+/gi) || [];
  return [...new Set(matches)];
}

// Fetches each URL through the dev-server endpoint and returns a context block.
async function fetchContext(urls) {
  const parts = [];
  for (const url of urls) {
    try {
      const response = await fetch("/api/fetch-url?url=" + encodeURIComponent(url));
      if (!response.ok) continue;
      const data = await response.json();
      if (data.text) {
        parts.push("Content of " + url + ":\n" + data.text);
      }
    } catch {
      // ignore individual fetch failures
    }
  }
  return parts.join("\n\n");
}

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
      // If the message contains links, read them first and hand the page text
      // to the model as context so it does not have to guess.
      const urls = extractUrls(text);
      if (urls.length > 0) {
        const context = await fetchContext(urls);
        if (context) {
          history.splice(history.length - 1, 0, {
            role: "system",
            content:
              "The user referenced the following web page content. Treat it as " +
              "the source of truth and do not invent details about it:\n\n" +
              context,
          });
        }
      }
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
