<script setup>
import { ref, nextTick } from 'vue'

const messages = ref([])          // { role: 'user' | 'bot', text, sources? }
const input = ref('')
const loading = ref(false)
const scroller = ref(null)

async function scrollToBottom() {
  await nextTick()
  if (scroller.value) scroller.value.scrollTop = scroller.value.scrollHeight
}

async function send() {
  const question = input.value.trim()
  if (!question || loading.value) return

  const apiBaseUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000'

  messages.value.push({ role: 'user', text: question })
  input.value = ''
  loading.value = true
  scrollToBottom()

  try {
    const res = await fetch(`${apiBaseUrl}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    })
    if (!res.ok) throw new Error(`Server returned ${res.status}`)
    const data = await res.json()
    messages.value.push({ role: 'bot', text: data.answer, sources: data.sources })
  } catch (err) {
    messages.value.push({
      role: 'bot',
      text: `Couldn't reach the backend. Is it running on port 8000? (${err.message})`,
      error: true,
    })
  } finally {
    loading.value = false
    scrollToBottom()
  }
}

function fileName(path) {
  return path.split(/[/\\]/).pop()
}
</script>

<template>
  <div class="app">
    <header class="header">
      <h1>Docs Chat</h1>
      <p>Ask anything about your documents.</p>
    </header>

    <main ref="scroller" class="thread">
      <div v-if="messages.length === 0" class="empty">
        Start by asking a question about the files you ingested.
      </div>

      <div v-for="(m, i) in messages" :key="i" class="row" :class="m.role">
        <div class="bubble" :class="{ error: m.error }">
          <p class="text">{{ m.text }}</p>
          <div v-if="m.sources && m.sources.length" class="sources">
            <span class="sources-label">Sources</span>
            <span v-for="s in m.sources" :key="s" class="chip">{{ fileName(s) }}</span>
          </div>
        </div>
      </div>

      <div v-if="loading" class="row bot">
        <div class="bubble typing"><span></span><span></span><span></span></div>
      </div>
    </main>

    <footer class="composer">
      <input
        v-model="input"
        type="text"
        placeholder="Ask a question…"
        @keyup.enter="send"
        :disabled="loading"
      />
      <button @click="send" :disabled="loading || !input.trim()">Send</button>
    </footer>
  </div>
</template>

<style>
:root {
  --bg: #0f1220;
  --panel: #171a2b;
  --user: #3b5bdb;
  --bot: #232741;
  --text: #e7e9f3;
  --muted: #9aa0b8;
  --accent: #6ee7b7;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); font-family: system-ui, sans-serif; }

.app {
  max-width: 760px;
  margin: 0 auto;
  height: 100vh;
  display: flex;
  flex-direction: column;
  color: var(--text);
}
.header { padding: 24px 20px 12px; }
.header h1 { margin: 0; font-size: 22px; }
.header p { margin: 4px 0 0; color: var(--muted); font-size: 14px; }

.thread {
  flex: 1;
  overflow-y: auto;
  padding: 16px 20px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.empty { color: var(--muted); text-align: center; margin: auto; }

.row { display: flex; }
.row.user { justify-content: flex-end; }
.row.bot { justify-content: flex-start; }

.bubble {
  max-width: 78%;
  padding: 12px 14px;
  border-radius: 14px;
  line-height: 1.5;
  font-size: 15px;
}
.row.user .bubble { background: var(--user); border-bottom-right-radius: 4px; }
.row.bot .bubble { background: var(--bot); border-bottom-left-radius: 4px; }
.bubble.error { background: #4a2130; }
.text { margin: 0; white-space: pre-wrap; }

.sources {
  margin-top: 10px;
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  align-items: center;
}
.sources-label { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }
.chip {
  font-size: 12px;
  background: rgba(110, 231, 183, 0.12);
  color: var(--accent);
  padding: 2px 8px;
  border-radius: 999px;
}

.typing { display: flex; gap: 4px; }
.typing span {
  width: 7px; height: 7px; border-radius: 50%;
  background: var(--muted);
  animation: blink 1.2s infinite both;
}
.typing span:nth-child(2) { animation-delay: 0.2s; }
.typing span:nth-child(3) { animation-delay: 0.4s; }
@keyframes blink { 0%, 80%, 100% { opacity: 0.3; } 40% { opacity: 1; } }

.composer {
  display: flex;
  gap: 8px;
  padding: 16px 20px 24px;
  border-top: 1px solid rgba(255, 255, 255, 0.06);
}
.composer input {
  flex: 1;
  padding: 12px 14px;
  border-radius: 12px;
  border: 1px solid rgba(255, 255, 255, 0.1);
  background: var(--panel);
  color: var(--text);
  font-size: 15px;
  outline: none;
}
.composer input:focus { border-color: var(--user); }
.composer button {
  padding: 0 20px;
  border: none;
  border-radius: 12px;
  background: var(--user);
  color: white;
  font-size: 15px;
  font-weight: 600;
  cursor: pointer;
}
.composer button:disabled { opacity: 0.5; cursor: not-allowed; }
</style>
