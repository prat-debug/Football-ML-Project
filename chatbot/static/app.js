/* Player-clustering chatbot UI.
 *
 * - Owns a session_id (UUID) stored in localStorage so the conversation
 *   survives a page refresh. Same key sends the same history back to the
 *   server, which keeps the per-session engine alive.
 * - POST /api/chat returns { reply, steps[] }; we render the steps as a
 *   collapsible trace under the assistant message so the user can see what
 *   the model asked for.
 * - The textarea auto-grows up to a max-height. Enter sends, Shift+Enter
 *   inserts a newline.
 *
 * Design intent: this is a notebook, not a chat widget. Assistant messages
 * are manuscript-style with a serif capital letter marker and a left rule;
 * user messages are right-aligned marginal notes. The Outputs panel uses
 * the same typography rules: Fraunces for headings, JetBrains Mono for
 * metadata, no shadows or gradients.
 */

(() => {
  'use strict';

  // ---------- Element refs ----------
  const messagesEl  = document.getElementById('messages');
  const messageList = document.getElementById('messageList');
  const formEl      = document.getElementById('composer');
  const inputEl     = document.getElementById('input');
  const sendBtn     = document.getElementById('sendBtn');
  const resetBtn    = document.getElementById('resetBtn');
  const tabChat     = document.getElementById('tabChat');
  const tabOutputs  = document.getElementById('tabOutputs');
  const panelChat    = document.getElementById('panelChat');
  const panelOutputs = document.getElementById('panelOutputs');
  const sessionLabel = document.getElementById('sessionIdLabel');
  const questionCountLabel = document.getElementById('questionCount');

  // ---------- Session ----------
  const SESSION_KEY = 'pcb.session_id';
  let sessionId = localStorage.getItem(SESSION_KEY);
  if (!sessionId) {
    sessionId = crypto.randomUUID();
    localStorage.setItem(SESSION_KEY, sessionId);
  }
  // Show the session id in the sidebar (truncated). A real product would
  // hide this, but it's useful while developing.
  if (sessionLabel) sessionLabel.textContent = sessionId.slice(0, 8) + '…';

  let questionCount = 0;
  if (questionCountLabel) {
    // Restore from localStorage so the counter survives a reload.
    const saved = parseInt(localStorage.getItem('pcb.question_count') || '0', 10);
    if (!Number.isNaN(saved)) {
      questionCount = saved;
      questionCountLabel.textContent = String(saved);
    }
  }
  function bumpQuestionCount() {
    questionCount += 1;
    if (questionCountLabel) questionCountLabel.textContent = String(questionCount);
    localStorage.setItem('pcb.question_count', String(questionCount));
  }

  // ---------- Auto-grow textarea ----------
  function autosize() {
    inputEl.style.height = 'auto';
    inputEl.style.height = Math.min(inputEl.scrollHeight, 160) + 'px';
    sendBtn.disabled = inputEl.value.trim().length === 0;
  }
  inputEl.addEventListener('input', autosize);
  autosize();

  inputEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      formEl.requestSubmit();
    }
  });

  // ---------- Markdown rendering (minimal, no external dep) ----------
  function escapeHtml(s) {
    return s.replace(/[&<>"']/g, (c) => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    }[c]));
  }

  function renderMarkdown(text) {
    const codeBlocks = [];
    text = text.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
      const idx = codeBlocks.length;
      codeBlocks.push(`<pre><code>${escapeHtml(code)}</code></pre>`);
      return `\u0000CODEBLOCK${idx}\u0000`;
    });

    text = text.replace(/`([^`\n]+)`/g, (_, code) => `<code>${escapeHtml(code)}</code>`);

    text = escapeHtml(text);

    text = text.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    text = text.replace(/(^|[^*])\*([^*\n]+)\*/g, '$1<em>$2</em>');

    text = text.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_, label, url) =>
      `<a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">${label}</a>`);

    text = text.replace(/(^|\n)((?:[-*] .+(?:\n|$))+)/g, (m, lead, block) => {
      const items = block.trim().split('\n').map((l) => `<li>${l.replace(/^[-*] /, '')}</li>`).join('');
      return `${lead}<ul>${items}</ul>`;
    });

    text = text
      .split(/\n{2,}/)
      .map((chunk) => {
        if (chunk.startsWith('\u0000CODEBLOCK')) return chunk;
        if (chunk.startsWith('<ul>')) return chunk;
        if (chunk.startsWith('<pre>')) return chunk;
        return `<p>${chunk.replace(/\n/g, '<br>')}</p>`;
      })
      .join('\n');

    text = text.replace(/\u0000CODEBLOCK(\d+)\u0000/g, (_, i) => codeBlocks[+i]);
    return text;
  }

  // ---------- Message rendering ----------
  function scrollToBottom() {
    messagesEl.scrollTo({ top: messagesEl.scrollHeight, behavior: 'smooth' });
  }

  function clearWelcome() {
    // Strip the welcome zone (heading + example grid) but keep the
    // messageList wrapper, which is the parent of every future message.
    document.querySelectorAll('.welcome-zone').forEach((el) => el.remove());
  }

  // Generate a stable capital letter for the assistant marker. The first
  // letter of the response is the obvious choice: short, distinctive, and
  // it varies per message so the eye can scan a long conversation.
  function assistantMarker(text) {
    const trimmed = (text || '').trim();
    if (!trimmed) return '·';
    const first = trimmed[0].toUpperCase();
    return /[A-Z]/.test(first) ? first : '·';
  }

  function addUserMessage(text) {
    clearWelcome();
    const row = document.createElement('div');
    row.className = 'bubble-user animate-fade-in';
    row.innerHTML = `
      <div>
        <div class="body">${escapeHtml(text)}</div>
        <div class="meta">You asked</div>
      </div>
      <div class="marker">YOU</div>`;
    messageList.appendChild(row);
    scrollToBottom();
  }

  function addTypingIndicator() {
    clearWelcome();
    const row = document.createElement('div');
    row.id = 'typingIndicator';
    row.className = 'bubble-assistant animate-fade-in';
    row.innerHTML = `
      <div class="marker">·</div>
      <div class="body">
        <div class="meta">Assistant · thinking</div>
        <div><span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span></div>
      </div>`;
    messageList.appendChild(row);
    scrollToBottom();
  }

  function removeTypingIndicator() {
    document.getElementById('typingIndicator')?.remove();
  }

  function formatArgs(args) {
    try {
      const s = JSON.stringify(args, null, 2);
      return s.length > 800 ? s.slice(0, 800) + '\n…' : s;
    } catch {
      return String(args);
    }
  }

  function addAssistantMessage(reply, steps) {
    removeTypingIndicator();

    const traceHtml = (steps && steps.length)
      ? `
        <details class="trace">
          <summary>${steps.length} tool call${steps.length === 1 ? '' : 's'}</summary>
          ${steps.map((s, i) => `
            <div class="step">
              <span class="step-num">step ${String(i + 1).padStart(2, '0')}</span>
              <span class="step-tool">${escapeHtml(s.tool)}</span>
              <pre>${escapeHtml(formatArgs(s.args))}</pre>
              <pre>${escapeHtml(s.result_preview)}</pre>
            </div>
          `).join('')}
        </details>`
      : '';

    const body = reply && reply.trim().length
      ? `<div class="prose-msg">${renderMarkdown(reply)}</div>`
      : `<div class="text-ink3 italic text-sm">(no response)</div>`;

    const row = document.createElement('div');
    row.className = 'bubble-assistant animate-fade-in';
    row.innerHTML = `
      <div class="marker">${escapeHtml(assistantMarker(reply))}</div>
      <div class="body">
        <div class="meta">Assistant</div>
        ${body}
        ${traceHtml}
      </div>`;
    messageList.appendChild(row);
    scrollToBottom();
  }

  function addError(message) {
    removeTypingIndicator();
    const row = document.createElement('div');
    row.className = 'bubble-assistant bubble-error animate-fade-in';
    row.innerHTML = `
      <div class="marker">!</div>
      <div class="body">
        <div class="meta">Error</div>
        <div>${escapeHtml(message)}</div>
      </div>`;
    messageList.appendChild(row);
    scrollToBottom();
  }

  // ---------- Send ----------
  let inFlight = false;

  async function send(text) {
    const trimmed = text.trim();
    if (!trimmed || inFlight) return;
    inFlight = true;
    sendBtn.disabled = true;

    addUserMessage(trimmed);
    bumpQuestionCount();
    inputEl.value = '';
    autosize();
    addTypingIndicator();

    try {
      const r = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, message: trimmed }),
      });
      if (!r.ok) {
        const detail = await r.json().catch(() => ({ detail: r.statusText }));
        throw new Error(detail.detail || `HTTP ${r.status}`);
      }
      const data = await r.json();
      addAssistantMessage(data.reply, data.steps);
    } catch (err) {
      addError(err.message || String(err));
    } finally {
      inFlight = false;
      sendBtn.disabled = inputEl.value.trim().length === 0;
      inputEl.focus();
    }
  }

  formEl.addEventListener('submit', (e) => {
    e.preventDefault();
    send(inputEl.value);
  });

  document.querySelectorAll('.example').forEach((btn) => {
    btn.addEventListener('click', () => {
      const prompt = btn.querySelector('div:last-child').textContent.trim();
      send(prompt);
    });
  });

  resetBtn.addEventListener('click', async () => {
    if (inFlight) return;
    await fetch('/api/reset', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId }),
    });
    localStorage.removeItem('pcb.question_count');
    localStorage.removeItem(SESSION_KEY);
    window.location.reload();
  });

  inputEl.focus();

  // ====================== Tab switching ======================
  function switchTab(which) {
    const isChat = which === 'chat';
    panelChat.classList.toggle('hidden', !isChat);
    panelOutputs.classList.toggle('hidden', isChat);
    tabChat.setAttribute('aria-selected', isChat ? 'true' : 'false');
    tabOutputs.setAttribute('aria-selected', isChat ? 'false' : 'true');
    // Active tab: ink underline; inactive: transparent.
    for (const [btn, active] of [[tabChat, isChat], [tabOutputs, !isChat]]) {
      btn.classList.toggle('border-ink', active);
      btn.classList.toggle('border-transparent', !active);
      btn.classList.toggle('text-ink', active);
      btn.classList.toggle('text-ink3', !active);
    }
    if (!isChat && !outputs.loaded) outputs.load();
  }
  tabChat.addEventListener('click', () => switchTab('chat'));
  tabOutputs.addEventListener('click', () => switchTab('outputs'));


  // ====================== Outputs panel ======================
  const outputs = (() => {
    const listEl       = document.getElementById('outputList');
    const statusEl     = document.getElementById('outputListStatus');
    const titleEl      = document.getElementById('outputTitle');
    const descEl       = document.getElementById('outputDescription');
    const searchEl     = document.getElementById('outputSearch');
    const downloadEl   = document.getElementById('outputDownload');
    const headEl       = document.getElementById('outputTableHead');
    const bodyEl       = document.getElementById('outputTableBody');
    const pagerInfoEl  = document.getElementById('outputPagerInfo');
    const prevBtn      = document.getElementById('outputPrevBtn');
    const nextBtn      = document.getElementById('outputNextBtn');

    const PAGE = 200;
    const state = {
      loaded: false,
      files: [],
      current: null,
    };

    function renderFileList() {
      listEl.innerHTML = '';
      if (state.files.length === 0) {
        statusEl.textContent = 'No CSV files in outputs/.';
        return;
      }
      statusEl.textContent = `${state.files.length} ${state.files.length === 1 ? 'file' : 'files'} · outputs/`;
      for (const f of state.files) {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'file-item';
        btn.dataset.name = f.name;
        btn.innerHTML = `
          <span class="name">${escapeHtml(f.name)}</span>
          <span class="meta">${f.rows.toLocaleString()} rows · ${formatBytes(f.size_bytes)}</span>`;
        btn.addEventListener('click', () => selectFile(f.name));
        listEl.appendChild(btn);
      }
    }

    function setActiveFile(name) {
      listEl.querySelectorAll('.file-item').forEach((el) => {
        el.classList.toggle('active', el.dataset.name === name);
      });
    }

    async function selectFile(name) {
      setActiveFile(name);
      const file = state.files.find((f) => f.name === name);
      titleEl.textContent = `${name}.csv`;
      descEl.textContent = (file && file.description) || '';

      try {
        const r = await fetch(`/api/outputs/${encodeURIComponent(name)}?limit=${PAGE}&offset=0`);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const data = await r.json();
        state.current = {
          name,
          columns: data.columns,
          rows: data.rows,
          total: data.total,
          offset: data.offset,
          sortKey: null,
          sortDir: 'asc',
        };
        downloadEl.classList.remove('hidden');
        downloadEl.onclick = () => triggerDownload(name);
        renderTable();
      } catch (err) {
        bodyEl.innerHTML = `<tr><td class="text-warm">Failed to load: ${escapeHtml(err.message)}</td></tr>`;
      }
    }

    function renderTable() {
      const cur = state.current;
      if (!cur) return;

      headEl.innerHTML = '';
      const tr = document.createElement('tr');
      for (const col of cur.columns) {
        const th = document.createElement('th');
        const isSorted = cur.sortKey === col;
        th.className = isSorted ? `sorted-${cur.sortDir}` : '';
        th.dataset.col = col;
        th.innerHTML = `${escapeHtml(col)}<span class="sort-ind"></span>`;
        th.addEventListener('click', () => toggleSort(col));
        tr.appendChild(th);
      }
      headEl.appendChild(tr);

      bodyEl.innerHTML = '';
      const filter = (searchEl.value || '').trim().toLowerCase();
      const visibleRows = cur.rows.filter((row) => {
        if (!filter) return true;
        for (const cell of row) {
          if (cell != null && String(cell).toLowerCase().includes(filter)) return true;
        }
        return false;
      });

      if (visibleRows.length === 0) {
        const tr2 = document.createElement('tr');
        const td = document.createElement('td');
        td.colSpan = cur.columns.length || 1;
        td.className = 'text-center text-ink3 py-6';
        td.textContent = cur.rows.length === 0 ? 'No rows in this file.' : 'No rows match the filter.';
        tr2.appendChild(td);
        bodyEl.appendChild(tr2);
      } else {
        const frag = document.createDocumentFragment();
        for (const row of visibleRows) {
          const tr2 = document.createElement('tr');
          for (let i = 0; i < cur.columns.length; i++) {
            const td = document.createElement('td');
            const v = row[i];
            td.textContent = v == null ? '' : String(v);
            if (isNumericLike(v)) td.classList.add('num');
            tr2.appendChild(td);
          }
          frag.appendChild(tr2);
        }
        bodyEl.appendChild(frag);
      }

      const start = cur.total === 0 ? 0 : cur.offset + 1;
      const end = cur.offset + cur.rows.length;
      pagerInfoEl.textContent = cur.total === 0
        ? '—'
        : `Showing ${start.toLocaleString()}–${end.toLocaleString()} of ${cur.total.toLocaleString()}`;
      prevBtn.disabled = cur.offset === 0;
      nextBtn.disabled = end >= cur.total;
    }

    function toggleSort(col) {
      const cur = state.current;
      if (!cur) return;
      if (cur.sortKey === col) {
        cur.sortDir = cur.sortDir === 'asc' ? 'desc' : 'asc';
      } else {
        cur.sortKey = col;
        cur.sortDir = 'asc';
      }
      const idx = cur.columns.indexOf(col);
      const dir = cur.sortDir === 'asc' ? 1 : -1;
      cur.rows.sort((a, b) => {
        const av = a[idx], bv = b[idx];
        const an = parseFloat(av), bn = parseFloat(bv);
        if (!Number.isNaN(an) && !Number.isNaN(bn)) return (an - bn) * dir;
        return String(av ?? '').localeCompare(String(bv ?? '')) * dir;
      });
      renderTable();
    }

    async function changePage(delta) {
      const cur = state.current;
      if (!cur) return;
      const newOffset = Math.max(0, cur.offset + delta * PAGE);
      if (newOffset === cur.offset) return;
      try {
        const r = await fetch(`/api/outputs/${encodeURIComponent(cur.name)}?limit=${PAGE}&offset=${newOffset}`);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const data = await r.json();
        cur.rows = data.rows;
        cur.offset = data.offset;
        cur.sortKey = null;
        cur.sortDir = 'asc';
        renderTable();
      } catch (err) {
        pagerInfoEl.textContent = `Failed to page: ${err.message}`;
      }
    }
    prevBtn.addEventListener('click', () => changePage(-1));
    nextBtn.addEventListener('click', () => changePage(1));

    let filterTimer = null;
    searchEl.addEventListener('input', () => {
      clearTimeout(filterTimer);
      filterTimer = setTimeout(renderTable, 100);
    });

    async function load() {
      try {
        const r = await fetch('/api/outputs');
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const data = await r.json();
        state.files = data.outputs || [];
        state.loaded = true;
        renderFileList();
      } catch (err) {
        statusEl.textContent = `Failed to load: ${err.message}`;
      }
    }

    return { load, get loaded() { return state.loaded; } };
  })();

  // ---------- Helpers ----------
  function formatBytes(n) {
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
    return `${(n / 1024 / 1024).toFixed(2)} MB`;
  }

  function isNumericLike(v) {
    if (v == null) return false;
    if (typeof v === 'number') return true;
    return /^-?\d+(\.\d+)?$/.test(String(v).trim());
  }

  function triggerDownload(name) {
    const a = document.createElement('a');
    a.href = `/api/outputs/${encodeURIComponent(name)}/download`;
    a.download = `${name}.csv`;
    document.body.appendChild(a);
    a.click();
    a.remove();
  }
})();
