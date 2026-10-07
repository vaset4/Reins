'use strict';

import { button, clear, downloadText, el, emptyState } from './dom.js';
import { loadRaw } from './api.js';
import { openRequestDirectory } from './inspection.js';

const TABS = [
  { key: 'prompt', label: '提示词' },
  { key: 'messages', label: '消息' },
  { key: 'tools', label: '工具' },
  { key: 'response', label: '响应' },
  { key: 'raw', label: '原始数据' },
];

export function createEvidenceController(root, titleNode) {
  let current = null;
  let generation = 0;
  return {
    /** 打开当前运行的请求目录；参数为会话及运行身份，返回无 */
    async openRequests(sessionId, runId) {
      const selected = ++generation;
      current = { type: 'requests', sessionId, runId };
      titleNode.textContent = '请求记录';
      await openRequestDirectory(root, { session_id: sessionId, run_id: runId }, () => selected === generation);
    },
    async openModel(call) {
      generation += 1;
      current = { type: 'model', call };
      titleNode.textContent = `模型调用 ${call.index}`;
      await renderModelEvidence(root, call);
    },
    openJson(title, data) {
      generation += 1;
      current = { type: 'json', data };
      titleNode.textContent = title;
      renderJson(root, data);
    },
    clear() {
      generation += 1;
      current = null;
      titleNode.textContent = '证据';
      clear(root);
      root.appendChild(emptyState('选择一个步骤', '点击模型调用、工具调用、风险提示或原始路径查看证据。'));
    },
    fullscreen() {
      if (!current) return;
      const code = root.querySelector('.code-block');
      const text = code?.textContent || JSON.stringify(current, null, 2);
      openFullscreen(titleNode.textContent, text);
    },
  };
}

async function renderModelEvidence(root, call) {
  clear(root);
  const paths = call.evidence || {};
  const request = await safeRaw(paths.model_request);
  const response = await safeRaw(paths.model_response);
  const parsed = await safeRaw(paths.parsed_plan);
  const tabs = buildTabs(root);
  const payload = { request, response, parsed, paths };
  for (const tab of TABS) {
    tabs.add(tab, () => renderEvidenceTab(tab.key, payload));
  }
  tabs.select('prompt');
}

function buildTabs(root) {
  const bar = el('div', 'evidence-tabs');
  const panel = el('div', 'evidence-tab-panel');
  root.append(bar, panel);
  const renderers = new Map();
  return {
    add(tab, render) {
      const node = button(tab.label, 'evidence-tab', () => this.select(tab.key));
      node.dataset.tab = tab.key;
      renderers.set(tab.key, render);
      bar.appendChild(node);
    },
    select(key) {
      for (const tab of bar.querySelectorAll('button')) {
        tab.classList.toggle('active', tab.dataset.tab === key);
      }
      clear(panel);
      panel.appendChild(renderers.get(key)());
    },
  };
}

function renderEvidenceTab(key, payload) {
  if (key === 'prompt') return codeBlock(promptText(payload.request));
  if (key === 'messages') return messagesView(payload.request);
  if (key === 'tools') return codeBlock(toolsText(payload.request));
  if (key === 'response') return codeBlock(responseText(payload.response, payload.parsed));
  return codeBlock(JSON.stringify(payload, null, 2));
}

function messagesView(raw) {
  const data = raw.data || {};
  const messages = data.request?.messages || [];
  if (!messages.length) return emptyState('没有消息', rawStatusLabel(raw.status));
  const wrap = el('div', 'message-list');
  for (const msg of messages) {
    const card = el('article', 'message-card');
    card.appendChild(el('h4', '', roleLabel(msg.role)));
    card.appendChild(codeBlock(msg.content || ''));
    wrap.appendChild(card);
  }
  return wrap;
}

function codeBlock(text) {
  const wrap = el('div', 'code-wrap');
  const actions = el('div', 'code-actions');
  actions.appendChild(button('复制', 'ghost-btn', () => copyText(text)));
  actions.appendChild(button('保存', 'ghost-btn', () => downloadText('evidence.txt', text)));
  const pre = el('pre', 'code-block');
  pre.textContent = text || '—';
  wrap.append(actions, pre);
  return wrap;
}

function renderJson(root, data) {
  clear(root);
  root.appendChild(codeBlock(JSON.stringify(data, null, 2)));
}

async function safeRaw(path) {
  if (!path || path.startsWith('(')) {
    return { status: 'missing', data: null, path };
  }
  try {
    const raw = await loadRaw(path);
    raw.path = path;
    return raw;
  } catch (err) {
    return { status: 'error', data: String(err), path };
  }
}

function promptText(raw) {
  const data = raw.data || {};
  return data.render_text_to_model || data.request?.messages?.map((m) => `${roleLabel(m.role)}: ${m.content}`).join('\n\n') || rawStatusLabel(raw.status);
}

function toolsText(raw) {
  const data = raw.data || {};
  return JSON.stringify(data.request?.tools || [], null, 2);
}

function responseText(response, parsed) {
  const data = response.data?.response || {};
  return [
    `是否成功: ${data.ok}`,
    `令牌: ${data.prompt_tokens || 0}/${data.completion_tokens || 0}/${data.total_tokens || 0}`,
    '',
    data.text || '',
    '',
    '解析结果:',
    JSON.stringify(parsed.data || {}, null, 2),
  ].join('\n');
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
  } catch (err) {
    window.alert(`复制失败：${err.message || err}`);
  }
}

function openFullscreen(title, text) {
  const w = window.open('', '_blank');
  if (!w) return;
  w.document.write(fullscreenHtml(title, text));
  w.document.close();
}

function fullscreenHtml(title, text) {
  return `<!doctype html><title>${escapeHtml(title)}</title><style>
body{margin:0;background:#f6f7f3;color:#1f271f;font:14px/1.6 Inter,Arial,sans-serif}
header{position:sticky;top:0;padding:14px 18px;background:#fff;border-bottom:1px solid #d9dfd5}
pre{white-space:pre-wrap;overflow-wrap:anywhere;margin:0;padding:18px;font:12px/1.65 Consolas,monospace}
</style><header>${escapeHtml(title)}</header><pre>${escapeHtml(text || '')}</pre>`;
}

function roleLabel(role) {
  const labels = {
    assistant: '助手',
    developer: '开发者',
    system: '系统',
    tool: '工具',
    user: '用户',
  };
  return labels[role] || role || '未知角色';
}

function rawStatusLabel(status) {
  const labels = {
    error: '读取失败',
    missing: '缺失',
    missing_file: '文件缺失',
    ok: '正常',
  };
  return labels[status] || status || '未知状态';
}

function escapeHtml(text) {
  return text.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
