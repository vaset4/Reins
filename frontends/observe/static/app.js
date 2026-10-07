'use strict';

import { fetchJson, loadHealth, loadPanels, loadRuns, loadSessions, loadSessionFiles, loadStory } from './api.js';
import { button, clear, el, shortId } from './dom.js';
import { createEvidenceController } from './evidence.js';
import { renderSessionFiles } from './sessionFiles.js';
import { renderStory } from './story.js';

const state = {
  sessions: [],
  panels: [],
  runs: [],
  sessionFiles: null,
  selectedSessionId: '',
  selectedRunId: '',
  autoRefresh: true,
  timer: 0,
};

const els = {
  sessionList: document.querySelector('#session-list'),
  sessionCount: document.querySelector('#session-count'),
  search: document.querySelector('#search-input'),
  runList: document.querySelector('#run-list'),
  runTitle: document.querySelector('#run-title'),
  story: document.querySelector('#story-root'),
  empty: document.querySelector('#empty-state'),
  dataRoot: document.querySelector('#data-root-badge'),
  refresh: document.querySelector('#btn-refresh'),
  auto: document.querySelector('#auto-refresh'),
  evidenceBody: document.querySelector('#evidence-body'),
  evidenceTitle: document.querySelector('#evidence-title'),
  evidenceClose: document.querySelector('#evidence-close'),
  evidenceFull: document.querySelector('#evidence-fullscreen'),
  debugPanels: document.querySelector('#debug-panels'),
};

const evidence = createEvidenceController(els.evidenceBody, els.evidenceTitle);

async function init() {
  evidence.clear();
  await loadShellData();
  bindEvents();
  scheduleRefresh();
}

async function loadShellData() {
  const [health, panels, sessions] = await Promise.all([loadHealth(), loadPanels(), loadSessions()]);
  state.panels = panels || [];
  state.sessions = sessions.sessions || [];
  renderHealth(health);
  renderSessions();
  if (state.sessions.length && !state.selectedSessionId) {
    await selectSession(state.sessions[0].session_id);
  }
}

function bindEvents() {
  els.search.addEventListener('input', renderSessions);
  els.refresh.addEventListener('click', refreshSelected);
  els.auto.addEventListener('change', () => {
    state.autoRefresh = els.auto.checked;
    scheduleRefresh();
  });
  els.evidenceClose.addEventListener('click', evidence.clear);
  els.evidenceFull.addEventListener('click', evidence.fullscreen);
}

function renderHealth(health) {
  const parts = String(health.data_root || '').split(/[\\/]/);
  els.dataRoot.textContent = parts.slice(-3).join('/') || '数据目录';
  els.dataRoot.title = health.data_root || '';
}

function renderSessions() {
  clear(els.sessionList);
  const query = els.search.value.trim().toLowerCase();
  const sessions = state.sessions.filter((s) => sessionMatches(s, query));
  els.sessionCount.textContent = String(sessions.length);
  for (const session of sessions) els.sessionList.appendChild(sessionButton(session));
}

function sessionButton(session) {
  const node = document.createElement('button');
  node.type = 'button';
  node.className = `session-item ${session.session_id === state.selectedSessionId ? 'active' : ''}`;
  node.appendChild(el('b', '', shortId(session.session_id)));
  node.appendChild(el('span', '', `${statusLabel(session.status)} · ${(session.summary || '').slice(0, 80)}`));
  node.addEventListener('click', () => selectSession(session.session_id));
  return node;
}

function sessionMatches(session, query) {
  if (!query) return true;
  return [session.session_id, session.last_run_id, session.summary, session.status]
    .join(' ')
    .toLowerCase()
    .includes(query);
}

async function selectSession(sessionId) {
  state.selectedSessionId = sessionId;
  const [data, files] = await Promise.all([loadRuns(sessionId), loadSessionFiles(sessionId)]);
  state.runs = [...(data.runs || [])].reverse();
  state.sessionFiles = files;
  renderSessions();
  renderRuns();
  if (state.runs.length) await selectRun(state.runs[0].run_id);
  if (!state.runs.length) clearSelectedRun();
}

function clearSelectedRun() {
  state.selectedRunId = '';
  els.runTitle.textContent = '没有运行';
  els.empty.hidden = false;
  clear(els.story);
  els.story.appendChild(button('请求记录', 'ghost-btn', () => evidence.openRequests(state.selectedSessionId, runId)));
  clear(els.debugPanels);
}

function renderRuns() {
  clear(els.runList);
  for (const run of state.runs) {
    const node = document.createElement('button');
    node.type = 'button';
    node.className = `run-pill ${run.run_id === state.selectedRunId ? 'active' : ''}`;
    node.textContent = `${shortId(run.run_id)} · ${statusLabel(run.status)}`;
    node.addEventListener('click', () => selectRun(run.run_id));
    els.runList.appendChild(node);
  }
}

async function selectRun(runId) {
  state.selectedRunId = runId;
  renderRuns();
  els.empty.hidden = true;
  els.runTitle.textContent = shortId(runId);
  const payload = await loadStory(runId);
  clear(els.story);
  els.story.appendChild(renderSessionFiles(state.sessionFiles, evidence));
  const runRoot = el('div', 'run-story-root');
  els.story.appendChild(runRoot);
  renderStory(runRoot, payload, evidence);
  await renderDebugPanels(runId);
}

async function renderDebugPanels(runId) {
  clear(els.debugPanels);
  const selected = state.panels.filter((p) => ['overview', 'timeline', 'token_accountant', 'tool_strip'].includes(p.id));
  for (const panel of selected) {
    const card = el('details', 'debug-panel');
    card.appendChild(el('summary', '', panelTitle(panel)));
    const body = el('pre', 'debug-json', '加载中...');
    card.appendChild(body);
    els.debugPanels.appendChild(card);
    fetchJson(`/api/runs/${encodeURIComponent(runId)}/panels/${panel.id}`)
      .then((data) => { body.textContent = JSON.stringify(data, null, 2); })
      .catch((err) => { body.textContent = String(err); });
  }
}

function panelTitle(panel) {
  const labels = {
    context_inspector: '上下文检查',
    overview: '总览',
    prompt_composition: '提示词组成',
    response_anatomy: '响应结构',
    timeline: '时间线',
    token_accountant: '令牌统计',
    tool_strip: '工具调用列表',
  };
  return labels[panel.id] || panel.title || panel.id;
}

function statusLabel(status) {
  const labels = {
    done: '已完成',
    failed: '失败',
    paused: '已暂停',
    running: '运行中',
  };
  return labels[status] || status || '未知';
}

async function refreshSelected() {
  await loadShellData();
  if (state.selectedSessionId) await selectSession(state.selectedSessionId);
}

function scheduleRefresh() {
  if (state.timer) window.clearInterval(state.timer);
  if (!state.autoRefresh) return;
  state.timer = window.setInterval(async () => {
    const run = state.runs.find((item) => item.run_id === state.selectedRunId);
    if (!run || ['done', 'failed', 'paused'].includes(run.status)) return;
    await selectRun(state.selectedRunId);
  }, 1500);
}

init().catch((err) => {
  els.story.appendChild(el('div', 'error-box', `观察页面加载失败：${err.message}`));
});
