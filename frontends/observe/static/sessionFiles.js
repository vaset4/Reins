'use strict';

import { button, chip, el, emptyState } from './dom.js';

const STATE_LABELS = {
  compatibility_task_id: '兼容任务',
  consecutive_readonly_count: '连续只读',
  focus_task_id: '焦点任务',
  hint_injection_count: '提示注入',
  last_checkpoint_at: '保存时间',
  last_checkpoint_id: 'Checkpoint ID',
  last_checkpoint_reason: '原因',
  last_checkpoint_state: '状态',
  last_run_event: '最后事件',
  last_run_id: 'Run ID',
  last_run_status: '状态',
  original_user_goal: '会话目标',
  recent_run_ids: '最近 run',
  schema_version: '协议版本',
  session_id: 'Session ID',
  summary: '会话摘要',
  updated_at: '更新时间',
  writeback_targets: '写回目标',
};

export function renderSessionFiles(inventory, evidence) {
  if (!inventory || inventory.status !== 'ok') {
    return emptyState('没有会话记录清单', '后端没有返回这个会话的持久记录信息。');
  }
  const section = el('section', 'story-section session-files');
  const raw = inventory.state?.raw || {};
  section.append(renderHeader(inventory, evidence), renderMetrics(inventory.summary || {}));
  section.append(renderSessionBrief(raw, evidence));
  section.append(renderSummaryFile(inventory.summary_file || {}, evidence));
  section.append(renderAdvancedEvidence(inventory, evidence));
  return section;
}

function renderHeader(inventory, evidence) {
  const head = el('div', 'session-files-head');
  const copy = el('div');
  copy.appendChild(el('h3', '', '会话证据浏览器'));
  copy.appendChild(el('p', '', `${inventory.root || inventory.session_id} 中的会话状态、摘要和跨运行记录。`));
  head.append(copy, button('打开完整清单', 'outline-btn', () => evidence.openJson('会话记录清单', inventory)));
  return head;
}

function renderMetrics(summary) {
  const row = el('div', 'coverage-chips');
  row.append(
    chip('独立会话文件', summary.top_level_files ?? 0, 'evidence'),
    chip('运行数', summary.runs ?? 0, 'model'),
    chip('事实行', summary.facts ?? 0, 'tool'),
    chip('诊断记录', summary.raw_files ?? 0, 'evidence'),
    chip('错误流', summary.error_files ?? 0, summary.error_files ? 'failed' : ''),
    chip('checkpoint', summary.checkpoint_events ?? 0, 'model'),
    chip('生命周期', summary.lifecycle_events ?? 0, 'tool'),
    chip('历史迁移', summary.legacy_state_transitions ?? 0, 'tool'),
    chip('上下文快照', summary.context_segments ?? 0, 'evidence'),
  );
  return row;
}

function renderSessionBrief(raw, evidence) {
  const grid = el('div', 'session-brief-grid');
  grid.append(
    briefCard('会话目标', fieldText(raw, 'original_user_goal'), [['Session', raw.session_id]], evidence, raw),
    briefCard('最近运行', fieldText(raw, 'last_run_id'), latestRows(raw), evidence, pick(raw, ['last_run_id', 'last_run_status', 'last_run_event', 'updated_at'])),
    briefCard('Checkpoint', fieldText(raw, 'last_checkpoint_state'), checkpointRows(raw), evidence, pick(raw, ['last_checkpoint_id', 'last_checkpoint_state', 'last_checkpoint_reason', 'last_checkpoint_at'])),
    writebackCard(raw.writeback_targets || {}, evidence),
  );
  return grid;
}

function briefCard(title, main, rows, evidence, payload) {
  const card = button('', 'session-brief-card', () => evidence.openJson(title, payload || {}));
  card.appendChild(el('span', 'brief-label', title));
  card.appendChild(el('b', 'brief-main', main || '—'));
  card.appendChild(metaList(rows));
  return card;
}

function writebackCard(targets, evidence) {
  const active = Object.entries(targets).filter(([, enabled]) => enabled).map(([name]) => name);
  const card = button('', 'session-brief-card', () => evidence.openJson('写回目标', targets));
  card.appendChild(el('span', 'brief-label', '写回目标'));
  card.appendChild(el('b', 'brief-main', active.length ? active.join(', ') : '无写回目标'));
  card.appendChild(metaList([['关闭', disabledTargets(targets)]]));
  return card;
}

function latestRows(raw) {
  return [
    ['状态', statusLabel(raw.last_run_status)],
    ['最后事件', raw.last_run_event],
    ['更新时间', raw.updated_at],
  ];
}

function checkpointRows(raw) {
  return [
    ['原因', raw.last_checkpoint_reason],
    ['时间', raw.last_checkpoint_at],
  ];
}

function renderSummaryFile(summary, evidence) {
  const panel = el('section', 'session-summary-panel');
  const head = el('div', 'session-subhead');
  head.append(el('h4', '', '会话摘要'), el('span', '', fileStatus(summary)));
  panel.appendChild(head);
  panel.appendChild(textBlock(summary.text || '没有 会话摘要 内容。'));
  panel.appendChild(actionBar([button('打开 会话摘要', 'ghost-btn', () => evidence.openJson('会话摘要', summary))]));
  return panel;
}

function renderAdvancedEvidence(inventory, evidence) {
  const details = browserDetails('高级证据：持久记录与会话状态', false);
  details.appendChild(renderGaps(inventory.not_displayed || []));
  details.appendChild(renderTopLevel(inventory.top_level || [], evidence));
  details.appendChild(renderState(inventory.state || {}, evidence));
  return details;
}

function renderTopLevel(files, evidence) {
  const box = el('section', 'advanced-block');
  box.appendChild(el('h4', '', `顶层文件 ${files.length}`));
  const list = el('div', 'coverage-list');
  for (const file of files) {
    const row = button('', 'coverage-row top-file-row', () => evidence.openJson(file.kind || '顶层文件', file));
    row.append(el('b', '', file.kind || '文件'), el('code', '', file.path || '—'));
    row.append(el('span', '', file.present ? `${file.bytes || 0} B` : '缺失'));
    list.appendChild(row);
  }
  box.appendChild(list);
  return box;
}

function renderState(state, evidence) {
  const box = el('section', 'advanced-block');
  box.appendChild(el('h4', '', '会话状态 原始字段'));
  box.appendChild(actionBar([button('打开 会话状态', 'ghost-btn', () => evidence.openJson('会话状态', state.raw || {}))]));
  for (const group of state.groups || []) box.appendChild(renderStateGroup(group, evidence));
  if (!(state.groups || []).length) box.appendChild(emptyState('没有 会话状态 字段', '会话状态文件缺失或为空。'));
  return box;
}

function renderStateGroup(group, evidence) {
  const box = el('section', 'state-group');
  box.appendChild(el('h4', '', group.title || group.key || '状态字段'));
  const list = el('div', 'state-field-list');
  for (const field of group.fields || []) {
    const row = button('', 'state-field-row', () => evidence.openJson(`state.${field.name}`, field));
    row.append(el('b', '', fieldLabel(field.name)), el('code', '', field.preview || '—'));
    list.appendChild(row);
  }
  box.appendChild(list);
  return box;
}

function renderGaps(gaps) {
  const wrap = el('div', 'coverage-gap-grid');
  for (const gap of gaps) {
    const card = el('article', 'coverage-gap');
    card.appendChild(el('b', '', gap.title || '尚未聚合的证据'));
    card.appendChild(el('span', '', gap.detail || ''));
    wrap.appendChild(card);
  }
  return wrap;
}

function browserDetails(title, open) {
  const details = el('details', 'coverage-details');
  details.open = open;
  details.appendChild(el('summary', '', title));
  return details;
}

function actionBar(actions) {
  const bar = el('div', 'session-file-actions');
  for (const action of actions) bar.appendChild(action);
  return bar;
}

function textBlock(text) {
  const pre = el('pre', 'summary-text');
  pre.textContent = text || '—';
  return pre;
}

function metaList(rows) {
  const list = el('dl', 'brief-meta');
  for (const [key, value] of rows) {
    list.append(el('dt', '', key), el('dd', '', formatBrief(value)));
  }
  return list;
}

function fieldText(raw, name) {
  return formatBrief(raw?.[name]);
}

function fieldLabel(name) {
  return STATE_LABELS[name] || name || '字段';
}

function formatBrief(value) {
  if (value === null || value === undefined || value === '') return '—';
  if (Array.isArray(value)) return `${value.length} 项`;
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
}

function disabledTargets(targets) {
  return Object.entries(targets).filter(([, enabled]) => !enabled).map(([name]) => name).join(', ') || '—';
}

function fileStatus(file) {
  if (!file.present) return '缺失';
  return `${file.bytes || 0} B`;
}

function pick(source, keys) {
  const out = {};
  for (const key of keys) out[key] = source?.[key];
  return out;
}

function statusLabel(status) {
  const labels = {
    done: '已完成',
    failed: '失败',
    paused: '已暂停',
    waiting_approval: '等待审批',
    waiting_user: '等待用户',
    running: '运行中',
  };
  return labels[status] || status || '未知';
}
