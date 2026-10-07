'use strict';

import { loadRaw } from './api.js';
import { button, chip, clear, el, emptyState, kvGrid, shortId, timePart } from './dom.js';

export function renderStory(root, payload, evidence) {
  clear(root);
  const story = payload.story;
  if (!story) {
    root.appendChild(emptyState('没有运行故事', '这个运行存在，但后端没有返回故事数据。'));
    return;
  }
  root.append(
    renderHero(story),
    renderWarnings(story.warnings || [], evidence),
    renderRunFiles(story.run_files || [], evidence),
    renderFactStream(story, evidence),
    renderContext(story.context || {}),
    renderDebug(story, evidence),
  );
}

function renderHero(story) {
  const overview = story.overview || {};
  const conv = story.conversation || {};
  const section = el('section', 'story-hero');
  const copy = el('div', 'story-copy');
  copy.appendChild(el('p', 'eyebrow', '运行故事'));
  copy.appendChild(el('h2', '', conv.user_goal || '未记录用户问题'));
  copy.appendChild(el('p', 'hero-note', '当前选中 run 的运行级状态、文件、facts 事件和可钻取证据。'));
  const meta = el('div', 'chip-row');
  meta.append(
    chip('状态', statusLabel(overview.status), overview.status || ''),
    chip('模型调用', overview.model_calls ?? 0, 'model'),
    chip('工具调用', overview.tool_calls ?? 0, 'tool'),
    chip('事实数', story.raw?.facts_count ?? 0, 'evidence'),
  );
  copy.appendChild(meta);
  section.append(copy, kvGrid([
    ['会话', shortId(overview.session_id)],
    ['运行', shortId(overview.run_id)],
    ['更新时间', overview.updated_at || '—'],
    ['摘要类型', summaryKindLabel(overview.summary_kind)],
  ]));
  return section;
}

function renderWarnings(warnings, evidence) {
  const section = el('section', 'story-section');
  section.appendChild(el('h3', '', '风险提示'));
  if (!warnings.length) {
    section.appendChild(emptyState('未发现风险提示', '仅表示确定性检查没有命中已知问题。'));
    return section;
  }
  const list = el('div', 'warning-grid');
  for (const warn of warnings) {
    const card = button('', `warning-card warning-${warn.kind || 'info'}`, () => evidence.openJson('风险提示证据', warn));
    card.appendChild(el('b', '', warningKindLabel(warn.kind)));
    card.appendChild(el('span', '', warn.message || ''));
    list.appendChild(card);
  }
  section.appendChild(list);
  return section;
}

function renderRunFiles(files, evidence) {
  const section = el('section', 'story-section');
  section.appendChild(el('h3', '', `运行文件 ${files.length}`));
  if (!files.length) {
    section.appendChild(emptyState('没有运行文件清单', '后端没有返回当前 run 的文件列表。'));
    return section;
  }
  const list = el('div', 'run-file-list');
  for (const file of files) list.appendChild(fileButton(file, evidence));
  section.appendChild(list);
  return section;
}

function fileButton(file, evidence) {
  const row = button('', `run-file-row ${file.present ? '' : 'missing'}`.trim(), () => openFile(file, evidence));
  row.append(el('b', '', file.kind || '文件'), el('code', '', file.path || file.name || '—'));
  row.append(el('span', '', file.present ? `${file.bytes || 0} B` : '缺失'));
  return row;
}

function renderFactStream(story, evidence) {
  const facts = story.fact_stream || [];
  const section = el('section', 'story-section');
  section.appendChild(el('h3', '', `运行事实流 ${facts.length}`));
  if (!facts.length) {
    section.appendChild(emptyState('没有运行事实', '这个 run 没有可展示的 facts.jsonl 行。'));
    return section;
  }
  const list = el('div', 'fact-stream-list');
  for (const fact of facts) list.appendChild(factButton(fact, evidence));
  section.appendChild(list);
  return section;
}

function factButton(fact, evidence) {
  const row = button('', `fact-row fact-${eventTone(fact.event)}`, () => openFact(fact, evidence));
  row.append(el('span', 'loop-time', timePart(fact.timestamp)), el('b', '', `${fact.index}. ${fact.title || fact.event}`));
  row.append(el('span', 'fact-detail', fact.detail || '—'), el('code', '', fact.event || 'event'));
  return row;
}

function openFact(fact, evidence) {
  if (fact.event === 'llm:response' && fact.model_call) {
    evidence.openModel(fact.model_call);
    return;
  }
  evidence.openJson(`${fact.index}. ${fact.title || fact.event}`, fact.raw || fact);
}

async function openFile(file, evidence) {
  if (!file.present) {
    evidence.openJson('文件缺失', file);
    return;
  }
  try {
    evidence.openJson(file.name || file.path, await loadRaw(file.path));
  } catch (err) {
    evidence.openJson('文件读取失败', { file, error: String(err) });
  }
}

function eventTone(event) {
  if (event === 'llm:response') return 'model';
  if (event?.startsWith('tool:')) return 'tool';
  if (event === 'run:lifecycle' || event === 'state:transition' || event === 'checkpoint:saved') return 'state';
  return 'plain';
}

function renderContext(context) {
  const section = el('section', 'story-section');
  section.appendChild(el('h3', '', '上下文组成'));
  const segments = context.segments || [];
  if (!segments.length) {
    section.appendChild(emptyState('没有上下文分段', '这次运行可能早于上下文分段事实记录。'));
    return section;
  }
  const total = context.total_tokens_est || 1;
  const list = el('div', 'segment-list');
  for (const seg of segments) list.appendChild(renderSegment(seg, total));
  section.appendChild(list);
  return section;
}

function renderSegment(seg, total) {
  const row = el('div', 'segment-row');
  row.appendChild(el('span', 'segment-name', seg.name || '分段'));
  const track = el('span', 'segment-track');
  const fill = el('span', `segment-fill segment-${seg.layer || 'dynamic'}`);
  fill.style.width = `${Math.max(3, ((seg.tokens_est || 0) / total) * 100)}%`;
  track.appendChild(fill);
  row.appendChild(track);
  row.appendChild(el('span', 'segment-count', `${seg.tokens_est || 0}`));
  return row;
}

function renderDebug(story, evidence) {
  const section = el('section', 'story-section');
  section.appendChild(el('h3', '', '调试证据'));
  const actions = el('div', 'debug-actions');
  for (const call of story.model_calls || []) {
    actions.appendChild(button(`打开模型 ${call.index}`, 'outline-btn', () => evidence.openModel(call)));
  }
  actions.appendChild(button('打开故事原始数据', 'outline-btn', () => evidence.openJson('运行故事原始数据', story)));
  section.appendChild(actions);
  return section;
}

function statusLabel(status) {
  const labels = {
    done: '已完成',
    failed: '失败',
    paused: '已暂停',
    waiting_approval: '等待审批',
    waiting_user: '等待用户',
    pending: '待返回',
    running: '运行中',
  };
  return labels[status] || status || '未知';
}

function summaryKindLabel(kind) {
  const labels = {
    latest_writeback: '最近一次写回',
  };
  return labels[kind] || kind || '—';
}

function warningKindLabel(kind) {
  const labels = {
    latency: '耗时',
    model: '模型',
    protocol: '协议',
    source: '来源',
    summary: '摘要',
  };
  return labels[kind] || kind || '提示';
}
