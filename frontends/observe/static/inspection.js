'use strict';

import { button, clear, el } from './dom.js';
import { fetchJson } from './api.js';

const PAGE_CHARS = 32768;

/** 【请求查看】【共享查询】展示指定运行的真实请求；参数为面板、归属和选择代次校验，返回无 */
export async function openRequestDirectory(root, scope, isCurrent) {
  await renderDirectory(root, { ...scope, action: 'requests' }, { isCurrent, generation: 0 });
}

/** 【请求查看】【目录分页】按稳定游标读取一页目录；参数为选择与代次，返回无 */
async function renderDirectory(root, query, navigation) {
  const generation = ++navigation.generation;
  const isCurrent = () => navigation.isCurrent() && generation === navigation.generation;
  try {
    const page = await inspect(query);
    if (!isCurrent()) return;
    clear(root);
    const controls = el('div', 'code-actions');
    if (query.action === 'attempts') {
      controls.appendChild(button('返回请求', 'ghost-btn', () => renderDirectory(root,
        { session_id: query.session_id, run_id: query.run_id, action: 'requests' }, navigation)));
    }
    root.appendChild(controls);
    for (const item of page.items) {
      const label = query.action === 'requests'
        ? `请求 ${item.request_index} · ${item.status}`
        : `尝试 ${item.attempt_index} · ${item.model} · ${item.status}`;
      root.appendChild(button(label, 'ghost-btn', () => query.action === 'requests'
        ? renderDirectory(root, { ...query, request_id: item.request_id, action: 'attempts', cursor: '' }, navigation)
        : renderDetail(root, { ...query, attempt_id: item.attempt_id, action: 'detail', section: 'request', offset: 0 }, navigation)));
    }
    if (!page.items.length) root.appendChild(el('p', '', '当前范围没有已保存的请求'));
    if (page.next_cursor) controls.appendChild(button('下一页', 'ghost-btn', () =>
      renderDirectory(root, { ...query, cursor: page.next_cursor }, navigation)));
  } catch (error) {
    if (isCurrent()) root.appendChild(el('p', '', `读取失败：${error.message}`));
  }
}

/** 【请求查看】【正文分页】按需读取输入、返回和真实来源；参数为尝试选择，返回无 */
async function renderDetail(root, query, navigation) {
  const generation = ++navigation.generation;
  const isCurrent = () => navigation.isCurrent() && generation === navigation.generation;
  try {
    const page = await inspect({ ...query, limit: PAGE_CHARS });
    if (!isCurrent()) return;
    clear(root);
    const controls = el('div', 'code-actions');
    controls.appendChild(button('返回尝试', 'ghost-btn', () => renderDirectory(root,
      { session_id: query.session_id, run_id: query.run_id, request_id: query.request_id, action: 'attempts' }, navigation)));
    for (const [section, label] of [['request', '实际输入'], ['response', '模型返回'], ['sources', '材料来源']]) {
      controls.appendChild(button(label, 'ghost-btn', () => renderDetail(root, { ...query, section, offset: 0 }, navigation)));
    }
    if (query.offset > 0) controls.appendChild(button('上一页', 'ghost-btn', () =>
      renderDetail(root, { ...query, offset: Math.max(0, query.offset - PAGE_CHARS) }, navigation)));
    if (page.has_more) controls.appendChild(button('下一页', 'ghost-btn', () =>
      renderDetail(root, { ...query, offset: page.next_offset }, navigation)));
    controls.appendChild(button('复制本页', 'ghost-btn', () => navigator.clipboard.writeText(page.text)
      .catch((error) => window.alert(`复制失败：${error.message}`))));
    const pre = el('pre', 'code-block');
    pre.textContent = page.text;
    root.append(controls, el('p', '', `${page.status || ''} · ${page.retention || ''} · ${page.offset}/${page.total_chars}`), pre);
  } catch (error) {
    if (isCurrent()) root.appendChild(el('p', '', `读取失败：${error.message}`));
  }
}

/** 【请求查看】【共享查询】调用与TUI相同的领域服务；参数为查询字段，返回页面 */
function inspect(query) {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(query)) {
    if (value !== undefined && value !== null && value !== '') params.set(key, String(value));
  }
  return fetchJson(`/api/inspection?${params}`);
}
