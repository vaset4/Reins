'use strict';

export function el(tag, className = '', text = '') {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}

export function clear(node) {
  node.replaceChildren();
}

export function button(label, className, onClick) {
  const node = el('button', className, label);
  node.type = 'button';
  node.addEventListener('click', onClick);
  return node;
}

export function chip(label, value = '', tone = '') {
  const node = el('span', `chip ${tone ? `chip-${tone}` : ''}`.trim());
  const strong = el('b', '', label);
  node.appendChild(strong);
  if (value !== '') node.appendChild(document.createTextNode(` ${value}`));
  return node;
}

export function kvGrid(items) {
  const grid = el('dl', 'kv-grid');
  for (const [key, value] of items) {
    const dt = el('dt', '', key);
    const dd = el('dd', '', formatValue(value));
    grid.append(dt, dd);
  }
  return grid;
}

export function emptyState(title, detail = '') {
  const box = el('div', 'empty-panel');
  box.appendChild(el('h3', '', title));
  if (detail) box.appendChild(el('p', '', detail));
  return box;
}

export function formatValue(value) {
  if (value === null || value === undefined || value === '') return '—';
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
}

export function shortId(value) {
  if (!value) return '—';
  return value.length > 18 ? `${value.slice(0, 18)}…` : value;
}

export function timePart(value) {
  if (!value || value.length < 19) return '—';
  return value.slice(11, 19);
}

export function downloadText(filename, text) {
  const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const link = el('a');
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}
