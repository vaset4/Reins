'use strict';

export async function fetchJson(url) {
  const resp = await fetch(url);
  if (!resp.ok) {
    throw new Error(`HTTP ${resp.status}: ${await responseText(resp)}`);
  }
  return resp.json();
}

export async function loadHealth() {
  return fetchJson('/api/health');
}

export async function loadPanels() {
  return fetchJson('/api/panels');
}

export async function loadSessions() {
  return fetchJson('/api/sessions');
}

export async function loadRuns(sessionId) {
  return fetchJson(`/api/sessions/${encodeURIComponent(sessionId)}/runs`);
}

export async function loadSessionFiles(sessionId) {
  return fetchJson(`/api/sessions/${encodeURIComponent(sessionId)}/files`);
}

export async function loadStory(runId) {
  return fetchJson(`/api/runs/${encodeURIComponent(runId)}/story`);
}

export async function loadPanel(runId, panelId) {
  const rid = encodeURIComponent(runId);
  const pid = encodeURIComponent(panelId);
  return fetchJson(`/api/runs/${rid}/panels/${pid}`);
}

export async function loadRaw(path) {
  return fetchJson(`/api/raw?path=${encodeURIComponent(path)}`);
}

async function responseText(resp) {
  try {
    return JSON.stringify(await resp.json());
  } catch (err) {
    return resp.text();
  }
}
