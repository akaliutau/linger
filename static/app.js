const qs = (sel) => document.querySelector(sel);
const stage1Input = qs('#stage1Input');
const stage1Preview = qs('#stage1Preview');
const stage1Empty = qs('#stage1Empty');
const stage1AnalyzeBtn = qs('#stage1AnalyzeBtn');
const stage1Status = qs('#stage1Status');
const stage1Result = qs('#stage1Result');
const sessionIdInput = qs('#sessionId');
const sessionNoteInput = qs('#sessionNote');
const configStrip = qs('#configStrip');
const liveVideo = qs('#liveVideo');
const liveCanvas = qs('#liveCanvas');
const cameraBtn = qs('#cameraBtn');
const liveBtn = qs('#liveBtn');
const enoughBtn = qs('#enoughBtn');
const coachText = qs('#coachText');
const intervalSec = qs('#intervalSec');
const intervalValue = qs('#intervalValue');
const liveCountdown = qs('#liveCountdown');
const liveKept = qs('#liveKept');
const liveLatency = qs('#liveLatency');
const bestGrid = qs('#bestGrid');
const feed = qs('#feed');
const chatInput = qs('#chatInput');
const chatBtn = qs('#chatBtn');
const voiceBtn = qs('#voiceBtn');
const voiceTranscript = qs('#voiceTranscript');
const finalizeStatus = qs('#finalizeStatus');
const speakToggle = qs('#speakToggle');
const installBtn = qs('#installBtn');

let stage1File = null;
let stage1Analysis = null;
let bestFrames = [];
let stream = null;
let ws = null;
let wsHeartbeat = null;
let wsReconnectTimer = null;
let liveActive = false;
let liveBusy = false;
let frameIndex = 0;
let nextCaptureAt = 0;
let recognition = null;
let deferredPrompt = null;
let lastSpoken = '';

function nowTime() {
  return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function logFeed(text) {
  const item = document.createElement('div');
  item.className = 'feed-item';
  item.innerHTML = `<time>${nowTime()}</time><div>${text}</div>`;
  feed.prepend(item);
}

function setCoach(text, speak = true) {
  coachText.textContent = text;
  if (speak && speakToggle.value === 'on' && 'speechSynthesis' in window) {
    if (lastSpoken !== text) {
      window.speechSynthesis.cancel();
      const utter = new SpeechSynthesisUtterance(text);
      utter.rate = 1;
      utter.pitch = 1;
      window.speechSynthesis.speak(utter);
      lastSpoken = text;
    }
  }
}

function ensureSessionId() {
  const raw = (sessionIdInput.value || '').trim();
  if (raw) return raw;
  const generated = `session-${Math.random().toString(36).slice(2, 8)}`;
  sessionIdInput.value = generated;
  return generated;
}

function fileToObjectUrl(file) {
  return URL.createObjectURL(file);
}

function renderStage1Result(data) {
  stage1Result.innerHTML = '';
  const cards = [
    ['Reading', `<strong>${data.object_label}</strong><div class="subtle">${data.one_line_summary}</div>`],
    ['Story signal', `<div>${data.story_signal}</div><div class="subtle">Quality ${data.quality_score_1_to_10}/10</div>`],
    ['Live goals', `<ul>${data.live_capture_goals.map((x) => `<li>${x}</li>`).join('')}</ul>`],
    ['Best angles', `<ul>${data.best_angles.map((x) => `<li>${x}</li>`).join('')}</ul>`],
    ['Avoid', `<ul>${data.things_to_avoid.map((x) => `<li>${x}</li>`).join('')}</ul>`],
  ];
  for (const [title, html] of cards) {
    const el = document.createElement('div');
    el.className = 'result-card';
    el.innerHTML = `<h3>${title}</h3>${html}`;
    stage1Result.appendChild(el);
  }
}

function applyStage1Ready(payload) {
  const analysis = payload.analysis || payload;
  stage1Analysis = analysis;

  renderStage1Result(analysis);

  const latency = payload.latency_ms ? `Done in ${payload.latency_ms} ms` : 'Stage 1 ready';
  const storage = payload.upload?.storage_mode ? ` · ${payload.upload.storage_mode}` : '';
  stage1Status.textContent = latency + storage;

  cameraBtn.disabled = false;
  liveBtn.disabled = !stream;
  enoughBtn.disabled = false;

  setCoach(analysis.opening_coach_line || 'Start the live hunt.', true);
  logFeed(`Stage 1 ready: ${analysis.object_label || 'item analyzed'}`);
}

function renderBestFrames() {
  bestGrid.innerHTML = '';
  if (!bestFrames.length) {
    bestGrid.innerHTML = '<div class="subtle">No kept frames yet.</div>';
    return;
  }
  for (const frame of bestFrames) {
    const card = document.createElement('div');
    card.className = 'thumb-card';
    const url = frame.preview_url || frame.local_preview_url || '';
    card.innerHTML = `
      <img src="${url}" alt="${frame.moment_label || 'frame'}" />
      <div class="thumb-meta">
        <strong>${frame.moment_label || 'Moment'}</strong>
        <div>Score ${frame.score_0_to_100}</div>
        <div class="subtle">${frame.cinematic_role || ''}</div>
        <div class="subtle">${frame.why || ''}</div>
      </div>
    `;
    bestGrid.appendChild(card);
  }
}

async function loadConfig() {
  const res = await fetch('/api/config');
  const cfg = await res.json();
  configStrip.innerHTML = '';
  [
    `storage ${cfg.storage_mode}`,
    `stage1 ${cfg.stage1_model}`,
    `frame ${cfg.frame_model}`,
    `live ${cfg.live_text_model}`,
    `vertex ${cfg.vertex_location}`,
  ].forEach((txt) => {
    const el = document.createElement('div');
    el.className = 'pill';
    el.textContent = txt;
    configStrip.appendChild(el);
  });
}

function connectWs() {
  const sessionId = ensureSessionId();

  if (wsReconnectTimer) {
    clearTimeout(wsReconnectTimer);
    wsReconnectTimer = null;
  }
  if (wsHeartbeat) {
    clearInterval(wsHeartbeat);
    wsHeartbeat = null;
  }
  if (ws) {
    try {
      ws.close();
    } catch {}
  }

  ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/session/${encodeURIComponent(sessionId)}`);

  ws.onopen = () => {
    logFeed('Live socket connected.');
    wsHeartbeat = setInterval(() => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send('ping');
      }
    }, 15000);
  };

  ws.onclose = () => {
    logFeed('Live socket closed.');
    if (wsHeartbeat) {
      clearInterval(wsHeartbeat);
      wsHeartbeat = null;
    }
    wsReconnectTimer = setTimeout(() => connectWs(), 1500);
  };

  ws.onerror = (err) => {
    console.error('ws error', err);
  };

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);

    if (data.type === 'stage1_ready') {
      applyStage1Ready(data);
    }
    if (data.type === 'live_guidance') {
      setCoach(data.text, true);
      logFeed(`Coach: ${data.text}`);
    }
    if (data.type === 'frame_result') {
      setCoach(data.guidance, true);
      liveKept.textContent = `Kept ${data.kept_total}`;
      liveLatency.textContent = `Avg ${data.avg_latency_ms} ms`;
      logFeed(`Frame ${data.frame.frame_index}: ${data.frame.score_0_to_100} ${data.selected ? 'kept' : 'skip'}`);
    }
    if (data.type === 'chat_reply') {
      setCoach(data.reply, true);
      logFeed(`You: ${data.user}<br/>Coach: ${data.reply}`);
    }
    if (data.type === 'finalized') {
      finalizeStatus.textContent = 'Finalized. Basket uploaded.';
      logFeed('Session finalized and basket exported.');
    }
  };
}

stage1Input.addEventListener('change', () => {
  const [file] = stage1Input.files || [];
  stage1File = file || null;
  if (!stage1File) return;
  stage1Preview.src = fileToObjectUrl(stage1File);
  stage1Preview.classList.remove('hidden');
  stage1Empty.classList.add('hidden');
  stage1AnalyzeBtn.disabled = false;
  connectWs();
});

stage1AnalyzeBtn.addEventListener('click', async () => {
  const sessionId = ensureSessionId();
  if (!stage1File) return;

  stage1Status.textContent = 'Analyzing item photo...';
  stage1AnalyzeBtn.disabled = true;

  try {
    const fd = new FormData();
    fd.append('session_id', sessionId);
    fd.append('note', sessionNoteInput.value || '');
    fd.append('file', stage1File);

    const res = await fetch('/api/stage1/analyze-photo', { method: 'POST', body: fd });
    const data = await res.json();

    if (!res.ok) {
      stage1Status.textContent = data.detail || 'Stage 1 failed';
      logFeed(`Stage 1 failed: ${data.detail || res.status}`);
      return;
    }

    applyStage1Ready(data);
  } catch (err) {
    console.error('stage1 analyze failed', err);
    stage1Status.textContent = `Stage 1 UI error: ${err?.message || err}`;
    logFeed(`Stage 1 UI error: ${err?.message || err}`);
  } finally {
    stage1AnalyzeBtn.disabled = false;
  }
});

intervalSec.addEventListener('input', () => {
  intervalValue.textContent = intervalSec.value;
});

cameraBtn.addEventListener('click', async () => {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { ideal: 'environment' },
        width: { ideal: 720 },
        height: { ideal: 1280 },
      },
      audio: false,
    });
    liveVideo.srcObject = stream;
    cameraBtn.textContent = 'Camera open';
    if (stage1Analysis) {
      liveBtn.disabled = false;
    }
    logFeed('Camera opened.');
  } catch (err) {
    logFeed(`Camera error: ${err}`);
    setCoach('Camera open failed. Check permissions.', false);
  }
});

async function startLive() {
  const sessionId = ensureSessionId();
  const fd = new FormData();
  fd.append('session_id', sessionId);
  const res = await fetch('/api/live/start', { method: 'POST', body: fd });
  const data = await res.json();
  if (res.ok) {
    setCoach(data.guidance, true);
    logFeed('Live hunt started.');
  }
}

function nextCountdownTick() {
  if (!liveActive) {
    liveCountdown.textContent = 'Idle';
    return;
  }
  const remain = Math.max(0, nextCaptureAt - Date.now());
  liveCountdown.textContent = remain <= 0 ? 'Capturing…' : `${Math.ceil(remain / 1000)}s`;
  requestAnimationFrame(nextCountdownTick);
}

async function captureFrame() {
  if (!liveActive || liveBusy || !stream) return;
  liveBusy = true;
  frameIndex += 1;
  const videoW = liveVideo.videoWidth || 720;
  const videoH = liveVideo.videoHeight || 1280;
  liveCanvas.width = videoW;
  liveCanvas.height = videoH;
  const ctx = liveCanvas.getContext('2d');
  ctx.drawImage(liveVideo, 0, 0, videoW, videoH);
  const blob = await new Promise((resolve) => liveCanvas.toBlob(resolve, 'image/jpeg', 0.82));
  const fd = new FormData();
  fd.append('session_id', ensureSessionId());
  fd.append('note', sessionNoteInput.value || '');
  fd.append('frame_index', String(frameIndex));
  fd.append('ts_ms', String(Date.now()));
  fd.append('file', blob, `frame-${frameIndex}.jpg`);
  const res = await fetch('/api/live/frame', { method: 'POST', body: fd });
  const data = await res.json();
  liveBusy = false;
  nextCaptureAt = Date.now() + Number(intervalSec.value) * 1000;
  if (!res.ok) {
    setCoach(data.detail || 'Frame analyze failed', false);
    return;
  }
  setCoach(data.guidance, true);
  liveKept.textContent = `Kept ${data.kept_total}`;
  liveLatency.textContent = `Avg ${data.avg_latency_ms} ms`;
  if (data.selected) {
    const localUrl = URL.createObjectURL(blob);
    bestFrames = [
      {
        ...data.candidate,
        local_preview_url: localUrl,
      },
      ...bestFrames.filter((x) => x.frame_index !== data.candidate.frame_index),
    ].sort((a, b) => b.score_0_to_100 - a.score_0_to_100).slice(0, 10);
    renderBestFrames();
  }
}

function liveLoop() {
  if (!liveActive) return;
  if (Date.now() >= nextCaptureAt && !liveBusy) captureFrame();
  requestAnimationFrame(liveLoop);
}

liveBtn.addEventListener('click', async () => {
  if (!stream) return;
  if (!liveActive) {
    liveActive = true;
    liveBtn.textContent = 'Stop live hunt';
    nextCaptureAt = Date.now();
    await startLive();
    nextCountdownTick();
    liveLoop();
  } else {
    liveActive = false;
    liveBtn.textContent = 'Start live hunt';
    liveCountdown.textContent = 'Stopped';
    logFeed('Live hunt stopped.');
  }
});

async function sendChatMessage(text) {
  if (!text.trim()) return;
  const fd = new FormData();
  fd.append('session_id', ensureSessionId());
  fd.append('message', text.trim());
  const res = await fetch('/api/live/chat', { method: 'POST', body: fd });
  const data = await res.json();
  if (res.ok) {
    setCoach(data.reply, true);
    logFeed(`You: ${text}<br/>Coach: ${data.reply}`);
  }
}

chatBtn.addEventListener('click', () => {
  const text = chatInput.value;
  chatInput.value = '';
  sendChatMessage(text);
});

chatInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    e.preventDefault();
    chatBtn.click();
  }
});

enoughBtn.addEventListener('click', async () => {
  finalizeStatus.textContent = 'Finalizing basket and pipeline hand-off...';
  const fd = new FormData();
  fd.append('session_id', ensureSessionId());
  fd.append('enough_moment', 'enough moment');
  const res = await fetch('/api/session/finalize', { method: 'POST', body: fd });
  const data = await res.json();
  if (!res.ok) {
    finalizeStatus.textContent = data.detail || 'Finalize failed';
    return;
  }
  finalizeStatus.textContent = `Done · ${data.finalized.basket_assets.length} basket images`;
  logFeed(`Finalize done. Basket assets: ${data.finalized.basket_assets.length}`);
  if (data.finalized.pipeline_result?.status && data.finalized.pipeline_result.status !== 'not_started') {
    logFeed(`Video pipeline: ${data.finalized.pipeline_result.status}`);
  }
});

function setupSpeechRecognition() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) {
    voiceBtn.disabled = true;
    voiceBtn.textContent = 'Mic notes unavailable';
    return;
  }
  recognition = new SR();
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.lang = 'en-US';
  let active = false;

  recognition.onresult = (event) => {
    let finalText = '';
    let interim = '';
    for (let i = event.resultIndex; i < event.results.length; i += 1) {
      const piece = event.results[i][0].transcript;
      if (event.results[i].isFinal) finalText += piece;
      else interim += piece;
    }
    voiceTranscript.textContent = interim || finalText;
    if (finalText.trim()) {
      sendChatMessage(finalText.trim());
      voiceTranscript.textContent = `Sent: ${finalText.trim()}`;
    }
  };
  recognition.onend = () => {
    if (active) recognition.start();
  };
  voiceBtn.addEventListener('click', () => {
    active = !active;
    voiceBtn.textContent = active ? 'Stop mic notes' : 'Start mic notes';
    if (active) recognition.start();
    else recognition.stop();
  });
}

window.addEventListener('beforeinstallprompt', (event) => {
  event.preventDefault();
  deferredPrompt = event;
  installBtn.classList.remove('hidden');
});
installBtn.addEventListener('click', async () => {
  if (!deferredPrompt) return;
  deferredPrompt.prompt();
  await deferredPrompt.userChoice;
  deferredPrompt = null;
  installBtn.classList.add('hidden');
});

window.addEventListener('error', (event) => {
  console.error('window error', event.error || event.message);
  logFeed(`JS error: ${event.message || event.error}`);
});

window.addEventListener('unhandledrejection', (event) => {
  console.error('unhandled rejection', event.reason);
  logFeed(`Promise error: ${event.reason?.message || event.reason}`);
});

if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/static/service-worker.js').catch(() => {});
}

loadConfig();
setupSpeechRecognition();
renderBestFrames();
