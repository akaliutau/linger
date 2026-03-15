import { useEffect, useMemo, useRef, useState } from 'react'

const APP_TITLE = 'Linger Story Camera'
const HERO_TIMEOUT_MS = 25000
const FRAME_TIMEOUT_MS = 18000
const FINALIZE_TIMEOUT_MS = 45000
const FRAME_INTERVAL_MS = 1000
const AUTO_STOP_AFTER_FRAMES = 12
const AUTO_STOP_TARGET_KEPT = 3
const MAX_CONSECUTIVE_ERRORS = 3

const initialDebug = {
  phase: 'intro',
  sessionId: '',
  lastError: '',
  lastLatencyMs: 0,
  frameLatencyAvgMs: 0,
  lastRawGuidance: '',
  lastNetwork: '',
  cameraState: '',
  videoReadyState: 0,
  videoDims: '',
  streamActive: false,
  trackInfo: null,
}

function makeSessionId() {
  return `linger-${Math.random().toString(36).slice(2, 8)}`
}

function nowIso() {
  return new Date().toISOString()
}

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

function shortError(err) {
  if (!err) return 'Unknown error'
  if (typeof err === 'string') return err
  return err.message || String(err)
}

async function fetchJson(url, options = {}, timeoutMs = 15000) {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  try {
    const response = await fetch(url, { ...options, signal: controller.signal })
    const text = await response.text()
    let data = null
    try {
      data = text ? JSON.parse(text) : null
    } catch {
      data = { raw: text }
    }
    if (!response.ok) {
      throw new Error(data?.detail || data?.error || `Request failed (${response.status})`)
    }
    return data
  } finally {
    clearTimeout(timer)
  }
}

function scoreTone(score) {
  if (score >= 88) return 'excellent'
  if (score >= 74) return 'good'
  if (score >= 55) return 'mid'
  return 'low'
}

function normalizeRole(role) {
  const raw = String(role || '').trim().toLowerCase().replace(/[^a-z0-9]+/g, '_')
  const mapping = {
    hero: 'hero_object',
    hero_shot: 'hero_object',
    object: 'hero_object',
    detail: 'detail_texture',
    texture: 'detail_texture',
    context: 'reveal_context',
    wide: 'reveal_context',
    reveal: 'reveal_context',
    problem: 'problem_area',
    opportunity: 'room_opportunity',
    room: 'room_opportunity',
    fit: 'fit_check',
    fitcheck: 'fit_check',
  }
  return mapping[raw] || raw || 'reveal_context'
}

function prettyRole(role) {
  return normalizeRole(role).replace(/_/g, ' ')
}

function frameImageUrl(frame, heroUrl = '') {
  const base = String(
    frame?.preview_url ||
    frame?.local_preview_url ||
    frame?.image_url ||
    ''
  ).trim()

  if (!base) return heroUrl || ''

  const stamp =
    frame?.ts_ms ||
    frame?.frame_index ||
    frame?.score_0_to_100 ||
    'review'

  return `${base}${base.includes('?') ? '&' : '?'}v=${encodeURIComponent(String(stamp))}`
}

function publicGuideFor(score, selected, keptTotal, role = '') {
  const normalized = normalizeRole(role)

  if (selected && normalized === 'room_opportunity') return 'Good corner. Take one tighter proof shot.'
  if (selected && normalized === 'fit_check') return 'Nice fit. Take one wider safety shot.'
  if (selected && normalized === 'hero_object') return `Strong hero. Add ${keptTotal < 2 ? 'two' : 'one'} support shots.`
  if (selected && normalized === 'detail_texture') return 'Good detail. Now get one wider context shot.'
  if (selected && normalized === 'problem_area') return 'Good problem signal. Show the surrounding area.'

  if (score >= 78) return 'Nice. Find the room need this could solve.'
  if (score >= 60) return 'Close. Hold steadier and show more room context.'
  if (score >= 40) return 'Try a cleaner corner or brighter angle.'
  return 'Move slower. Look for the empty spot or problem area.'
}

function shouldSpeakSpike(previousBest, score, selected, nowMs, lastSpeakAt, role = '') {
  if (!selected) return false
  if (nowMs - lastSpeakAt < 9000) return false

  const normalized = normalizeRole(role)
  if (score >= 78 && ['room_opportunity', 'fit_check', 'hero_object'].includes(normalized)) {
    return true
  }

  if (score < 82) return false
  if (score - previousBest < 8 && previousBest !== 0) return false
  return true
}

function voiceLineFromFrameResult(data, score, selected, keptTotal, role) {
  const clean = String(
    data?.guidance ||
    data?.analysis?.micro_direction ||
    publicGuideFor(score, selected, keptTotal, role)
  ).trim()
  return clean || 'Take one cleaner support shot.'
}

function cameraAvailable() {
  return Boolean(window.isSecureContext && navigator.mediaDevices && navigator.mediaDevices.getUserMedia)
}

function videoReady(video) {
  return Boolean(video && video.srcObject && video.readyState >= 2 && video.videoWidth > 0 && video.videoHeight > 0)
}

function summarizeTrack(track) {
  if (!track) return null
  let settings = {}
  let constraints = {}
  try { settings = track.getSettings?.() || {} } catch {}
  try { constraints = track.getConstraints?.() || {} } catch {}
  return {
    kind: track.kind,
    label: track.label,
    enabled: track.enabled,
    muted: track.muted,
    readyState: track.readyState,
    settings,
    constraints,
  }
}

function iconState(score) {
  if (score >= 84) return 'on'
  if (score >= 62) return 'mid'
  return 'off'
}

function IntroScreen({ onStart, config, loadingConfig }) {
  return (
    <div className="screen intro-screen">
      <div className="intro-glow" />
      <div className="intro-card">
        <div className="eyebrow">Demo PoC</div>
        <h1>{APP_TITLE}</h1>
        <p>
          One tap to enter the camera. Capture a hero image, harvest strong moments, then generate a story.
        </p>
        <button className="primary-button xl" onClick={onStart}>
          Start the story
        </button>
        <div className="intro-meta">
          {loadingConfig ? 'Reading backend…' : `Storage ${config.storage_mode || '—'} · Stage1 ${config.stage1_model || '—'}`}
        </div>
      </div>
    </div>
  )
}

function TopHud({ score, kept, framesSeen, guide, latencyAvg, mode }) {
  return (
    <div className="hud top">
      <div className={`score-pill ${scoreTone(score)}`}>
        <span className="score-label">score</span>
        <strong>{score || '—'}</strong>
      </div>
      <div className="hud-cluster">
        <MiniStat label="kept" value={`${kept}`} />
        <MiniStat label="frames" value={`${framesSeen}`} />
        <MiniStat label="avg ms" value={latencyAvg ? `${latencyAvg}` : '—'} />
      </div>
      <div className="hud-icons">
        <HudIcon label="sharp" state={iconState(score)} />
        <HudIcon label="story" state={kept > 0 ? 'on' : iconState(score)} />
        <HudIcon label="pace" state={mode === 'harvesting' ? 'on' : 'mid'} />
      </div>
      <div className="guide-pill">{guide}</div>
    </div>
  )
}

function MiniStat({ label, value }) {
  return (
    <div className="mini-stat">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}

function HudIcon({ label, state }) {
  return (
    <div className={`hud-icon ${state}`}>
      <span className="dot" />
      <span>{label}</span>
    </div>
  )
}

function CameraScreen({
  videoRef,
  phaseLabel,
  primaryLabel,
  onPrimary,
  onSecondary,
  secondaryLabel,
  busy,
  score,
  kept,
  framesSeen,
  latencyAvg,
  guide,
  heroThumb,
  status,
  mode,
}) {
  return (
    <div className="screen camera-screen">
      <video ref={videoRef} className="camera-video" playsInline muted autoPlay />
      <div className="camera-overlay top-gradient" />
      <div className="camera-overlay bottom-gradient" />

      <TopHud score={score} kept={kept} framesSeen={framesSeen} latencyAvg={latencyAvg} guide={guide} mode={mode} />

      <div className="hero-thumb-wrap">
        {heroThumb ? <img src={heroThumb} alt="Hero still" className="hero-thumb" /> : <div className="hero-thumb placeholder" />}
      </div>

      <div className="phase-chip">{phaseLabel}</div>
      <div className="status-strip">{status}</div>

      <div className="camera-controls">
        {secondaryLabel ? (
          <button className="ghost-button" onClick={onSecondary} disabled={busy}>
            {secondaryLabel}
          </button>
        ) : (
          <div className="ghost-spacer" />
        )}
        <button className={`capture-button ${busy ? 'busy' : ''}`} onClick={onPrimary} disabled={busy}>
          <span className="capture-ring" />
          <span className="capture-text">{busy ? 'Working…' : primaryLabel}</span>
        </button>
      </div>
    </div>
  )
}

function ReviewScreen({ heroUrl, analysis, generating, onGenerate, onReset, finalizeInfo, bestFrames }) {
  const hook = finalizeInfo?.story_seed?.hook || analysis?.story_signal || 'A grounded second-life idea built from the best room evidence.'
  const strategy = finalizeInfo?.story_seed?.generation_strategy || ''
  const shownFrames = (
    finalizeInfo?.shot_manifest ||
    finalizeInfo?.story_seed?.selected_frame_items ||
    bestFrames ||
    []
  ).slice(0, 3)

  return (
    <div className="screen review-screen">
      <div className="review-shell">
        <div className="review-card">
          <div className="review-header">
            <img src={heroUrl} alt="Hero" className="review-hero" />
            <div>
              <div className="eyebrow">Collection ready</div>
              <h2>{analysis?.object_label || 'Story seed'}</h2>
              <p className="muted">{analysis?.one_line_summary || 'A compact visual premise for the next reel.'}</p>
            </div>
          </div>

          <div className="idea-box">
            <div className="idea-label">Idea</div>
            <p>{hook}</p>
            {strategy ? <p className="muted">{strategy}</p> : null}
          </div>

          {!!shownFrames.length && (
            <div className="best-strip">
              {shownFrames.map((frame) => (
                <div className="best-pill" key={frame.local_path || frame.file_name || frame.frame_index || frame.preview_url || frame.local_preview_url}>
                  <img src={frameImageUrl(frame, heroUrl)} alt={frame.moment_label || 'Best frame'} loading="eager" />
                  <div>
                    <strong>{frame.moment_label || prettyRole(frame.cinematic_role)}</strong>
                    <span>{prettyRole(frame.cinematic_role)} · {frame.score_0_to_100}/100</span>
                    {frame.best_future_use ? <small>{frame.best_future_use}</small> : null}
                  </div>
                </div>
              ))}
            </div>
          )}

          {finalizeInfo?.pipeline_result?.status && (
            <div className="pipeline-box">
              <strong>Story pipeline</strong>
              <span>{finalizeInfo.pipeline_result.status}</span>
            </div>
          )}

          <div className="review-actions">
            <button className="primary-button" onClick={onGenerate} disabled={generating}>
              {generating ? 'Generating…' : 'Generate story'}
            </button>
            <button className="secondary-button" onClick={onReset} disabled={generating}>
              Start again
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

function DebugDrawer({ open, onToggle, debug, events, config, stateSummary }) {
  return (
    <>
      <button className="debug-toggle" onClick={onToggle}>
        {open ? 'Hide debug' : 'Debug'}
      </button>
      {open && (
        <aside className="debug-drawer">
          <section>
            <h3>State</h3>
            <pre>{JSON.stringify({ ...debug, ...stateSummary }, null, 2)}</pre>
          </section>
          <section>
            <h3>Config</h3>
            <pre>{JSON.stringify(config, null, 2)}</pre>
          </section>
          <section>
            <h3>Events</h3>
            <pre>{JSON.stringify(events.slice(0, 18), null, 2)}</pre>
          </section>
        </aside>
      )}
    </>
  )
}

export default function App() {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const streamRef = useRef(null)
  const harvestTimerRef = useRef(null)
  const requestInFlightRef = useRef(false)
  const audioRef = useRef(null)
  const videoProbeRef = useRef(null)
  const lastSpeakAtRef = useRef(0)
  const voiceModeRef = useRef('unknown')
  const mountedRef = useRef(true)
  const modeRef = useRef('hero')
  const scoreRef = useRef(0)
  const framesSeenRef = useRef(0)
  const keptCountRef = useRef(0)
  const consecutiveErrorsRef = useRef(0)

  const [config, setConfig] = useState({})
  const [loadingConfig, setLoadingConfig] = useState(true)
  const [phase, setPhase] = useState('intro')
  const [mode, setMode] = useState('hero')
  const [sessionId, setSessionId] = useState(makeSessionId())
  const [status, setStatus] = useState('Ready.')
  const [guide, setGuide] = useState('Tap Capture when the frame feels clean.')
  const [cameraBusy, setCameraBusy] = useState(false)
  const [score, setScore] = useState(0)
  const [heroUrl, setHeroUrl] = useState('')
  const [analysis, setAnalysis] = useState(null)
  const [bestFrames, setBestFrames] = useState([])
  const [framesSeen, setFramesSeen] = useState(0)
  const [keptCount, setKeptCount] = useState(0)
  const [latencyAvg, setLatencyAvg] = useState(0)
  const [consecutiveErrors, setConsecutiveErrors] = useState(0)
  const [generating, setGenerating] = useState(false)
  const [finalizeInfo, setFinalizeInfo] = useState(null)
  const [debugOpen, setDebugOpen] = useState(false)
  const [debug, setDebug] = useState(initialDebug)
  const [events, setEvents] = useState([])
  const [cameraBootPending, setCameraBootPending] = useState(false)
  const [streamNonce, setStreamNonce] = useState(0)

  const stateSummary = useMemo(
    () => ({ phase, mode, score, framesSeen, keptCount, consecutiveErrors, sessionId }),
    [phase, mode, score, framesSeen, keptCount, consecutiveErrors, sessionId],
  )

  function pushEvent(type, payload = {}) {
    const item = { ts: nowIso(), type, ...payload }
    console.log('[linger]', item)
    setEvents((prev) => [item, ...prev].slice(0, 50))
  }

  function updateDebug(patch) {
    setDebug((prev) => ({ ...prev, ...patch }))
  }

  function stopVideoProbe() {
    if (videoProbeRef.current) {
      clearInterval(videoProbeRef.current)
      videoProbeRef.current = null
    }
  }

  function attachVideoDiagnostics(video) {
    if (!video) return () => {}
    const names = ['loadedmetadata', 'loadeddata', 'canplay', 'playing', 'pause', 'waiting', 'stalled', 'suspend', 'emptied', 'resize', 'error']
    const handlers = names.map((name) => {
      const handler = () => {
        const payload = {
          event: name,
          readyState: video.readyState,
          width: video.videoWidth || 0,
          height: video.videoHeight || 0,
          currentTime: Number(video.currentTime || 0).toFixed(3),
          paused: video.paused,
          ended: video.ended,
        }
        updateDebug({
          cameraState: name,
          videoReadyState: video.readyState,
          videoDims: `${video.videoWidth || 0}x${video.videoHeight || 0}`,
        })
        pushEvent('video_event', payload)
      }
      video.addEventListener(name, handler)
      return [name, handler]
    })
    return () => {
      handlers.forEach(([name, handler]) => video.removeEventListener(name, handler))
    }
  }

  async function waitForVideoReady(video, timeoutMs = 6000) {
    const started = performance.now()
    while (performance.now() - started < timeoutMs) {
      if (videoReady(video)) return true
      await wait(120)
    }
    return videoReady(video)
  }

  async function attachStreamToVideo(stream, reason = 'attach') {
    const video = videoRef.current
    if (!video) {
      pushEvent('video_missing', { reason })
      updateDebug({ cameraState: 'video_missing', streamActive: !!stream })
      return false
    }

    video.muted = true
    video.autoplay = true
    video.playsInline = true
    video.setAttribute('muted', '')
    video.setAttribute('autoplay', '')
    video.setAttribute('playsinline', 'true')
    video.setAttribute('webkit-playsinline', 'true')
    video.srcObject = stream

    try {
      await video.play()
      pushEvent('video_play_called', { reason })
    } catch (err) {
      pushEvent('video_play_failed', { reason, error: shortError(err) })
    }

    const ok = await waitForVideoReady(video, 7000)
    const track = stream?.getVideoTracks?.()[0]
    const trackInfo = summarizeTrack(track)
    updateDebug({
      cameraState: ok ? 'video_ready' : 'video_not_ready',
      videoReadyState: video.readyState || 0,
      videoDims: `${video.videoWidth || 0}x${video.videoHeight || 0}`,
      streamActive: Boolean(stream),
      trackInfo,
    })
    pushEvent('video_attach_result', {
      reason,
      ok,
      readyState: video.readyState || 0,
      width: video.videoWidth || 0,
      height: video.videoHeight || 0,
      trackInfo,
    })
    return ok
  }

  function startVideoProbe() {
    stopVideoProbe()
    videoProbeRef.current = setInterval(() => {
      const video = videoRef.current
      const stream = streamRef.current
      const track = stream?.getVideoTracks?.()[0] || null
      const payload = {
        phase,
        mode: modeRef.current,
        hasVideo: Boolean(video),
        hasStream: Boolean(stream),
        readyState: video?.readyState || 0,
        width: video?.videoWidth || 0,
        height: video?.videoHeight || 0,
        currentTime: Number(video?.currentTime || 0).toFixed(3),
        paused: Boolean(video?.paused),
        hidden: document.hidden,
        trackReadyState: track?.readyState || 'none',
        trackMuted: track?.muted ?? null,
        trackEnabled: track?.enabled ?? null,
      }
      updateDebug({
        cameraState: 'probe',
        videoReadyState: payload.readyState,
        videoDims: `${payload.width}x${payload.height}`,
        streamActive: payload.hasStream,
        trackInfo: summarizeTrack(track),
      })
      pushEvent('video_probe', payload)
    }, 2500)
  }

  async function loadConfig() {
    setLoadingConfig(true)
    try {
      const data = await fetchJson('/api/config', {}, 10000)
      if (!mountedRef.current) return
      setConfig(data || {})
      pushEvent('config_ready', { storage: data?.storage_mode, stage1: data?.stage1_model })
    } catch (err) {
      pushEvent('config_failed', { error: shortError(err) })
      updateDebug({ lastError: shortError(err) })
    } finally {
      if (mountedRef.current) setLoadingConfig(false)
    }
  }

  useEffect(() => {
    mountedRef.current = true
    pushEvent('app_effect_mounted', { strictModeSafe: true })
    loadConfig()
    return () => {
      pushEvent('app_effect_cleanup', { strictModeSafe: true })
      mountedRef.current = false
      stopCamera()
      stopHarvestLoop()
      if (heroUrl) URL.revokeObjectURL(heroUrl)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  async function startStory() {
    if (!cameraAvailable()) {
      const message = !window.isSecureContext
        ? 'Camera needs HTTPS on mobile. Open this app over HTTPS or localhost.'
        : 'Camera API is not available in this browser.'
      setStatus(message)
      setGuide(message)
      updateDebug({ lastError: message, phase: 'intro' })
      pushEvent('camera_unavailable', { message, secure: window.isSecureContext })
      return
    }

    setCameraBusy(true)
    setPhase('camera')
    setMode('hero')
    setStatus('Opening camera…')
    setGuide('Allow camera access if the browser asks.')
    updateDebug({ phase: 'camera', sessionId, cameraState: 'camera_screen_requested' })
    pushEvent('camera_open_requested', { secure: window.isSecureContext, visibility: document.visibilityState })
    pushEvent('camera_screen_requested', { sessionId })
    setCameraBootPending(true)
  }

  async function bootCameraForMountedScreen() {
    pushEvent('camera_boot_effect_start', { phase, sessionId, hasVideoRef: Boolean(videoRef.current) })
    await wait(60)
    try {
      let devices = []
      try {
        devices = await navigator.mediaDevices.enumerateDevices()
        pushEvent('media_devices', { devices: devices.map((d) => ({ kind: d.kind, label: d.label, deviceId: d.deviceId ? 'set' : '' })) })
      } catch (err) {
        pushEvent('media_devices_failed', { error: shortError(err) })
      }

      pushEvent('camera_boot_before_gum', { hasVideoRef: Boolean(videoRef.current) })
      let stream = null
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: { ideal: 'environment' },
            width: { ideal: 720 },
            height: { ideal: 1280 },
          },
          audio: false,
        })
        pushEvent('camera_stream_opened', { strategy: 'preferred', trackInfo: summarizeTrack(stream.getVideoTracks?.()[0]) })
      } catch (firstErr) {
        pushEvent('camera_stream_retry', { error: shortError(firstErr) })
        stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false })
        pushEvent('camera_stream_opened', { strategy: 'fallback', trackInfo: summarizeTrack(stream.getVideoTracks?.()[0]) })
      }

      if (!mountedRef.current) {
        pushEvent('camera_boot_unmounted')
        stream?.getTracks?.().forEach((track) => track.stop())
        return
      }

      streamRef.current = stream
      const trackInfo = summarizeTrack(stream.getVideoTracks?.()[0])
      updateDebug({
        phase: 'camera',
        sessionId,
        streamActive: true,
        trackInfo,
        cameraState: 'stream_opened',
      })
      setStatus('Opening preview…')
      setGuide('Look for one clean, balanced shot.')
      startVideoProbe()
      pushEvent('camera_boot_after_gum', { hasVideoRef: Boolean(videoRef.current), trackInfo })
      setStreamNonce((n) => n + 1)

      const attached = await attachStreamToVideo(stream, 'boot_camera_after_gum')
      if (attached) {
        setStatus('Frame one clean hero still.')
        setGuide('Look for one clean, balanced shot.')
      } else {
        setStatus('Preview did not start. Try re-opening the camera.')
        setGuide('The stream is live, but the video element is not receiving frames yet.')
      }
    } catch (err) {
      const message = shortError(err)
      setStatus(message)
      setGuide('Camera failed to open.')
      updateDebug({ lastError: message, cameraState: 'camera_error' })
      pushEvent('camera_error', { error: message })
    } finally {
      setCameraBusy(false)
      setCameraBootPending(false)
    }
  }

  function stopCamera() {
    stopVideoProbe()
    if (streamRef.current) {
      const tracks = streamRef.current.getTracks()
      pushEvent('camera_stopping', { tracks: tracks.map((track) => summarizeTrack(track)) })
      tracks.forEach((track) => track.stop())
      streamRef.current = null
    }
    if (videoRef.current) {
      videoRef.current.pause?.()
      videoRef.current.srcObject = null
    }
    updateDebug({ streamActive: false, cameraState: 'stopped' })
  }

  function stopHarvestLoop() {
    if (harvestTimerRef.current) {
      clearTimeout(harvestTimerRef.current)
      harvestTimerRef.current = null
    }
    requestInFlightRef.current = false
  }

  function scheduleNextFrame(delay = FRAME_INTERVAL_MS) {
    stopHarvestLoop()
    harvestTimerRef.current = setTimeout(() => {
      void processHarvestTick()
    }, delay)
  }

  async function captureBlob(maxWidth = 1600, quality = 0.9) {
    const video = videoRef.current
    if (!video) throw new Error('Camera video element is missing')

    const ok = await waitForVideoReady(video, 3500)
    if (!ok) {
      const payload = {
        readyState: video.readyState || 0,
        width: video.videoWidth || 0,
        height: video.videoHeight || 0,
        currentTime: Number(video.currentTime || 0).toFixed(3),
        paused: video.paused,
        hasStream: Boolean(video.srcObject),
      }
      pushEvent('capture_not_ready', payload)
      throw new Error(`Camera preview not ready (${payload.width}x${payload.height}, readyState ${payload.readyState})`)
    }

    const width = video.videoWidth
    const height = video.videoHeight
    const scale = Math.min(1, maxWidth / Math.max(width, height))
    const outWidth = Math.max(1, Math.round(width * scale))
    const outHeight = Math.max(1, Math.round(height * scale))
    let canvas = canvasRef.current
    if (!canvas) {
      canvas = document.createElement('canvas')
      canvasRef.current = canvas
    }
    canvas.width = outWidth
    canvas.height = outHeight
    const ctx = canvas.getContext('2d', { alpha: false, willReadFrequently: true })
    if (!ctx) throw new Error('Canvas is not available')
    ctx.drawImage(video, 0, 0, outWidth, outHeight)

    const sample = ctx.getImageData(0, 0, Math.min(outWidth, 24), Math.min(outHeight, 24)).data
    let total = 0
    for (let i = 0; i < sample.length; i += 4) total += sample[i] + sample[i + 1] + sample[i + 2]
    const avgLuma = sample.length ? Math.round(total / (sample.length / 4) / 3) : 0
    pushEvent('capture_sample', { width: outWidth, height: outHeight, avgLuma, currentTime: Number(video.currentTime || 0).toFixed(3) })
    updateDebug({ videoReadyState: video.readyState || 0, videoDims: `${width}x${height}`, cameraState: `capture_luma_${avgLuma}` })

    const blob = await new Promise((resolve, reject) => {
      canvas.toBlob((result) => {
        if (!result) reject(new Error('Image capture failed'))
        else resolve(result)
      }, 'image/jpeg', quality)
    })
    return blob
  }

  async function handlePrimary() {
    if (cameraBusy) return
    if (mode === 'hero') {
      await runStage1Capture()
      return
    }
    if (mode === 'ready') {
      await startHarvest()
      return
    }
    if (mode === 'harvesting') {
      enterReview('Manual finish')
    }
  }

  async function runStage1Capture() {
    try {
      setCameraBusy(true)
      setStatus('Reading the first shot…')
      const blob = await captureBlob(1600, 0.92)
      const previewUrl = URL.createObjectURL(blob)
      if (heroUrl) URL.revokeObjectURL(heroUrl)
      setHeroUrl(previewUrl)

      const form = new FormData()
      form.append('session_id', sessionId)
      form.append('note', '')
      form.append('file', new File([blob], `hero_${Date.now()}.jpg`, { type: 'image/jpeg' }))
      const started = performance.now()
      const data = await fetchJson('/api/stage1/analyze-photo', { method: 'POST', body: form }, HERO_TIMEOUT_MS)
      const latencyMs = Math.round(performance.now() - started)

      setAnalysis(data.analysis)
      setMode('ready')
      setStatus(data.analysis?.one_line_summary || 'Ready for context.')
      setGuide(data.analysis?.opening_coach_line || 'Tap Record context. Look for where this could belong.')
      updateDebug({ phase: 'camera', lastLatencyMs: latencyMs, sessionId, lastNetwork: 'stage1' })
      pushEvent('stage1_ready', { latencyMs, label: data.analysis?.object_label })
    } catch (err) {
      const message = shortError(err)
      setStatus(message)
      setGuide('Try Capture again when the frame is steady.')
      updateDebug({ lastError: message, lastNetwork: 'stage1' })
      pushEvent('stage1_failed', { error: message })
    } finally {
      setCameraBusy(false)
    }
  }

  async function startHarvest() {
    try {
      setCameraBusy(true)
      setMode('harvesting')
      setStatus('Scanning the room…')
      const form = new FormData()
      form.append('session_id', sessionId)
      const data = await fetchJson('/api/live/start', { method: 'POST', body: form }, 15000)
      const startGuide = (data.guidance || analysis?.opening_coach_line || publicGuideFor(score, false, keptCount, 'room_opportunity')).trim()
      setGuide(startGuide)
      updateDebug({ lastRawGuidance: data.guidance || '', lastNetwork: 'live_start' })
      pushEvent('harvest_started', { rawGuidance: data.guidance || '' })
      scheduleNextFrame(250)
    } catch (err) {
      const message = shortError(err)
      setMode('ready')
      setStatus(message)
      updateDebug({ lastError: message, lastNetwork: 'live_start' })
      pushEvent('harvest_start_failed', { error: message })
    } finally {
      setCameraBusy(false)
    }
  }

  async function processHarvestTick() {
    if (requestInFlightRef.current || modeRef.current !== 'harvesting') return
    requestInFlightRef.current = true
    let nextDelay = FRAME_INTERVAL_MS

    try {
      const blob = await captureBlob(1280, 0.82)
      const nextFrameIndex = framesSeenRef.current + 1
      const form = new FormData()
      form.append('session_id', sessionId)
      form.append('note', '')
      form.append('frame_index', String(nextFrameIndex))
      form.append('ts_ms', String(Date.now()))
      form.append('file', new File([blob], `frame_${nextFrameIndex}.jpg`, { type: 'image/jpeg' }))

      const started = performance.now()
      const data = await fetchJson('/api/live/frame', { method: 'POST', body: form }, FRAME_TIMEOUT_MS)
      const latencyMs = Math.round(performance.now() - started)
      nextDelay = Math.max(150, FRAME_INTERVAL_MS - latencyMs)

      const nextFramesSeen = nextFrameIndex
      const nextScore = Number(data.score || data.analysis?.score_0_to_100 || 0)
      const nextKept = Number(data.kept_total || 0)
      const effectiveKept = Number(data.effective_kept_total || nextKept)
      const selected = Boolean(data.selected)
      const fallbackSelected = Boolean(data.fallback_selected)
      const nextRole = normalizeRole(data.candidate?.cinematic_role || data.analysis?.cinematic_role)
      const nextGuide = String(data.guidance || publicGuideFor(nextScore, selected || fallbackSelected, effectiveKept, nextRole)).trim()
      const previousBest = scoreRef.current

      framesSeenRef.current = nextFramesSeen
      scoreRef.current = nextScore
      keptCountRef.current = effectiveKept
      consecutiveErrorsRef.current = 0
      setFramesSeen(nextFramesSeen)
      setScore(nextScore)
      setKeptCount(effectiveKept)
      setLatencyAvg(Number(data.avg_latency_ms || 0))
      setGuide(nextGuide)
      setStatus(selected ? 'Strong moment saved.' : 'Scanning for the next strong frame.')
      setConsecutiveErrors(0)

      updateDebug({
        lastLatencyMs: latencyMs,
        frameLatencyAvgMs: Number(data.avg_latency_ms || 0),
        lastRawGuidance: data.guidance || '',
        lastNetwork: 'live_frame',
      })
      pushEvent('frame', {
        frameIndex: nextFrameIndex,
        score: nextScore,
        selected,
        role: nextRole,
        latencyMs,
        rawGuidance: data.guidance || '',
      })

      if ((selected || fallbackSelected) && data.candidate) {
        setBestFrames((prev) => {
          const next = [data.candidate, ...prev.filter((item) => item.local_path !== data.candidate.local_path)]
          return next.slice(0, 8)
        })
      }

      if (shouldSpeakSpike(previousBest, nextScore, selected || fallbackSelected, Date.now(), lastSpeakAtRef.current, nextRole)) {
        const line = voiceLineFromFrameResult(data, nextScore, selected || fallbackSelected, effectiveKept, nextRole)
        lastSpeakAtRef.current = Date.now()
        void speakGuide(line)
        pushEvent('voice_spike', { score: nextScore, role: nextRole, line })
      }

      const shouldAutoStop =
        Boolean(data.should_stop) ||
        ((effectiveKept >= AUTO_STOP_TARGET_KEPT && nextFramesSeen >= 5) || nextFramesSeen >= AUTO_STOP_AFTER_FRAMES)

      if (shouldAutoStop) {
        const usedFallbackOnly = effectiveKept > 0 && nextKept === 0
        enterReview(
          data.stop_reason ||
          (effectiveKept >= AUTO_STOP_TARGET_KEPT
            ? 'Collection ready.'
            : usedFallbackOnly
              ? 'Best attempt saved. Review the top frame.'
              : 'No usable frame yet. Try another pass.')
        )
        return
      }
    } catch (err) {
      const message = shortError(err)
      const nextErrors = consecutiveErrorsRef.current + 1
      consecutiveErrorsRef.current = nextErrors
      setConsecutiveErrors(nextErrors)
      setGuide('Holding position. The model is taking longer than usual.')
      setStatus(message)
      updateDebug({ lastError: message, lastNetwork: 'live_frame' })
      pushEvent('frame_failed', { error: message, attempt: nextErrors })
      nextDelay = 1200

      if (nextErrors >= MAX_CONSECUTIVE_ERRORS) {
        enterReview('Stopped safely after repeated model issues.')
        return
      }
    } finally {
      requestInFlightRef.current = false
      if (modeRef.current === 'harvesting') {
        scheduleNextFrame(nextDelay)
      }
    }
  }

  async function speakGuide(text) {
    const clean = text?.trim()
    if (!clean) return

    if (voiceModeRef.current !== 'browser') {
      try {
        const data = await fetchJson(
          '/api/tts/guide',
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: clean }),
          },
          8000,
        )
        if (data?.ok && data?.url) {
          voiceModeRef.current = 'server'
          if (audioRef.current) {
            audioRef.current.pause()
          }
          const audio = new Audio(data.url)
          audioRef.current = audio
          await audio.play().catch(() => {})
          return
        }
      } catch {
        voiceModeRef.current = 'browser'
      }
    }

    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel()
      const utterance = new SpeechSynthesisUtterance(clean)
      utterance.rate = 0.96
      utterance.pitch = 0.95
      const voices = window.speechSynthesis.getVoices() || []
      const preferred = voices.find((voice) => /Google US English|en-US|Samantha/i.test(voice.name))
      if (preferred) utterance.voice = preferred
      window.speechSynthesis.speak(utterance)
    }
  }

  function enterReview(message) {
    stopHarvestLoop()
    stopCamera()
    setPhase('review')
    setMode('review')
    setStatus(message)
    setGuide('Collection ready. Review the three best story inputs.')
    updateDebug({ phase: 'review' })
    pushEvent('enter_review', { message })
    void syncReviewFramesFromSession()
  }

  async function syncReviewFramesFromSession() {
    try {
      const state = await fetchJson(`/api/session/${encodeURIComponent(sessionId)}`, {}, 10000)
      const frames = (
        (Array.isArray(state.best_frames) && state.best_frames.length
          ? state.best_frames
          : state.fallback_top_frame
            ? [state.fallback_top_frame]
            : []
        )
      ).slice(0, 3)

      setBestFrames(frames)
      pushEvent('review_frames_synced', { count: frames.length })
    } catch (err) {
      const message = shortError(err)
      updateDebug({ lastError: message, lastNetwork: 'session_review_sync' })
      pushEvent('review_frames_sync_failed', { error: message })
    }
  }

  async function generateStory() {
    try {
      setGenerating(true)
      setStatus('Preparing story basket…')
      const form = new FormData()
      form.append('session_id', sessionId)
      form.append('enough_moment', 'Collection ready')

      const finalized = await fetchJson('/api/session/finalize', { method: 'POST', body: form }, FINALIZE_TIMEOUT_MS)
      const state = await fetchJson(`/api/session/${encodeURIComponent(sessionId)}`, {}, 15000)

      const mergedFinalize = {
        ...(finalized.finalized || {}),
        story_seed: state.story_seed || finalized.finalized?.story_seed || null,
      }

      setFinalizeInfo(mergedFinalize)
      const displayFrames =
        mergedFinalize?.shot_manifest ||
        mergedFinalize?.story_seed?.selected_frame_items ||
        (state.best_frames || []).slice(0, 3) ||
        (state.fallback_top_frame ? [state.fallback_top_frame] : [])
      setBestFrames(Array.isArray(displayFrames) ? displayFrames.slice(0, 3) : [])
      pushEvent('finalized', {
        basketAssets: finalized.finalized?.basket_assets?.length || 0,
        selectedCount: finalized.finalized?.selected_count || 0,
        usedFallbackFrame: Boolean(finalized.finalized?.used_fallback_frame),
      })

      setStatus(
        finalized.finalized?.pipeline_result?.status === 'finished'
          ? 'Story started.'
          : 'Bundle ready for video pipeline.'
      )
    } catch (err) {
      const message = shortError(err)
      setStatus(message)
      updateDebug({ lastError: message, lastNetwork: 'finalize' })
      pushEvent('finalize_failed', { error: message })
    } finally {
      setGenerating(false)
    }
  }

  function resetAll() {
    stopHarvestLoop()
    stopCamera()
    if (heroUrl) URL.revokeObjectURL(heroUrl)
    setSessionId(makeSessionId())
    setPhase('intro')
    setMode('hero')
    setStatus('Ready.')
    setGuide('Tap Capture when the frame feels clean.')
    scoreRef.current = 0
    framesSeenRef.current = 0
    keptCountRef.current = 0
    consecutiveErrorsRef.current = 0
    setScore(0)
    setFramesSeen(0)
    setKeptCount(0)
    setLatencyAvg(0)
    setHeroUrl('')
    setAnalysis(null)
    setBestFrames([])
    setFinalizeInfo(null)
    setGenerating(false)
    setConsecutiveErrors(0)
    setDebug(initialDebug)
    setEvents([])
    lastSpeakAtRef.current = 0
    requestInFlightRef.current = false
  }

  const phaseLabel = useMemo(() => {
    if (mode === 'hero') return 'Hero capture'
    if (mode === 'ready') return 'Ready for context'
    if (mode === 'harvesting') return 'Recording context'
    return 'Review'
  }, [mode])

  const primaryLabel = useMemo(() => {
    if (mode === 'hero') return 'Capture'
    if (mode === 'ready') return 'Record context'
    if (mode === 'harvesting') return 'Collection ready'
    return 'Capture'
  }, [mode])

  const secondaryLabel = useMemo(() => {
    if (mode === 'harvesting') return 'Stop'
    return ''
  }, [mode])


  useEffect(() => {
    return () => {
      if (heroUrl) URL.revokeObjectURL(heroUrl)
    }
  }, [heroUrl])

  useEffect(() => {
    modeRef.current = mode
  }, [mode])

  useEffect(() => {
    if (phase !== 'camera' || !cameraBootPending) return
    let cancelled = false
    void (async () => {
      pushEvent('camera_boot_effect_tick', { hasVideoRef: Boolean(videoRef.current), phase })
      await wait(0)
      if (cancelled) return
      await bootCameraForMountedScreen()
    })()
    return () => {
      cancelled = true
    }
  }, [cameraBootPending, phase, sessionId])

  useEffect(() => {
    scoreRef.current = score
  }, [score])

  useEffect(() => {
    if (phase !== 'review') return
    void syncReviewFramesFromSession()
  }, [phase, sessionId])

  useEffect(() => {
    if (phase !== 'camera' || !streamRef.current || !videoRef.current) return
    pushEvent('camera_attach_effect_tick', { phase, hasStream: !!streamRef.current, hasVideoRef: !!videoRef.current, streamNonce })
    let cancelled = false
    const cleanupDiagnostics = attachVideoDiagnostics(videoRef.current)
    void (async () => {
      const ok = await attachStreamToVideo(streamRef.current, 'phase_camera_effect')
      if (cancelled) return
      if (ok) {
        setStatus((prev) => (prev === 'Opening preview…' ? 'Frame one clean hero still.' : prev))
        setGuide((prev) => prev || 'Look for one clean, balanced shot.')
      } else {
        setStatus('Preview did not start. Try Start again or another browser camera.')
        setGuide('The stream opened, but the preview never produced frames.')
      }
    })()
    return () => {
      cancelled = true
      cleanupDiagnostics()
    }
  }, [phase, sessionId, streamNonce])

  useEffect(() => {
    framesSeenRef.current = framesSeen
  }, [framesSeen])

  useEffect(() => {
    keptCountRef.current = keptCount
  }, [keptCount])

  useEffect(() => {
    consecutiveErrorsRef.current = consecutiveErrors
  }, [consecutiveErrors])

  useEffect(() => {
    updateDebug({ phase, sessionId })
  }, [phase, sessionId, streamNonce])

  useEffect(() => {
    const onError = (event) => {
      updateDebug({ lastError: event.message || 'Unexpected JS error' })
      pushEvent('window_error', { message: event.message || 'Unexpected JS error' })
    }
    const onRejection = (event) => {
      updateDebug({ lastError: shortError(event.reason) })
      pushEvent('promise_rejection', { message: shortError(event.reason) })
    }
    window.addEventListener('error', onError)
    window.addEventListener('unhandledrejection', onRejection)
    return () => {
      window.removeEventListener('error', onError)
      window.removeEventListener('unhandledrejection', onRejection)
    }
  }, [])

  return (
    <div className="app-shell">
      {phase === 'intro' && <IntroScreen onStart={startStory} config={config} loadingConfig={loadingConfig} />}

      {phase === 'camera' && (
        <CameraScreen
          videoRef={videoRef}
          phaseLabel={phaseLabel}
          mode={mode}
          primaryLabel={primaryLabel}
          secondaryLabel={secondaryLabel}
          onPrimary={() => void handlePrimary()}
          onSecondary={() => enterReview('Stopped manually.')}
          busy={cameraBusy}
          score={score}
          kept={keptCount}
          framesSeen={framesSeen}
          latencyAvg={latencyAvg}
          guide={guide}
          heroThumb={heroUrl}
          status={status}
        />
      )}

      {phase === 'review' && (
        <ReviewScreen
          heroUrl={heroUrl}
          analysis={analysis}
          generating={generating}
          onGenerate={() => void generateStory()}
          onReset={resetAll}
          finalizeInfo={finalizeInfo}
          bestFrames={bestFrames}
        />
      )}

      <DebugDrawer
        open={debugOpen}
        onToggle={() => setDebugOpen((prev) => !prev)}
        debug={debug}
        events={events}
        config={config}
        stateSummary={stateSummary}
      />
    </div>
  )
}
