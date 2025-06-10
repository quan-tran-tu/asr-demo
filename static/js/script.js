 
// ---------------------------
//  Global state & settings
// ---------------------------
const host               = location.host;                          // e.g. "localhost:8000" or "my‑domain.com"
const WS_PATH            = "/transcribe";                         // websocket endpoint
const API_ROOT           = `${location.protocol}//${host}/api`;    // REST endpoints root
const SILENCE_MS         = 25_000;                                 // 25 s summarisation threshold


const SLICE_MS      = 500;   // 0 .5 s increment
let cumulativeBlobs = [];    // blobs captured so far (for growing window)
let pendingPhrases   = [];   // phrases waiting for auto‑summary
let lastSpeechMs     = 0;    // last speech timestamp
let silenceTimer     = null; // interval timer for silence detection
let isRecording      = false;
let startTime        = null;
let timerInterval    = null;
let currentMode      = "mic"; // "mic" or "file"
let selectedAudioFile= null;
let currentSpeaker   = 1;
let ws               = null;
let wsConnected      = false;
let myvad            = null;
let mediaStream     = null;   // raw getUserMedia stream
let mediaRecorder   = null;   // MediaRecorder instance
let recChunks       = [];     // blobs for the whole session

// ---------------------------
//  DOM elements
// ---------------------------
let startBtn            = document.getElementById("startBtn");
const statusIndicator     = document.getElementById("statusIndicator");
const statusText          = document.getElementById("statusText");
const realtimeTranscription = document.getElementById("realtimeTranscription");
const emptyState          = document.getElementById("emptyState");
const wordCount           = document.getElementById("wordCount");
const timeElapsed         = document.getElementById("timeElapsed");
const regenerateSummary   = document.getElementById("regenerateSummary");
const summaryEditor       = document.getElementById("summaryEditor");
const toolbarBtns         = document.querySelectorAll(".toolbar-btn");
const fontFamily          = document.getElementById("fontFamily");
const fontSize            = document.getElementById("fontSize");
const textColorBtn        = document.getElementById("textColorBtn");
const textColorPicker     = document.getElementById("textColorPicker");
const highlightColorBtn   = document.getElementById("highlightColorBtn");
const highlightColorPicker= document.getElementById("highlightColorPicker");
const microphoneBtn       = document.getElementById("microphoneBtn");
const audioFileBtn        = document.getElementById("audioFileBtn");
const autoSummary         = document.getElementById("autoSummary");
const languageSelect      = document.getElementById("languageSelect");
const noiseFilter         = document.getElementById("noiseFilter");
const speakerDetection    = document.getElementById("speakerDetection");
const spellCheck          = document.getElementById("spellCheck");

const SELECTED_CLASS   = "bg-primary-50 text-primary-700 py-2 px-4 rounded-lg font-medium flex items-center justify-center hover:bg-primary-100 transition";
const UNSELECTED_CLASS = "bg-gray-100  text-gray-700  py-2 px-4 rounded-lg font-medium flex items-center justify-center hover:bg-gray-200  transition";

// ---------------------------
//  Initialisation
// ---------------------------
window.addEventListener("DOMContentLoaded", () => {
  updateSourceButtons();
  updateEmptyStateMessage();
  startBtn.addEventListener("click", toggleRecording);
});

// ---------------------------
//  Toolbar   
// ---------------------------
toolbarBtns.forEach(btn => btn.addEventListener("click", () => {
  const cmd = btn.dataset.command;
  if (cmd) {
    document.execCommand(cmd);
    updateActiveButtons();
  }
}));
fontFamily.onchange          = () => document.execCommand("fontName", false, fontFamily.value);
fontSize.onchange            = () => document.execCommand("fontSize", false, fontSize.value);
textColorBtn.onclick         = () => textColorPicker.click();
highlightColorBtn.onclick    = () => highlightColorPicker.click();
textColorPicker.oninput      = () => document.execCommand("foreColor", false, textColorPicker.value);
highlightColorPicker.oninput = () => document.execCommand("hiliteColor", false, highlightColorPicker.value);
summaryEditor.addEventListener("paste", e => {
  e.preventDefault();
  const txt = (e.clipboardData || e.originalEvent.clipboardData).getData("text/plain");
  document.execCommand("insertText", false, txt);
});
regenerateSummary.onclick = () => maybeAutoSummarize(true);

// ---------------------------
//  Mode switching (mic / file)
// ---------------------------
function updateSourceButtons() {
  microphoneBtn.className = currentMode === "mic"  ? SELECTED_CLASS : UNSELECTED_CLASS;
  audioFileBtn.className  = currentMode === "file" ? SELECTED_CLASS : UNSELECTED_CLASS;
}
function updateEmptyStateMessage() {
  emptyState.querySelector("p").textContent =
    currentMode === "mic"
      ? "Bấm \"Bắt đầu ghi âm\" để bắt đầu chuyển giọng nói thành văn bản"
      : "Bấm \"Bắt đầu phiên âm\" để chuyển tệp âm thanh thành văn bản";
}

microphoneBtn.onclick = () => {
  currentMode = "mic";
  selectedAudioFile = null;
  updateSourceButtons();
  updateEmptyStateMessage();
  resetStartBtn("<i class=\"fas fa-play mr-3\"></i>Bắt đầu ghi âm", toggleRecording);
};

audioFileBtn.onclick = () => {
  currentMode = "file";
  updateSourceButtons();
  updateEmptyStateMessage();
  resetStartBtn("<i class=\"fas fa-play mr-3\"></i>Bắt đầu phiên dịch", toggleTranscribe);
  const inp = Object.assign(document.createElement("input"), { type: "file", accept: "audio/*" });
  inp.onchange = () => { if (inp.files.length) selectedAudioFile = inp.files[0]; };
  inp.click();
};
function blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror   = reject;
    reader.onloadend = () => {
      // result looks like "data:audio/webm;base64,AAAA..."
      const [, base64] = reader.result.split(",");
      resolve(base64);
    };
    reader.readAsDataURL(blob);
  });
}
// ---------------------------
//  Recording control
// ---------------------------
async function toggleRecording() { isRecording ? stopRecording() : startRecording(); }
async function sendCumulativeAudio() {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;

  const combo = new Blob(cumulativeBlobs, { type: "audio/webm" });
  const b64   = await blobToBase64(combo);     // <— safe converter

  ws.send(JSON.stringify({ audio: b64 }));
}

async function startRecording() {
  /* ---------- UI SET‑UP ---------- */
  isRecording = true;
  resetStartBtn("<i class=\"fas fa-stop mr-3\"></i>Dừng ghi âm", toggleRecording);
  startBtn.classList.replace("gradient-bg", "bg-red-500");
  statusIndicator.classList.replace("bg-gray-400", "bg-green-500");
  statusText.textContent = "Đang ghi âm";
  emptyState.style.display = "none";
  realtimeTranscription.innerHTML = "";

  startTime     = new Date();
  timerInterval = setInterval(updateTimer, 1000);
  updateTimer();                         // show 00:00:00 immediately

  /* ---------- WEBSOCKET ---------- */
  try {
    await connectWebSocket();            // now waits until OPEN
  } catch (e) {
    alert("Không kết nối được máy chủ!");
    await stopRecording();
    return;
  }

  /* ---------- MICROPHONE ---------- */
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    console.error("getUserMedia error:", err);
    alert("Không truy cập được microphone!");
    await stopRecording();
    return;
  }

  /* ---------- MEDIARECORDER ---------- */
  cumulativeBlobs = [];                                      // reset buffer
  mediaRecorder   = new MediaRecorder(mediaStream, { mimeType: "audio/webm" });

  // grow the buffer every SLICE_MS and send to backend
  mediaRecorder.ondataavailable = async e => {
    if (e.data && e.data.size) {           // ignore empty chunks
      cumulativeBlobs.push(e.data);        // for streaming
      recChunks.push(e.data);              // for final save
    }
    await sendCumulativeAudio();                             // push latest
  };

  mediaRecorder.start(SLICE_MS);                             // slice size 0.5 s

  /* ---------- (OPTIONAL) SILENCE‑BASED SUMMARY ---------- */
  // Keep this if you still want auto‑summary after long pauses.
  // Otherwise, remove both `silenceTimer` and `checkSilence()`.
  silenceTimer = setInterval(checkSilence, 1000);
}

async function uploadFullRecording(blob) {
  try {
    const fd = new FormData();
    fd.append("file", new File([blob], "session.webm", { type: blob.type }));
    const r  = await fetch(`${API_ROOT}/upload_recording`, { method: "POST", body: fd });
    const d  = await r.json();
    console.log("[REC] stored at", d.saved_as);
  } catch (e) {
    console.error("Failed to upload full recording:", e);
  }
}
async function stopRecording() {
  if (ws && wsConnected) ws.close();

  isRecording = false;
  resetStartBtn("<i class=\"fas fa-play mr-3\"></i>Bắt đầu ghi âm", toggleRecording);
  startBtn.classList.replace("bg-red-500", "gradient-bg");
  statusIndicator.classList.replace("bg-green-500", "bg-gray-400");
  statusText.textContent = "Sẵn sàng";
  clearInterval(timerInterval);
  clearInterval(silenceTimer);

  await maybeAutoSummarize();      // ⬅️ keep summaries first
  await enhanceWholeTranscript();  // ⬅️ single spell-/grammar pass
  updateSpeakerStats();

  pendingPhrases = [];
  lastSpeechMs   = 0;
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    await new Promise(r => {
      mediaRecorder.onstop = r;         // wait until encoder finishes
      mediaRecorder.stop();
    });
    const fullBlob = new Blob(recChunks, { type: "audio/webm" });
    recChunks = [];                     // reset for next session
    uploadFullRecording(fullBlob);      // fire-and-forget
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach(t => t.stop());
    mediaStream = null;
  }
  cumulativeBlobs = [];
}

// ---------------------------
//  Silence auto‑summary
// ---------------------------
async function checkSilence() {
  if (!autoSummary.checked) return; 
  if (!pendingPhrases.length) return;
  if (Date.now() - lastSpeechMs < SILENCE_MS) return;

  const block = pendingPhrases.join(" ");
  pendingPhrases = [];

  summaryEditor.innerHTML = "<em>Đang tóm tắt…</em>";
  const sum = await fetchSummary(block);
  summaryEditor.innerHTML = sum ? marked.parse(sum) : "<span class=\"text-red-600\">Không thể tạo tóm tắt!</span>";
}

// ---------------------------
//  WebSocket
// ---------------------------
function connectWebSocket() {
  return new Promise((resolve, reject) => {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(`${proto}//${host}${WS_PATH}`);

    ws.onopen  = () => { wsConnected = true;  resolve(); };
    ws.onerror = e  => { console.error("[WS] error", e); reject(e); };
    ws.onclose = () => { wsConnected = false; };

    ws.onmessage = ev => {
      const { status, transcription, text, message } = JSON.parse(ev.data);
      if (status !== "success") { console.error("[WS] server error:", message); return; }
    
      // 🔄  wipe the old transcript completely
      realtimeTranscription.innerHTML = "";
      pendingCards   = [];         // clear card buffer
      pendingPhrases = [];         // clear summary buffer
    
      addTranscriptionItem(1, transcription ?? text ?? ""); // always speaker 1
      wordCount.textContent = realtimeTranscription.textContent.trim().split(/\s+/).length;
    };
  });
}

// ---------------------------
//  VAD
// ---------------------------
// async function initVAD() {
//   try {
//     myvad = await vad.MicVAD.new({
//       onSpeechStart: () => (statusText.textContent = "Nghe…"),
//       onSpeechEnd  : async (audio) => {
//         statusText.textContent = "Đang xử lý…";
//         if (!wsConnected) return;
//         const pcm = Float32ArrayToInt16(audio);
//         if (pcm.length / 16_000 < 0.5) return; // ignore <0.5 s
//         const b64 = int16ToBase64(pcm);
//         ws.send(JSON.stringify({
//           audio: b64,
//           denoise: noiseFilter.checked,
//           // enhance: spellCheck.checked
//         }));
//       },
//       positiveSpeechThreshold : 0.8,
//       negativeSpeechThreshold : 0.8,
//       minSpeechFrames         : 4,
//       preSpeechPadFrames      : 1,
//       redemptionFrames        : 3
//     });
//     return true;
//   } catch (e) {
//     console.error("VAD init failed:", e);
//     alert("Không khởi tạo được microphone/VAD!");
//     return false;
//   }
// }

function Float32ArrayToInt16(arr) {
  const pcm = new Int16Array(arr.length);
  for (let i = 0; i < arr.length; i++) pcm[i] = Math.max(-1, Math.min(1, arr[i])) * 0x7FFF;
  return pcm;
}
function int16ToBase64(pcm) {
  const bytes = new Uint8Array(pcm.buffer);
  return btoa(String.fromCharCode(...bytes));
}

async function enhanceWholeTranscript() {
  if (!spellCheck.checked) return;                // user didn’t ask for it
  const raw = realtimeTranscription.textContent.trim();
  if (!raw) return;

  statusText.textContent = "Đang cải thiện chính tả…";
  try {
    const r = await fetch(`${API_ROOT}/enhance_text`, {
      method : "POST",
      headers: { "Content-Type": "application/json" },
      body   : JSON.stringify({ text: raw })
    });
    const d = await r.json();
    if (d.text) {
      // keep markdown rendering consistent with the rest of the UI
      realtimeTranscription.innerHTML = marked.parse(d.text);
    }
  } catch (e) {
    console.error(e);
  } finally {
    statusText.textContent = "Sẵn sàng";
  }
}
// ---------------------------
//  File transcription (upload)
// ---------------------------
async function toggleTranscribe() {
  if (!selectedAudioFile) { alert("Bạn chưa chọn file âm thanh!"); return; }
  startBtn.disabled = true;
  statusIndicator.classList.replace("bg-gray-400", "bg-blue-500");
  statusText.textContent = "Đang xử lý tệp…";

  try {
    const fd = new FormData();
    fd.append("file", selectedAudioFile);
    fd.append("denoise", noiseFilter.checked);
    fd.append("diarize", speakerDetection.checked);
    fd.append("enhance", spellCheck.checked);

    const res = await fetch(`${API_ROOT}/transcribe`, { method: "POST", body: fd });
    if (!res.ok) throw new Error((await res.json()).error || `HTTP ${res.status}`);
    const data = await res.json();

    emptyState.style.display = "none";
    realtimeTranscription.innerHTML = "";

    if (data.segments) {
      data.segments.forEach(seg => {
        const ts = msToTimestamp(seg.start, seg.end);
        addTranscriptionItem(seg.speaker.replace("SPEAKER_", ""), `[${ts}] ${seg.text}`);
      });
    } else {
      addTranscriptionItem(1, data.transcription);
    }

    if (data.original_audio || data.enhanced_audio) addAudioPlayers(data.original_audio, data.enhanced_audio);
    await maybeAutoSummarize();
  } catch (err) {
    console.error(err);
    alert("Có lỗi khi phiên dịch file!");
  } finally {
    startBtn.disabled = false;
    statusIndicator.classList.replace("bg-blue-500", "bg-gray-400");
    statusText.textContent = "Sẵn sàng";
  }
}

// ---------------------------
//  Helpers
// ---------------------------
function resetStartBtn(html, handler) {
  const clone         = startBtn.cloneNode(false);
  clone.innerHTML     = html;
  clone.className     = "w-full gradient-bg text-white py-3 rounded-lg font-bold text-lg flex items-center justify-center hover:opacity-90 transition hover-glow";
  clone.onclick       = handler;
  startBtn.replaceWith(clone);
  startBtn = clone;
}

function updateTimer() {
  const t = new Date(Date.now() - startTime);
  timeElapsed.textContent = `${t.getUTCHours().toString().padStart(2, "0")}:${t.getUTCMinutes().toString().padStart(2, "0")}:${t.getUTCSeconds().toString().padStart(2, "0")}`;
}

function addTranscriptionItem(spk, text) {
  const phrase = text.trim();
  if (!phrase) return;

  const colors = ["text-primary-600", "text-orange-600", "text-red-600"];
  const bgs    = ["bg-primary-50", "bg-orange-50", "bg-red-50"];
  const idx    = spk >= 1 && spk <= colors.length ? spk - 1 : 0;

  const lastCard = realtimeTranscription.lastElementChild;
  if (lastCard && lastCard.dataset.speaker === spk.toString()) {
    lastCard.querySelector("p").innerHTML += " " + phrase;
    lastCard.querySelector(".card-time").textContent = formatTime(new Date());
    return;
  }

  const div = document.createElement("div");
  div.className     = `mb-3 p-3 rounded-lg ${bgs[idx]} transition-all`;
  div.dataset.speaker = spk;
  div.innerHTML = `
    <div class=\"flex items-start\">
      <span class=\"font-medium ${colors[idx]} mr-2 w-24 flex-shrink-0\">Người nói ${spk}:</span>
      <p class=\"flex-grow\">${phrase}</p>
    </div>
    <div class=\"text-xs text-gray-500 mt-1 text-right card-time\">${formatTime(new Date())}</div>`;

  realtimeTranscription.appendChild(div);
  realtimeTranscription.scrollTop = realtimeTranscription.scrollHeight;
}

function formatTime(d) {
  return `${d.getHours().toString().padStart(2, "0")}:${d.getMinutes().toString().padStart(2, "0")}`;
}

function msToTimestamp(start, end) {
  const s = ms2ts(start);
  const e = ms2ts(end);
  return `${s}-${e}`;
}
function ms2ts(m) {
  return new Date(m).toISOString().slice(11, 19);
}

function updateActiveButtons() {
  toolbarBtns.forEach(btn => {
    const cmd = btn.dataset.command;
    if (cmd) btn.classList.toggle("active", document.queryCommandState(cmd));
  });
}

async function fetchSummary(raw) {
  try {
    const r = await fetch(`${API_ROOT}/summarize_text`, {
      method : "POST",
      headers: { "Content-Type": "application/json" },
      body   : JSON.stringify({ text: raw, lang: languageSelect.value })
    });
    const d = await r.json();
    return d.summary || null;
  } catch (e) {
    console.error(e);
    return null;
  }
}

async function maybeAutoSummarize(force = false) {
  if (!force && !autoSummary.checked) return;
  const plain = realtimeTranscription.textContent.trim();
  if (!plain) return;
  summaryEditor.innerHTML = "<em>Đang tạo tóm tắt…</em>";
  const summary = await fetchSummary(plain);
  summaryEditor.innerHTML = summary ? marked.parse(summary) : "<span class=\"text-red-600\">Không thể tạo tóm tắt!</span>";
}

function updateSpeakerStats() {
  const els = document.querySelectorAll("#speakerList > div");
  if (!els.length) return;
  els[0].querySelector("p:last-child").textContent = "45% thời lượng";
  els[1].querySelector("p:last-child").textContent = "35% thời lượng";
  els[2].querySelector("p:last-child").textContent = "20% thời lượng";
}

function addAudioPlayers(origB64, enhB64) {
  const area = document.getElementById("audioPreviewArea");
  if (!area) return;
  area.innerHTML = "";
  const make = (label, b64) => `
    <p class=\"text-sm font-medium\">${label}</p>
    <audio controls class=\"w-full rounded\" src=\"data:audio/wav;base64,${b64}\"></audio>`;
  if (origB64) area.insertAdjacentHTML("beforeend", make("Gốc:", origB64));
  if (enhB64)  area.insertAdjacentHTML("beforeend", make("Sau lọc nhiễu:", enhB64));
}
