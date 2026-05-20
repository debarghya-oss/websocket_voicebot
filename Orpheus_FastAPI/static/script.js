document.addEventListener('DOMContentLoaded', () => {
    console.log("DOM loaded");

    // ================================================================
    // CONFIG
    // ================================================================
    const RAG_BASE = 'http://localhost:8080';  // RAG service mounted on main FastAPI app

    // ================================================================
    // UI REFERENCES
    // ================================================================
    const appForm              = document.getElementById('app-form');
    const textInput            = document.getElementById('text_input');
    const generateButton       = document.getElementById('generate-button');
    const audioPlayer          = document.getElementById('audio-player');
    const modeSelect           = document.getElementById('mode_select');
    const chatHistoryDisplay   = document.getElementById('chat-history-display');
    const ttsVoiceSelect       = document.getElementById('tts_voice_dd');
    const recordButton         = document.getElementById('record-button');
    const recordStatus         = document.getElementById('record-status');

    // STT
    const sttEngineSelect      = document.getElementById('stt_engine_select');
    const indicLangGroup       = document.getElementById('indic-lang-group');
    const indicDecodeGroup     = document.getElementById('indic-decode-group');
    const indicLangSelect      = document.getElementById('indic_language_select');
    const indicDecodeSelect    = document.getElementById('indic_decode_select');

    // LLM
    const llmModelSelect       = document.getElementById('llm_model_select');
    const refreshLlmModelsBtn  = document.getElementById('refresh-llm-models-btn');
    const llmMaxTokensInput    = document.getElementById('llm_max_tokens_input');

    // User Prompt
    const applyPromptBtn       = document.getElementById('apply-prompt-btn');
    const userPromptTextarea   = document.getElementById('user-prompt-textarea');

    // Upload
    const uploadButton         = document.getElementById('upload-button');
    const fileInput            = document.getElementById('file-input');
    const fileChipRow          = document.getElementById('file-chip-row');

    // ================================================================
    // STATIC DATA
    // ================================================================
    const ttsVoices = ["tara", "jess", "leo", "leah", "dan", "mia", "zac", "zoe"];

    // ================================================================
    // AUDIO STATE
    // ================================================================
    let audioContext               = null;
    let currentAudioSampleRate     = 8000;
    let isPlayingAudio             = false;
    let nextAudioStartTime         = 0;
    let useWebSocketForAudio       = true;

    // ================================================================
    // CHAT / LLM STATE
    // ================================================================
    let chatHistory                           = [];
    let currentLLMStreamController            = null;
    let accumulatedLLMTextForDisplay          = "";
    let currentAssistantMessageContentElement = null;

    // ================================================================
    // RECORDING STATE
    // ================================================================
    let mediaRecorder      = null;
    let audioChunks        = [];
    let isRecording        = false;
    let isPushToTalkActive = false;
    let spaceBarIsDown     = false;

    // ================================================================
    // RAG STATE
    // key: filename, value: { jobId, status, chunks, topic, file }
    // ================================================================
    let ragFiles = {};        // tracks all ingested / ingesting files
    let ragActive = false;    // true when at least one file is fully ingested

    // ================================================================
    // INIT TTS VOICES
    // ================================================================
    function initTTSVoices() {
        if (!ttsVoiceSelect) return;
        ttsVoices.forEach(v => {
            const o = document.createElement('option');
            o.value = v;
            o.textContent = v.charAt(0).toUpperCase() + v.slice(1);
            if (v === "tara") o.selected = true;
            ttsVoiceSelect.appendChild(o);
        });
    }

    // ================================================================
    // STT PARAMS
    // ================================================================
    function getSttParams() {
        const engine     = sttEngineSelect   ? sttEngineSelect.value   : 'indic';
        const language   = indicLangSelect   ? indicLangSelect.value   : 'bn';
        const decodeMode = indicDecodeSelect ? indicDecodeSelect.value : 'ctc';
        return { engine, language, decodeMode };
    }

    function updateIndicVisibility() {
        const { engine } = getSttParams();
        const isIndic = engine === 'indic';
        if (indicLangGroup)  indicLangGroup.style.display  = isIndic ? '' : 'none';
        if (indicDecodeGroup) indicDecodeGroup.style.display = isIndic ? '' : 'none';
    }

    async function onSttEngineChange() {
        updateIndicVisibility();
        const { engine, language, decodeMode } = getSttParams();
        if (recordStatus) recordStatus.textContent = `Engine: ${engine}`;
        try {
            const body = { engine };
            if (engine === 'indic') { body.language = language; body.decode_mode = decodeMode; }
            const res = await fetch('/api/stt/set_engine', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            });
            if (res.ok && recordStatus) recordStatus.textContent = `✓ Engine: ${engine}`;
        } catch (err) { console.error("STT engine switch failed:", err); }
    }

    if (sttEngineSelect) {
        sttEngineSelect.addEventListener('change', onSttEngineChange);
        if (indicLangSelect)  indicLangSelect.addEventListener('change', onSttEngineChange);
        if (indicDecodeSelect) indicDecodeSelect.addEventListener('change', onSttEngineChange);
        onSttEngineChange();
    }

    // ================================================================
    // FETCH LLM MODELS
    // ================================================================
    async function fetchLLMModels() {
        if (!llmModelSelect) return;
        llmModelSelect.innerHTML = '<option value="">— loading… —</option>';
        try {
            const res = await fetch("/api/llm/models");
            const data = await res.json();
            const models = data.models || [];
            llmModelSelect.innerHTML = '';
            if (models.length === 0) {
                llmModelSelect.innerHTML = '<option value="">— no models found —</option>';
                return;
            }
            models.forEach((m, i) => {
                const o = document.createElement('option');
                o.value = m; o.textContent = m;
                if (i === 0) o.selected = true;
                llmModelSelect.appendChild(o);
            });
        } catch (err) {
            console.error("Failed to fetch models:", err);
            llmModelSelect.innerHTML = '<option value="">— fetch error —</option>';
        }
    }

    if (refreshLlmModelsBtn) refreshLlmModelsBtn.addEventListener('click', fetchLLMModels);
    fetchLLMModels();

    // ================================================================
    // CHAT HISTORY
    // ================================================================
    function renderChatHistory() {
        if (!chatHistoryDisplay) return null;
        chatHistoryDisplay.innerHTML = '';
        let lastAssistantDiv = null;
        chatHistory.forEach(msg => {
            const wrap  = document.createElement('div');
            wrap.classList.add('chat-message', msg.role === 'user' ? 'user-message' : 'assistant-message');
            const inner = document.createElement('div');
            inner.classList.add('message-content-wrapper');
            const strong = document.createElement('strong');
            strong.textContent = msg.role === 'user' ? 'You:' : 'Assistant:';
            inner.appendChild(strong);
            const content = document.createElement('div');
            if (msg.role === 'assistant' && msg.isStreaming) content.classList.add('streaming-llm-content');
            content.textContent = msg.content;
            inner.appendChild(content);
            wrap.appendChild(inner);
            chatHistoryDisplay.appendChild(wrap);
            if (msg.role === 'assistant') lastAssistantDiv = content;
        });
        chatHistoryDisplay.scrollTop = chatHistoryDisplay.scrollHeight;
        return lastAssistantDiv;
    }

    function addUserMessage(text) {
        chatHistory.push({ role: 'user', content: text });
        renderChatHistory();
    }

    function addAssistantMessage(text, isStreaming = false) {
        const last = chatHistory.length > 0 ? chatHistory[chatHistory.length - 1] : null;
        if (isStreaming && last && last.role === 'assistant' && last.isStreaming) {
            last.content = text;
        } else {
            chatHistory.push({ role: 'assistant', content: text, isStreaming });
        }
        return renderChatHistory();
    }

    // ================================================================
    // USER PROMPT → RAG MODELFILE
    // ================================================================
    if (applyPromptBtn && userPromptTextarea) {
        applyPromptBtn.addEventListener('click', async () => {
            const promptText = userPromptTextarea.value.trim();
            if (!promptText) return;

            applyPromptBtn.disabled = true;
            applyPromptBtn.textContent = 'Applying…';

            // Store globally for non-RAG LLM calls
            window.userSystemPrompt = promptText;

            try {
                // Upload prompt as a text Modelfile to the RAG service
                const blob     = new Blob([promptText], { type: 'text/plain' });
                const formData = new FormData();
                formData.append('file', blob, 'system_prompt.txt');

                const res  = await fetch(`${RAG_BASE}/modelfile/upload`, {
                    method: 'POST',
                    body: formData
                });

                if (res.ok) {
                    const data = await res.json();
                    console.log('[Modelfile] Applied:', data);
                    applyPromptBtn.textContent = '✓ Applied';
                    applyPromptBtn.classList.add('applied');
                } else {
                    const err = await res.json().catch(() => ({}));
                    console.warn('[Modelfile] Upload failed:', err);
                    applyPromptBtn.textContent = '✓ Saved locally';
                    applyPromptBtn.classList.add('applied');
                }
            } catch (err) {
                // RAG service may not be running — still store locally
                console.warn('[Modelfile] RAG service unreachable, stored locally:', err);
                applyPromptBtn.textContent = '✓ Saved locally';
                applyPromptBtn.classList.add('applied');
            } finally {
                applyPromptBtn.disabled = false;
                setTimeout(() => {
                    applyPromptBtn.textContent = 'Apply';
                    applyPromptBtn.classList.remove('applied');
                }, 2500);
            }
        });
    }

    // ================================================================
    // FILE UPLOAD → RAG INGEST
    // ================================================================
    function formatSize(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    }

    function fileIcon(name) {
        const ext = name.split('.').pop().toLowerCase();
        return { pdf: '📄', txt: '📝', md: '📝', doc: '📄', docx: '📄' }[ext] || '📁';
    }

    // Update the overall ragActive flag
    function refreshRagActive() {
        ragActive = Object.values(ragFiles).some(f => f.status === 'done');
    }

    // Render / update a single file chip
    function upsertFileChip(filename, state) {
        let chip = fileChipRow.querySelector(`[data-filename="${CSS.escape(filename)}"]`);

        if (!chip) {
            chip = document.createElement('div');
            chip.className = 'file-chip';
            chip.dataset.filename = filename;
            fileChipRow.appendChild(chip);
            fileChipRow.classList.add('visible');
            if (uploadButton) uploadButton.classList.add('has-file');
        }

        const { status, chunks, size } = state;

        let statusIcon, statusColor, statusText;
        switch (status) {
            case 'queued':
            case 'processing':
                statusIcon  = '⏳';
                statusColor = 'var(--text-muted)';
                statusText  = status === 'queued' ? 'Queued…' : 'Ingesting…';
                break;
            case 'done':
                statusIcon  = '✓';
                statusColor = 'var(--accent)';
                statusText  = `${chunks} chunks`;
                break;
            case 'failed':
                statusIcon  = '✗';
                statusColor = 'var(--error)';
                statusText  = 'Failed';
                break;
            default:
                statusIcon  = '…';
                statusColor = 'var(--text-muted)';
                statusText  = status;
        }

        chip.innerHTML = `
            <span class="file-chip-icon">${fileIcon(filename)}</span>
            <span class="file-chip-name" title="${filename}">${filename}</span>
            <span class="file-chip-size">${formatSize(size)}</span>
            <span class="file-chip-status" style="color:${statusColor};font-size:0.7rem;font-weight:600;flex-shrink:0;">
                ${statusIcon} ${statusText}
            </span>
            <button type="button" class="file-chip-remove" title="Remove">✕</button>
        `;

        chip.querySelector('.file-chip-remove').addEventListener('click', () => {
            delete ragFiles[filename];
            chip.remove();
            refreshRagActive();
            if (!fileChipRow.children.length) {
                fileChipRow.classList.remove('visible');
                if (uploadButton) uploadButton.classList.remove('has-file');
            }
        });
    }

    // Poll ingest job until it finishes
    async function pollIngestJob(filename, jobId) {
        const POLL_INTERVAL = 1500;   // ms
        const MAX_POLLS     = 80;     // ~2 min max
        let polls           = 0;

        return new Promise(resolve => {
            const timer = setInterval(async () => {
                polls++;
                try {
                    const res  = await fetch(`${RAG_BASE}/ingest/status/${jobId}`);
                    const data = await res.json();

                    const status = data.status;
                    ragFiles[filename].status = status;
                    ragFiles[filename].chunks = data.chunks_ingested || 0;
                    upsertFileChip(filename, ragFiles[filename]);

                    if (status === 'done' || status === 'failed' || polls >= MAX_POLLS) {
                        clearInterval(timer);
                        refreshRagActive();
                        resolve(status);
                    }
                } catch (err) {
                    console.error(`[RAG] Poll error for ${filename}:`, err);
                    ragFiles[filename].status = 'failed';
                    upsertFileChip(filename, ragFiles[filename]);
                    clearInterval(timer);
                    refreshRagActive();
                    resolve('failed');
                }
            }, POLL_INTERVAL);
        });
    }

    // Upload a file to the RAG ingest endpoint
    async function ingestFile(file) {
        const filename = file.name;

        // Register the file
        ragFiles[filename] = { status: 'queued', chunks: 0, size: file.size, file };
        upsertFileChip(filename, ragFiles[filename]);

        try {
            const formData = new FormData();
            formData.append('file', file, filename);
            formData.append('topic', 'general');

            const res = await fetch(`${RAG_BASE}/ingest/`, {
                method: 'POST',
                body: formData
            });

            if (!res.ok) {
                const err = await res.json().catch(() => ({}));
                throw new Error(err.detail || `HTTP ${res.status}`);
            }

            const data = await res.json();
            ragFiles[filename].jobId  = data.job_id;
            ragFiles[filename].status = data.status;
            upsertFileChip(filename, ragFiles[filename]);

            // Poll until done/failed
            await pollIngestJob(filename, data.job_id);

        } catch (err) {
            console.error(`[RAG] Ingest failed for ${filename}:`, err);
            ragFiles[filename].status = 'failed';
            upsertFileChip(filename, ragFiles[filename]);
            refreshRagActive();
        }
    }

    // Wire upload button
    if (uploadButton && fileInput) {
        uploadButton.addEventListener('click', () => fileInput.click());

        fileInput.addEventListener('change', () => {
            const files = Array.from(fileInput.files);
            files.forEach(file => {
                if (!ragFiles[file.name]) {   // skip duplicates already tracked
                    ingestFile(file);
                }
            });
            fileInput.value = '';
        });
    }

    // ================================================================
    // RAG ASK  (replaces LLM stream when docs are ingested)
    // ================================================================
    async function askRAG(question, top_k = 5) {
        const res = await fetch(`${RAG_BASE}/ask`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, top_k })
        });
        if (!res.ok) throw new Error(`RAG /ask returned ${res.status}`);
        return await res.json();   // { answer, sources, … }
    }

    // ================================================================
    // RECORDING
    // ================================================================
    const SVG_MIC = `
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"
         style="width:1.2em;height:1.2em;">
        <path d="M12 1a4 4 0 0 1 4 4v7a4 4 0 0 1-8 0V5a4 4 0 0 1 4-4z"/>
    </svg>`;

    async function startRecording() {
        if (!navigator.mediaDevices?.getUserMedia) { alert("Browser doesn't support recording."); return; }
        try {
            const stream   = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder  = new MediaRecorder(stream);
            audioChunks    = [];
            mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
            mediaRecorder.onstop = async () => {
                const blob = new Blob(audioChunks, { type: 'audio/webm' });
                audioChunks = [];
                stream.getTracks().forEach(t => t.stop());
                await sendAudioForTranscription(blob, 'recording.webm');
            };
            mediaRecorder.start();
            isRecording = true;
            if (recordButton) { recordButton.innerHTML = '🛑'; recordButton.classList.add('recording'); }
            if (recordStatus) recordStatus.textContent = 'Recording…';
        } catch (err) {
            console.error("Mic error:", err);
            alert("Microphone permission denied.");
            isRecording = false;
        }
    }

    function stopRecording() {
        if (mediaRecorder && mediaRecorder.state === 'recording') {
            mediaRecorder.stop();
            isRecording = false;
            if (recordButton) { recordButton.innerHTML = SVG_MIC; recordButton.classList.remove('recording'); }
            if (recordStatus) recordStatus.textContent = 'Processing…';
        }
    }

    async function sendAudioForTranscription(blob, fileName) {
        const { engine, language, decodeMode } = getSttParams();
        try {
            if (recordStatus) recordStatus.textContent = `Transcribing (${engine})…`;
            let result;
            if (useWebSocketForAudio) {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                result = await sendAudioViaWebSocket(blob, `${protocol}//${window.location.host}/ws/stt`);
            } else {
                const params = new URLSearchParams({ engine });
                if (engine === 'indic') { params.set('language', language); params.set('decode_mode', decodeMode); }
                const form = new FormData();
                form.append('audio_file', blob, fileName);
                const res = await fetch(`/api/stt/transcribe?${params}`, { method: 'POST', body: form });
                result = await res.json();
                if (!res.ok) throw new Error(result.detail || 'STT failed');
            }
            if (textInput) textInput.value = result.text || '';
            if (recordStatus) recordStatus.textContent = `✓ ${engine}`;
        } catch (err) {
            console.error("STT ERROR:", err);
            alert(`STT Error: ${err.message}`);
            if (recordStatus) recordStatus.textContent = 'STT failed!';
        }
    }

    if (recordButton) {
        recordButton.addEventListener('click', () => {
            if (isRecording) stopRecording(); else startRecording();
        });
    }

    // Push-to-talk
    document.addEventListener('keydown', async e => {
        if (e.code !== 'Space') return;
        if (document.activeElement?.tagName.toLowerCase() === 'textarea') return;
        e.preventDefault();
        if (!isRecording && !spaceBarIsDown) {
            spaceBarIsDown = true; isPushToTalkActive = true;
            await startRecording();
        }
    });
    document.addEventListener('keyup', async e => {
        if (e.code !== 'Space') return;
        if (spaceBarIsDown) {
            spaceBarIsDown = false;
            if (isRecording && isPushToTalkActive) stopRecording();
            isPushToTalkActive = false;
        }
    });

    // ================================================================
    // TTS – HTTP streaming
    // ================================================================
    async function streamTTSAudio(response) {
        const headerSampleRate = parseInt(response.headers.get('X-Sample-Rate') || '8000');
        currentAudioSampleRate = headerSampleRate;
        if (!audioContext) audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: currentAudioSampleRate });
        if (audioContext.state === 'suspended') await audioContext.resume();

        const reader = response.body.getReader();
        let nextStartTime = audioContext.currentTime + 0.2;
        let leftoverBytes = new Uint8Array(0);

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            const combined = new Uint8Array(leftoverBytes.length + value.length);
            combined.set(leftoverBytes); combined.set(value, leftoverBytes.length);
            const numFloats = Math.floor(combined.length / 4);
            const completeBytesLength = numFloats * 4;
            leftoverBytes = combined.slice(completeBytesLength);
            if (numFloats > 0) {
                const float32Data = new Float32Array(combined.buffer.slice(combined.byteOffset, combined.byteOffset + completeBytesLength));
                const audioBuffer = audioContext.createBuffer(1, numFloats, currentAudioSampleRate);
                audioBuffer.copyToChannel(float32Data, 0);
                const source = audioContext.createBufferSource();
                source.buffer = audioBuffer; source.connect(audioContext.destination);
                if (nextStartTime < audioContext.currentTime) nextStartTime = audioContext.currentTime + 0.05;
                source.start(nextStartTime);
                nextStartTime += audioBuffer.duration;
            }
        }
    }

    // ================================================================
    // BASE64 UTILITIES
    // ================================================================
    function base64ToArrayBuffer(base64) {
        const b = atob(base64), bytes = new Uint8Array(b.length);
        for (let i = 0; i < b.length; i++) bytes[i] = b.charCodeAt(i);
        return bytes.buffer;
    }
    function arrayBufferToBase64(buffer) {
        const bytes = new Uint8Array(buffer);
        let bin = '';
        for (let i = 0; i < bytes.byteLength; i++) bin += String.fromCharCode(bytes[i]);
        return btoa(bin);
    }

    // ================================================================
    // TTS – WebSocket streaming
    // ================================================================
    function streamTTSAudioWebSocket(wsUrl, ttsRequest = null) {
        return new Promise((resolve, reject) => {
            const ws = new WebSocket(wsUrl);
            ws.onopen = () => { if (ttsRequest) ws.send(JSON.stringify(ttsRequest)); };
            ws.onmessage = async (event) => {
                try {
                    const msg = JSON.parse(event.data);
                    if (msg.type === 'tts_started') {
                        currentAudioSampleRate = msg.sample_rate || 8000;
                        if (!audioContext) audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: currentAudioSampleRate });
                        if (audioContext.state === 'suspended') await audioContext.resume();
                        isPlayingAudio = true;
                        nextAudioStartTime = audioContext.currentTime + 0.2;
                    } else if (msg.type === 'tts_audio_chunk') {
                        const float32Data = new Float32Array(base64ToArrayBuffer(msg.audio));
                        if (float32Data.length > 0) {
                            const audioBuffer = audioContext.createBuffer(1, float32Data.length, currentAudioSampleRate);
                            audioBuffer.copyToChannel(float32Data, 0);
                            const source = audioContext.createBufferSource();
                            source.buffer = audioBuffer; source.connect(audioContext.destination);
                            if (nextAudioStartTime < audioContext.currentTime) nextAudioStartTime = audioContext.currentTime + 0.05;
                            source.start(nextAudioStartTime);
                            nextAudioStartTime += audioBuffer.duration;
                        }
                    } else if (msg.type === 'tts_done') {
                        isPlayingAudio = false; ws.close(); resolve();
                    } else if (msg.type === 'error') {
                        ws.close(); reject(new Error(msg.message));
                    }
                } catch (err) { reject(err); }
            };
            ws.onerror  = err => reject(err);
            ws.onclose  = () => { isPlayingAudio = false; };
        });
    }

    // ================================================================
    // STT – WebSocket
    // ================================================================
    async function sendAudioViaWebSocket(blob, wsUrl) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = e => {
                const base64Audio = arrayBufferToBase64(new Uint8Array(e.target.result));
                const ws = new WebSocket(wsUrl);
                ws.onopen = () => {
                    const { engine, language, decodeMode } = getSttParams();
                    ws.send(JSON.stringify({ engine, language, decode_mode: decodeMode, audio: base64Audio }));
                };
                ws.onmessage = event => {
                    const msg = JSON.parse(event.data);
                    if (msg.type === 'stt_result') { resolve({ text: msg.text, engine: msg.engine, language: msg.language }); ws.close(); }
                    else if (msg.type === 'error')  { reject(new Error(msg.message)); ws.close(); }
                };
                ws.onerror = err => reject(err);
            };
            reader.onerror = () => reject(new Error("Failed to read blob"));
            reader.readAsArrayBuffer(blob);
        });
    }

    // ================================================================
    // TTS HELPER — send text to TTS and play it
    // ================================================================
    async function speakText(text) {
        const voice       = ttsVoiceSelect ? ttsVoiceSelect.value : "tara";
        const ttsTemp     = parseFloat(document.getElementById('tts_temp_slider')?.value   || "2.0");
        const ttsTopP     = parseFloat(document.getElementById('tts_top_p_slider')?.value   || "0.9");
        const bufferGrps  = parseInt(document.getElementById('tts_buffer_groups_slider')?.value || "5");
        const paddingMs   = parseInt(document.getElementById('tts_padding_ms_slider')?.value    || "0");
        const batchGrps   = parseInt(document.getElementById('tts_batch_groups_slider')?.value  || "7");

        const ttsReq = {
            text, voice,
            tts_temperature: ttsTemp, tts_top_p: ttsTopP,
            buffer_groups: bufferGrps, padding_ms: paddingMs,
            min_decode_batch_groups: batchGrps
        };

        if (useWebSocketForAudio) {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            await streamTTSAudioWebSocket(`${protocol}//${window.location.host}/ws/tts`, ttsReq);
        } else {
            const res = await fetch('/api/tts/stream', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(ttsReq)
            });
            if (!res.ok) throw new Error("TTS request failed");
            await streamTTSAudio(res);
        }
    }

    // ================================================================
    // FORM SUBMIT
    // ================================================================
    if (appForm) {
        appForm.addEventListener('submit', async e => {
            e.preventDefault();

            const userText = textInput.value.trim();
            if (!userText) { alert("Enter some text."); return; }

            addUserMessage(userText);
            textInput.value = '';

            const mode = modeSelect ? modeSelect.value : 'llm_only';

            if (generateButton) {
                generateButton.disabled    = true;
                generateButton.textContent = "Generating…";
            }

            // ── TTS-only mode ────────────────────────────────────────
            if (mode === 'tts_only') {
                try {
                    await speakText(userText);
                } catch (err) {
                    console.error("TTS Error:", err);
                    alert("TTS generation failed.");
                } finally {
                    if (generateButton) {
                        generateButton.disabled    = false;
                        generateButton.textContent = "Generate";
                    }
                }
                return;
            }

            // ── LLM / RAG (+ optional TTS) mode ─────────────────────
            const assistantEl = addAssistantMessage("", true);
            currentAssistantMessageContentElement = assistantEl;
            accumulatedLLMTextForDisplay = "";

            try {

                // ── RAG path: at least one document is fully ingested ──
                if (ragActive) {
                    if (generateButton) generateButton.textContent = "Searching docs…";

                    const ragResult = await askRAG(userText, 5);
                    const answer    = ragResult.answer || "No answer returned.";

                    // Display answer
                    accumulatedLLMTextForDisplay = answer;
                    if (currentAssistantMessageContentElement) {
                        currentAssistantMessageContentElement.textContent = answer;

                        // Append source citations if any
                        if (ragResult.sources && ragResult.sources.length > 0) {
                            const srcDiv = document.createElement('div');
                            srcDiv.style.cssText = 'margin-top:8px;font-size:0.72rem;color:var(--text-muted);border-top:1px solid var(--border);padding-top:6px;';
                            srcDiv.textContent = '📚 Sources: ' + ragResult.sources
                                .map(s => `${s.document} (${s.topic}, score: ${s.score})`)
                                .join(' · ');
                            currentAssistantMessageContentElement.parentElement.appendChild(srcDiv);
                        }
                    }

                    // Update history
                    const lastMsg = chatHistory[chatHistory.length - 1];
                    if (lastMsg && lastMsg.role === 'assistant') {
                        lastMsg.content = answer; lastMsg.isStreaming = false;
                    }
                    chatHistoryDisplay.scrollTop = chatHistoryDisplay.scrollHeight;

                    // Speak if LLM+TTS
                    if (mode === 'llm_tts' && answer.trim()) {
                        if (generateButton) generateButton.textContent = "Generating TTS…";
                        await speakText(answer);
                    }

                } else {
                    // ── Standard LLM streaming path ───────────────────
                    const temp       = parseFloat(document.getElementById('llm_temp_slider')?.value       || "0.7");
                    const topP       = parseFloat(document.getElementById('llm_top_p_slider')?.value       || "0.9");
                    const repPenalty = parseFloat(document.getElementById('llm_rep_penalty_slider')?.value || "1.1");
                    const topK       = parseInt(document.getElementById('llm_top_k_slider')?.value         || "45");
                    const maxTokens  = parseInt(llmMaxTokensInput?.value                                   || "-1");
                    const model      = llmModelSelect ? llmModelSelect.value : null;

                    const historyForAPI = chatHistory
                        .filter(msg => !msg.isStreaming && msg.role !== 'system')
                        .slice(0, -1)
                        .map(msg => ({ role: msg.role, content: msg.content }));

                    const requestBody = {
                        prompt: userText,
                        history: historyForAPI,
                        model, temperature: temp, top_p: topP,
                        max_tokens: maxTokens, repetition_penalty: repPenalty, top_k: topK
                    };

                    // Inject user system prompt if set
                    if (window.userSystemPrompt) {
                        requestBody.system_prompt = window.userSystemPrompt;
                    }

                    const response = await fetch('/api/llm/chat/stream', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(requestBody)
                    });

                    if (!response.ok) throw new Error(`Server returned ${response.status}: ${response.statusText}`);

                    const reader  = response.body.getReader();
                    const decoder = new TextDecoder('utf-8');
                    let done      = false;

                    while (!done) {
                        const { value, done: readerDone } = await reader.read();
                        done = readerDone;
                        if (value) {
                            accumulatedLLMTextForDisplay += decoder.decode(value, { stream: true });
                            if (currentAssistantMessageContentElement) {
                                currentAssistantMessageContentElement.textContent = accumulatedLLMTextForDisplay;
                                chatHistoryDisplay.scrollTop = chatHistoryDisplay.scrollHeight;
                            }
                        }
                    }

                    const lastMsg = chatHistory[chatHistory.length - 1];
                    if (lastMsg && lastMsg.role === 'assistant') {
                        lastMsg.content = accumulatedLLMTextForDisplay; lastMsg.isStreaming = false;
                    }

                    if (mode === 'llm_tts' && accumulatedLLMTextForDisplay.trim()) {
                        if (generateButton) generateButton.textContent = "Generating TTS…";
                        await speakText(accumulatedLLMTextForDisplay);
                    }
                }

            } catch (err) {
                console.error("Generation Error:", err);
                accumulatedLLMTextForDisplay += `\n[Error: ${err.message}]`;
                if (currentAssistantMessageContentElement)
                    currentAssistantMessageContentElement.textContent = accumulatedLLMTextForDisplay;
            } finally {
                if (generateButton) {
                    generateButton.disabled    = false;
                    generateButton.textContent = "Generate";
                }
                if (currentAssistantMessageContentElement)
                    currentAssistantMessageContentElement.classList.remove('streaming-llm-content');
            }
        });
    }

    // ================================================================
    // BOOT
    // ================================================================
    initTTSVoices();
    renderChatHistory();
    if (recordButton) recordButton.innerHTML = SVG_MIC;
});