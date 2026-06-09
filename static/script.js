document.addEventListener('DOMContentLoaded', () => {
    // ================================================================
    // UI REFERENCES
    // ================================================================
    const $  = id => document.getElementById(id);
    const UI = {
        form:           $('app-form'),
        textInput:      $('text_input'),
        generateBtn:    $('generate-button'),
        audioPlayer:    $('audio-player'),
        modeSelect:     $('mode_select'),
        chatDisplay:    $('chat-history-display'),
        ttsEngine:      $('tts_engine_select'),
        ttsVoice:       $('tts_voice_dd'),
        recordBtn:      $('record-button'),
        recordStatus:   $('record-status'),
        sttEngine:      $('stt_engine_select'),
        indicLangGroup: $('indic-lang-group'),
        indicDecodeGroup:$('indic-decode-group'),
        indicLang:      $('indic_language_select'),
        indicDecode:    $('indic_decode_select'),
        llmModel:       $('llm_model_select'),
        refreshModels:  $('refresh-llm-models-btn'),
        llmMaxTokens:   $('llm_max_tokens_input'),
        convToggleBtn:  $('conversation-toggle-btn'),
        vadDot:         $('vad-dot'),
        convStatus:     $('conversation-status'),
    };

    const sliderVal = (id, fallback) => parseFloat($(`${id}_slider`)?.value ?? fallback);
    const sliderInt = (id, fallback) => parseInt($(`${id}_slider`)?.value ?? fallback);

    // ================================================================
    // STATE
    // ================================================================
    let audioCtx              = null;
    let audioSampleRate       = 8000;
    let nextAudioStart        = 0;

    let chatHistory           = [];
    let accLLMText            = '';
    let currentAssistantEl    = null;

    let mediaRecorder         = null;
    let audioChunks           = [];
    let isRecording           = false;
    let spaceDown             = false;

    let convActive            = false;
    let convWs                = null;
    let convStream            = null;
    let convAudioCtx          = null;
    let convProcessor         = null;
    let convSourceNodes       = [];
    let convNextStart         = 0;
    let convLLMText           = '';

    const WS_PROTO = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl    = path => `${WS_PROTO}//${location.host}${path}`;

    const SVG_MIC = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" style="width:1.2em;height:1.2em"><path d="M12 1a4 4 0 0 1 4 4v7a4 4 0 0 1-8 0V5a4 4 0 0 1 4-4z"/></svg>`;

    // ================================================================
    // AUDIO UTILITIES
    // ================================================================
    function b64ToArrayBuffer(b64) {
        const bin = atob(b64), buf = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
        return buf.buffer;
    }

    function arrayBufferToB64(buf) {
        const bytes = new Uint8Array(buf);
        let s = '';
        for (let i = 0; i < bytes.byteLength; i++) s += String.fromCharCode(bytes[i]);
        return btoa(s);
    }

    async function ensureAudioCtx(sampleRate) {
        if (!audioCtx) {
            audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate });
        }
        if (audioCtx.state === 'suspended') await audioCtx.resume();
    }

    function scheduleAudioChunk(float32Data, sampleRate) {
        if (!float32Data.length) return;
        const buf = audioCtx.createBuffer(1, float32Data.length, sampleRate);
        buf.copyToChannel(float32Data, 0);
        const src = audioCtx.createBufferSource();
        src.buffer = buf;
        src.connect(audioCtx.destination);
        if (nextAudioStart < audioCtx.currentTime) nextAudioStart = audioCtx.currentTime + 0.05;
        src.start(nextAudioStart);
        nextAudioStart += buf.duration;
        return src;
    }

    // ================================================================
    // STT PARAMS
    // ================================================================
    function getSttParams() {
        return {
            engine:     UI.sttEngine?.value    ?? 'indic',
            language:   UI.indicLang?.value    ?? 'bn',
            decodeMode: UI.indicDecode?.value  ?? 'ctc',
        };
    }

    function getLLMParams() {
        return {
            model:              UI.llmModel?.value ?? '',
            temperature:        sliderVal('llm_temp', 0.7),
            top_p:              sliderVal('llm_top_p', 0.9),
            repetition_penalty: sliderVal('llm_rep_penalty', 1.1),
            top_k:              sliderInt('llm_top_k', 45),
            max_tokens:         parseInt(UI.llmMaxTokens?.value ?? '-1'),
        };
    }

    function getTTSParams(text = '') {
        const engine = UI.ttsEngine?.value ?? 'orpheus';
        let voiceDescription = '';
        
        // Get voice description for Parler TTS
        if (engine === 'parler' && ttsEnginesInfo?.engines?.parler?.descriptions) {
            voiceDescription = ttsEnginesInfo.engines.parler.descriptions[UI.ttsVoice?.value] || '';
        }
        
        return {
            text,
            voice:                  UI.ttsVoice?.value ?? 'tara',
            tts_temperature:        sliderVal('tts_temp', 2.0),
            tts_top_p:              sliderVal('tts_top_p', 0.9),
            buffer_groups:          sliderInt('tts_buffer_groups', 5),
            padding_ms:             sliderInt('tts_padding_ms', 0),
            min_decode_batch_groups:sliderInt('tts_batch_groups', 7),
            tts_model:              engine,
            voice_description:      voiceDescription,
            tts_repetition_penalty: sliderVal('tts_rep_penalty', 1.2),
        };
    }

    // ================================================================
    // INDIC VISIBILITY
    // ================================================================
    function updateIndicVisibility() {
        const isIndic = getSttParams().engine === 'indic';
        if (UI.indicLangGroup)   UI.indicLangGroup.style.display   = isIndic ? '' : 'none';
        if (UI.indicDecodeGroup) UI.indicDecodeGroup.style.display  = isIndic ? '' : 'none';
    }

    async function onSttEngineChange() {
        updateIndicVisibility();
        const { engine, language, decodeMode } = getSttParams();
        if (UI.recordStatus) UI.recordStatus.textContent = `Engine: ${engine}`;
        try {
            const body = { engine, ...(engine === 'indic' && { language, decode_mode: decodeMode }) };
            const res  = await fetch('/api/stt/set_engine', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            });
            if (res.ok && UI.recordStatus) UI.recordStatus.textContent = `✓ Engine: ${engine}`;
        } catch (err) {
            console.error('STT engine switch failed:', err);
        }
    }

    if (UI.sttEngine) {
        UI.sttEngine.addEventListener('change', onSttEngineChange);
        UI.indicLang?.addEventListener('change', onSttEngineChange);
        UI.indicDecode?.addEventListener('change', onSttEngineChange);
        onSttEngineChange();
    }

    // ================================================================
    // LLM MODELS
    // ================================================================
    async function fetchLLMModels() {
        if (!UI.llmModel) return;
        UI.llmModel.innerHTML = '<option value="">— loading… —</option>';
        try {
            const data   = await fetch('/api/llm/models').then(r => r.json());
            const models = data.models ?? [];
            UI.llmModel.innerHTML = models.length
                ? models.map((m, i) => `<option value="${m}"${i === 0 ? ' selected' : ''}>${m}</option>`).join('')
                : '<option value="">— no models found —</option>';
            console.log(`Loaded ${models.length} model(s)`);
        } catch (err) {
            UI.llmModel.innerHTML = '<option value="">— fetch error —</option>';
            console.error('Failed to fetch models:', err);
        }
    }

    UI.refreshModels?.addEventListener('click', fetchLLMModels);
    // ================================================================
    // TTS VOICES
    // ================================================================
    let ttsEnginesInfo = null;
    
    async function fetchTTSEnginesInfo() {
        try {
            const res = await fetch('/api/tts/engines');
            ttsEnginesInfo = await res.json();
            updateTTSVoices();
        } catch (e) {
            console.error("Failed to fetch TTS engines:", e);
        }
    }
    
    function updateTTSVoices() {
        const engine = UI.ttsEngine?.value ?? 'orpheus';
        if (!UI.ttsVoice || !ttsEnginesInfo) return;
        
        UI.ttsVoice.innerHTML = '';
        let voices = [];
        
        if (engine === 'orpheus') {
            voices = ['tara', 'jess', 'leo', 'leah', 'dan', 'mia', 'zac', 'zoe'];
        } else if (engine === 'parler') {
            voices = ttsEnginesInfo.engines.parler.voices || [];
        }
        
        voices.forEach((v, idx) => {
            const o = document.createElement('option');
            o.value = v;
            if (engine === 'orpheus') {
                o.textContent = v.charAt(0).toUpperCase() + v.slice(1);
            } else {
                o.textContent = v.replace(/_/g, ' ').split(' ').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
            }
            if (idx === 0) o.selected = true;
            UI.ttsVoice.appendChild(o);
        });
    }
    
    if (UI.ttsEngine) {
        UI.ttsEngine.addEventListener('change', async () => {
            await fetch('/api/tts/set_engine', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ engine: UI.ttsEngine.value })
            }).catch(e => console.error("Failed to set TTS engine:", e));
            updateTTSVoices();
        });
    }
    
    fetchTTSEnginesInfo();

    // ================================================================
    // CHAT HISTORY
    // ================================================================
    function renderChatHistory() {
        if (!UI.chatDisplay) return null;
        UI.chatDisplay.innerHTML = '';
        let lastAssistantEl = null;
        chatHistory.forEach(msg => {
            const wrap    = document.createElement('div');
            wrap.className = `chat-message ${msg.role === 'user' ? 'user-message' : 'assistant-message'}`;
            const inner   = document.createElement('div');
            inner.className = 'message-content-wrapper';
            const strong  = document.createElement('strong');
            strong.textContent = msg.role === 'user' ? 'You:' : 'Assistant:';
            const content = document.createElement('div');
            if (msg.role === 'assistant' && msg.isStreaming) content.className = 'streaming-llm-content';
            content.textContent = msg.content;
            inner.append(strong, content);
            wrap.appendChild(inner);
            UI.chatDisplay.appendChild(wrap);
            if (msg.role === 'assistant') lastAssistantEl = content;
        });
        UI.chatDisplay.scrollTop = UI.chatDisplay.scrollHeight;
        return lastAssistantEl;
    }

    function addUserMessage(text) {
        chatHistory.push({ role: 'user', content: text });
        renderChatHistory();
    }

    function addAssistantMessage(text, isStreaming = false) {
        const last = chatHistory.at(-1);
        if (isStreaming && last?.role === 'assistant' && last.isStreaming) {
            last.content = text;
        } else {
            chatHistory.push({ role: 'assistant', content: text, isStreaming });
        }
        return renderChatHistory();
    }

    function scrollChat() {
        if (UI.chatDisplay) UI.chatDisplay.scrollTop = UI.chatDisplay.scrollHeight;
    }

    // ================================================================
    // RECORDING
    // ================================================================
    async function startRecording() {
        if (!navigator.mediaDevices?.getUserMedia) { alert("Browser doesn't support recording."); return; }
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            audioChunks   = [];
            mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
            mediaRecorder.onstop = async () => {
                const blob = new Blob(audioChunks, { type: 'audio/webm' });
                audioChunks = [];
                stream.getTracks().forEach(t => t.stop());
                await sendAudioForSTT(blob);
            };
            mediaRecorder.start();
            isRecording = true;
            if (UI.recordBtn) { UI.recordBtn.innerHTML = '🛑'; UI.recordBtn.classList.add('recording'); }
            if (UI.recordStatus) UI.recordStatus.textContent = 'Recording…';
        } catch (err) {
            console.error('Mic error:', err);
            alert('Microphone permission denied.');
            isRecording = false;
        }
    }

    function stopRecording() {
        if (mediaRecorder?.state === 'recording') {
            mediaRecorder.stop();
            isRecording = false;
            if (UI.recordBtn) { UI.recordBtn.innerHTML = SVG_MIC; UI.recordBtn.classList.remove('recording'); }
            if (UI.recordStatus) UI.recordStatus.textContent = 'Processing…';
        }
    }

    // ================================================================
    // STT via WebSocket
    // ================================================================
    async function sendAudioForSTT(blob) {
        const { engine, language, decodeMode } = getSttParams();
        if (UI.recordStatus) UI.recordStatus.textContent = `Transcribing (${engine})…`;
        try {
            const result = await new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onload = e => {
                    const ws = new WebSocket(wsUrl('/ws/stt'));
                    ws.onopen = () => ws.send(JSON.stringify({
                        engine, language, decode_mode: decodeMode,
                        audio: arrayBufferToB64(e.target.result),
                    }));
                    ws.onmessage = ev => {
                        const msg = JSON.parse(ev.data);
                        if (msg.type === 'stt_result') { resolve(msg); ws.close(); }
                        else if (msg.type === 'error')  { reject(new Error(msg.message)); ws.close(); }
                    };
                    ws.onerror = () => reject(new Error('STT WebSocket error'));
                };
                reader.onerror = () => reject(new Error('Failed to read blob'));
                reader.readAsArrayBuffer(blob);
            });
            if (UI.textInput) UI.textInput.value = result.text ?? '';
            if (UI.recordStatus) UI.recordStatus.textContent = `✓ ${engine}`;
        } catch (err) {
            console.error('STT error:', err);
            alert(`STT Error: ${err.message}`);
            if (UI.recordStatus) UI.recordStatus.textContent = 'STT failed!';
        }
    }

    UI.recordBtn?.addEventListener('click', () => isRecording ? stopRecording() : startRecording());

    document.addEventListener('keydown', async e => {
        if (e.code !== 'Space' || document.activeElement?.tagName.toLowerCase() === 'textarea') return;
        e.preventDefault();
        if (!isRecording && !spaceDown) { spaceDown = true; await startRecording(); }
    });
    document.addEventListener('keyup', e => {
        if (e.code !== 'Space' || !spaceDown) return;
        spaceDown = false;
        if (isRecording) stopRecording();
    });

    // ================================================================
    // TTS via WebSocket — returns a Promise that resolves when done
    // ================================================================
    function streamTTSviaWS(ttsReq) {
        return new Promise((resolve, reject) => {
            const ws = new WebSocket(wsUrl('/ws/tts'));
            ws.onopen = () => ws.send(JSON.stringify(ttsReq));
            ws.onmessage = async ev => {
                try {
                    const msg = JSON.parse(ev.data);
                    if (msg.type === 'tts_started') {
                        audioSampleRate = msg.sample_rate ?? 8000;
                        await ensureAudioCtx(audioSampleRate);
                        nextAudioStart = audioCtx.currentTime + 0.2;
                    } else if (msg.type === 'tts_audio_chunk') {
                        scheduleAudioChunk(new Float32Array(b64ToArrayBuffer(msg.audio)), audioSampleRate);
                    } else if (msg.type === 'tts_done') {
                        ws.close(); resolve();
                    } else if (msg.type === 'error') {
                        ws.close(); reject(new Error(msg.message));
                    }
                } catch (err) { reject(err); }
            };
            ws.onerror  = err => reject(err);
            ws.onclose  = () => {};
        });
    }

    // ================================================================
    // LLM STREAM → text
    // ================================================================
    async function streamLLM(userText) {
        const historyForAPI = chatHistory
            .filter(m => !m.isStreaming && m.role !== 'system')
            .slice(0, -1)
            .map(({ role, content }) => ({ role, content }));

        const res = await fetch('/api/llm/chat/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt: userText, history: historyForAPI, ...getLLMParams() }),
        });
        if (!res.ok) throw new Error(`Server ${res.status}: ${res.statusText}`);

        const reader  = res.body.getReader();
        const decoder = new TextDecoder();
        let   result  = '';
        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            result += decoder.decode(value, { stream: true });
            accLLMText = result;
            if (currentAssistantEl) { currentAssistantEl.textContent = result; scrollChat(); }
        }
        return result;
    }

    // ================================================================
    // FORM SUBMIT
    // ================================================================
    UI.form?.addEventListener('submit', async e => {
        e.preventDefault();
        const userText = UI.textInput.value.trim();
        if (!userText) { alert('Enter some text.'); return; }

        addUserMessage(userText);
        UI.textInput.value = '';
        const mode = UI.modeSelect?.value ?? 'llm_only';

        const setBtn = (label, disabled) => {
            if (UI.generateBtn) { UI.generateBtn.disabled = disabled; UI.generateBtn.textContent = label; }
        };
        setBtn('Generating…', true);

        try {
            if (mode === 'tts_only') {
                await streamTTSviaWS(getTTSParams(userText));
                return;
            }

            // LLM (+ optional TTS)
            currentAssistantEl = addAssistantMessage('', true);
            accLLMText = '';

            const finalText = await streamLLM(userText);

            const last = chatHistory.at(-1);
            if (last?.role === 'assistant') { last.content = finalText; last.isStreaming = false; }

            if (mode === 'llm_tts' && finalText.trim()) {
                setBtn('Generating TTS…', true);
                await streamTTSviaWS(getTTSParams(finalText));
            }
        } catch (err) {
            console.error('Error:', err);
            accLLMText += `\n[Error: ${err.message}]`;
            if (currentAssistantEl) currentAssistantEl.textContent = accLLMText;
        } finally {
            setBtn('Generate', false);
            currentAssistantEl?.classList.remove('streaming-llm-content');
            currentAssistantEl = null;
        }
    });

    // ================================================================
    // CONVERSATION MODE
    // ================================================================
    const setConvStatus = (text, active = false) => {
        if (UI.convStatus) { UI.convStatus.textContent = text; UI.convStatus.classList.toggle('active-status', active); }
    };
    const setVadDot = speaking => UI.vadDot?.classList.toggle('speaking', speaking);

    function cancelConvTTS() {
        convSourceNodes.forEach(n => { try { n.stop(); } catch (_) {} });
        convSourceNodes  = [];
        convNextStart    = 0;
    }

    function scheduleConvChunk(float32Data) {
        if (!audioCtx || !float32Data.length) return;
        const buf = audioCtx.createBuffer(1, float32Data.length, audioSampleRate);
        buf.copyToChannel(float32Data, 0);
        const src = audioCtx.createBufferSource();
        src.buffer = buf;
        src.connect(audioCtx.destination);
        if (convNextStart < audioCtx.currentTime) convNextStart = audioCtx.currentTime + 0.05;
        src.start(convNextStart);
        convNextStart += buf.duration;
        convSourceNodes.push(src);
    }

    function handleConvMessage(msg) {
        switch (msg.type) {
            case 'vad_state':   setVadDot(msg.is_speaking); break;
            case 'listening':   setConvStatus('Listening…', true); setVadDot(false); break;
            case 'processing': {
                const labels = { stt: 'Transcribing…', llm: 'Thinking…', tts: 'Speaking…' };
                setConvStatus(labels[msg.stage] ?? 'Processing…', true);
                break;
            }
            case 'transcript':
                addUserMessage(msg.text);
                setConvStatus('Thinking…', true);
                break;
            case 'llm_chunk':
                if (!convLLMText) {
                    currentAssistantEl = addAssistantMessage('', true);
                    convLLMText = '';
                }
                convLLMText += msg.content;
                if (currentAssistantEl) { currentAssistantEl.textContent = convLLMText; scrollChat(); }
                break;
            case 'llm_done': {
                const last = chatHistory.at(-1);
                if (last?.role === 'assistant') { last.content = convLLMText; last.isStreaming = false; }
                currentAssistantEl?.classList.remove('streaming-llm-content');
                currentAssistantEl = null;
                convLLMText = '';
                setConvStatus('Speaking…', true);
                break;
            }
            case 'tts_started':
                audioSampleRate = msg.sample_rate ?? 8000;
                ensureAudioCtx(audioSampleRate);
                convNextStart   = (audioCtx?.currentTime ?? 0) + 0.15;
                convSourceNodes = [];
                setConvStatus('Speaking…', true);
                break;
            case 'tts_audio_chunk':
                scheduleConvChunk(new Float32Array(b64ToArrayBuffer(msg.audio)));
                break;
            case 'interrupt':
                cancelConvTTS();
                convLLMText = ''; currentAssistantEl = null;
                setConvStatus('Listening…', true);
                break;
            case 'error':
                console.error('[Conversation] Server error:', msg.message);
                setConvStatus(`Error: ${msg.message}`, false);
                break;
        }
    }

    function startMicStreaming() {
        if (!convAudioCtx || !convStream || !convWs) return;
        const source     = convAudioCtx.createMediaStreamSource(convStream);
        const actualRate = convAudioCtx.sampleRate;
        convProcessor    = convAudioCtx.createScriptProcessor(4096, 1, 1);

        convProcessor.onaudioprocess = e => {
            if (!convActive || convWs?.readyState !== WebSocket.OPEN) return;
            const f32 = e.inputBuffer.getChannelData(0);
            const i16 = new Int16Array(f32.length);
            for (let i = 0; i < f32.length; i++) {
                const s = Math.max(-1, Math.min(1, f32[i]));
                i16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
            }
            convWs.send(JSON.stringify({ type: 'audio_chunk', audio: arrayBufferToB64(i16.buffer) }));
        };
        source.connect(convProcessor);
        convProcessor.connect(convAudioCtx.destination);
        console.log(`[Conversation] Mic streaming at ${actualRate}Hz`);
    }

    async function startConversation() {
        if (convActive) return;
        try {
            convStream = await navigator.mediaDevices.getUserMedia({
                audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
            });
        } catch (err) {
            alert('Microphone permission is required for conversation mode.');
            return;
        }

        convAudioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
        convWs       = new WebSocket(wsUrl('/ws/conversation'));

        convWs.onopen = () => {
            const { engine, language, decodeMode } = getSttParams();
            convWs.send(JSON.stringify({
                type: 'config',
                engine, language, decode_mode: decodeMode,
                ...getLLMParams(),
                ...getTTSParams(),
                sample_rate: convAudioCtx.sampleRate,
            }));
            startMicStreaming();
        };
        convWs.onmessage = ev => { try { handleConvMessage(JSON.parse(ev.data)); } catch (err) { console.error(err); } };
        convWs.onerror   = err => console.error('[Conversation] WS error:', err);
        convWs.onclose   = () => { if (convActive) stopConversation(); };

        convActive = true;
        if (UI.convToggleBtn) { UI.convToggleBtn.textContent = '⏹ End Conversation'; UI.convToggleBtn.classList.add('active'); }
        setConvStatus('Listening…', true);
        setVadDot(false);
    }

    function stopConversation() {
        convActive = false;
        cancelConvTTS();
        try { convWs?.send(JSON.stringify({ type: 'stop' })); convWs?.close(); } catch (_) {}
        convWs = null;
        try { convProcessor?.disconnect(); } catch (_) {}
        convProcessor = null;
        try { convAudioCtx?.close(); } catch (_) {}
        convAudioCtx = null;
        convStream?.getTracks().forEach(t => t.stop());
        convStream = null;
        convLLMText = ''; currentAssistantEl = null;
        if (UI.convToggleBtn) { UI.convToggleBtn.textContent = '🎙️ Start Conversation'; UI.convToggleBtn.classList.remove('active'); }
        setConvStatus('Inactive', false);
        setVadDot(false);
    }

    UI.convToggleBtn?.addEventListener('click', () => convActive ? stopConversation() : startConversation());

    // ================================================================
    // BOOT
    // ================================================================
    renderChatHistory();
    if (UI.recordBtn) UI.recordBtn.innerHTML = SVG_MIC;
});