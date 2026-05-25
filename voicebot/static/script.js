document.addEventListener('DOMContentLoaded', () => {
    console.log("DOM loaded");

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

    // ================================================================
    // STATIC DATA
    // ================================================================
    const ttsVoices = ["tara", "jess", "leo", "leah", "dan", "mia", "zac", "zoe"];

    // ================================================================
    // AUDIO STATE
    // ================================================================
    let audioContext               = null;
    let currentAudioSampleRate     = 8000;  // Default 8kHz for landline, will be updated by server
    let audioBufferQueue           = [];
    let isPlayingAudio             = false;
    let nextAudioStartTime         = 0;
    let currentAudioBufferDuration = 0;
    let clientMinBufferDuration    = 0.1;
    let fetchStreamReaderTTS       = null;
    let currentTTSPlaybackResolver = null;
    let useWebSocketForAudio       = true;  // Enable WebSocket audio streaming by default

    // ================================================================
    // CHAT / LLM STATE
    // ================================================================
    let chatHistory                           = [];
    let currentLLMStreamController            = null;
    let llmDisplayQueue                       = [];
    let isDisplayingFromLLMQueue              = false;
    let initialTextDisplayDelayMs             = 950;
    let subsequentChunkDisplayIntervalMs      = 40;
    let firstLLMChunkReceived                 = false;
    let llmStreamCompleted                    = false;
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
    // INIT TTS VOICES
    // ================================================================
    function initTTSVoices() {
        if (!ttsVoiceSelect) return;

        ttsVoices.forEach(v => {
            const o = document.createElement('option');
            o.value = v;
            o.textContent = v.charAt(0).toUpperCase() + v.slice(1);

            if (v === "tara") {
                o.selected = true;
            }

            ttsVoiceSelect.appendChild(o);
        });
    }

    // ================================================================
    // STT PARAMS
    // ================================================================
    function getSttParams() {
        const engine = sttEngineSelect
            ? sttEngineSelect.value
            : 'indic';

        const language = indicLangSelect
            ? indicLangSelect.value
            : 'bn';

        const decodeMode = indicDecodeSelect
            ? indicDecodeSelect.value
            : 'ctc';

        console.log("[STT PARAMS]", {
            engine,
            language,
            decodeMode
        });

        return {
            engine,
            language,
            decodeMode
        };
    }

    // ================================================================
    // INDIC VISIBILITY
    // ================================================================
    function updateIndicVisibility() {
        const { engine } = getSttParams();

        const isIndic = engine === 'indic';

        if (indicLangGroup) {
            indicLangGroup.style.display = isIndic ? '' : 'none';
        }

        if (indicDecodeGroup) {
            indicDecodeGroup.style.display = isIndic ? '' : 'none';
        }
    }

    // ================================================================
    // STT ENGINE CHANGE
    // ================================================================
    async function onSttEngineChange() {

        updateIndicVisibility();

        const { engine, language, decodeMode } = getSttParams();

        console.log("STT engine changed:", engine);

        if (recordStatus) {
            recordStatus.textContent = `Engine: ${engine}`;
        }

        try {

            const body = {
                engine
            };

            if (engine === 'indic') {
                body.language = language;
                body.decode_mode = decodeMode;
            }

            console.log("Sending STT engine update:", body);

            const res = await fetch('/api/stt/set_engine', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(body)
            });

            const data = await res.json().catch(() => ({}));

            console.log("STT backend response:", data);

            if (res.ok) {
                if (recordStatus) {
                    recordStatus.textContent = `✓ Engine: ${engine}`;
                }
            } else {
                console.error("STT backend error:", data);
            }

        } catch (err) {
            console.error("STT engine switch failed:", err);
        }
    }

    // ================================================================
    // INITIALIZE STT LISTENERS
    // ================================================================
    if (sttEngineSelect) {

        sttEngineSelect.addEventListener(
            'change',
            onSttEngineChange
        );

        if (indicLangSelect) {
            indicLangSelect.addEventListener(
                'change',
                onSttEngineChange
            );
        }

        if (indicDecodeSelect) {
            indicDecodeSelect.addEventListener(
                'change',
                onSttEngineChange
            );
        }

        // IMPORTANT
        onSttEngineChange();
    }

    // ================================================================
    // FETCH LLM MODELS
    // ================================================================
    async function fetchLLMModels() {

        if (!llmModelSelect) return;

        llmModelSelect.innerHTML =
            '<option value="">— loading… —</option>';

        try {

            const res = await fetch('/api/llm/models');

            const data = await res.json();

            const models = data.models || [];

            llmModelSelect.innerHTML = '';

            if (models.length === 0) {
                llmModelSelect.innerHTML =
                    '<option value="">— no models found —</option>';
                return;
            }

            models.forEach((m, i) => {

                const o = document.createElement('option');

                o.value = m;
                o.textContent = m;

                if (i === 0) {
                    o.selected = true;
                }

                llmModelSelect.appendChild(o);
            });

            console.log(`Loaded ${models.length} model(s)`);

        } catch (err) {

            console.error("Failed to fetch models:", err);

            llmModelSelect.innerHTML =
                '<option value="">— fetch error —</option>';
        }
    }

    if (refreshLlmModelsBtn) {
        refreshLlmModelsBtn.addEventListener(
            'click',
            fetchLLMModels
        );
    }

    fetchLLMModels();

    // ================================================================
    // CHAT HISTORY
    // ================================================================
    function renderChatHistory() {

        if (!chatHistoryDisplay) return null;

        chatHistoryDisplay.innerHTML = '';

        let lastAssistantDiv = null;

        chatHistory.forEach(msg => {

            const wrap = document.createElement('div');

            wrap.classList.add(
                'chat-message',
                msg.role === 'user'
                    ? 'user-message'
                    : 'assistant-message'
            );

            const inner = document.createElement('div');

            inner.classList.add(
                'message-content-wrapper'
            );

            const strong = document.createElement('strong');

            strong.textContent =
                msg.role === 'user'
                    ? 'You:'
                    : 'Assistant:';

            inner.appendChild(strong);

            const content = document.createElement('div');

            if (
                msg.role === 'assistant' &&
                msg.isStreaming
            ) {
                content.classList.add(
                    'streaming-llm-content'
                );
            }

            content.textContent = msg.content;

            inner.appendChild(content);

            wrap.appendChild(inner);

            chatHistoryDisplay.appendChild(wrap);

            if (msg.role === 'assistant') {
                lastAssistantDiv = content;
            }
        });

        chatHistoryDisplay.scrollTop =
            chatHistoryDisplay.scrollHeight;

        return lastAssistantDiv;
    }

    function addUserMessage(text) {
        chatHistory.push({
            role: 'user',
            content: text
        });

        renderChatHistory();
    }

    function addAssistantMessage(
        text,
        isStreaming = false
    ) {

        const last =
            chatHistory.length > 0
                ? chatHistory[chatHistory.length - 1]
                : null;

        if (
            isStreaming &&
            last &&
            last.role === 'assistant' &&
            last.isStreaming
        ) {

            last.content = text;

        } else {

            chatHistory.push({
                role: 'assistant',
                content: text,
                isStreaming
            });
        }

        return renderChatHistory();
    }

    // ================================================================
    // RECORDING
    // ================================================================
    const SVG_MIC = `
    <svg xmlns="http://www.w3.org/2000/svg"
         viewBox="0 0 24 24"
         fill="currentColor"
         style="width:1.2em;height:1.2em;">
        <path d="M12 1a4 4 0 0 1 4 4v7a4 4 0 0 1-8 0V5a4 4 0 0 1 4-4z"/>
    </svg>`;

    async function startRecording() {

        if (!navigator.mediaDevices?.getUserMedia) {
            alert("Browser doesn't support recording.");
            return;
        }

        try {

            const stream =
                await navigator.mediaDevices.getUserMedia({
                    audio: true
                });

            mediaRecorder = new MediaRecorder(stream);

            audioChunks = [];

            mediaRecorder.ondataavailable = e => {
                audioChunks.push(e.data);
            };

            mediaRecorder.onstop = async () => {

                const blob = new Blob(audioChunks, {
                    type: 'audio/webm'
                });

                audioChunks = [];

                stream.getTracks().forEach(t => t.stop());

                await sendAudioForTranscription(
                    blob,
                    'recording.webm'
                );
            };

            mediaRecorder.start();

            isRecording = true;

            if (recordButton) {
                recordButton.innerHTML = '🛑';
                recordButton.classList.add('recording');
            }

            if (recordStatus) {
                recordStatus.textContent = 'Recording…';
            }

        } catch (err) {

            console.error("Mic error:", err);

            alert("Microphone permission denied.");

            isRecording = false;
        }
    }

    function stopRecording() {

        if (
            mediaRecorder &&
            mediaRecorder.state === 'recording'
        ) {

            mediaRecorder.stop();

            isRecording = false;

            if (recordButton) {
                recordButton.innerHTML = SVG_MIC;
                recordButton.classList.remove('recording');
            }

            if (recordStatus) {
                recordStatus.textContent = 'Processing…';
            }
        }
    }

    // ================================================================
    // SEND AUDIO TO STT
    // ================================================================
    async function sendAudioForTranscription(
        blob,
        fileName
    ) {

        const {
            engine,
            language,
            decodeMode
        } = getSttParams();

        console.log("Sending transcription request:", {
            engine,
            language,
            decodeMode
        });

        try {

            if (recordStatus) {
                recordStatus.textContent =
                    `Transcribing (${engine})…`;
            }

            let result;

            // Use WebSocket if enabled, otherwise use HTTP
            if (useWebSocketForAudio) {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}/ws/stt`;
                result = await sendAudioViaWebSocket(blob, wsUrl);
            } else {
                // Original HTTP-based approach
                let url = '/api/stt/transcribe';

                const params = new URLSearchParams();

                params.set('engine', engine);

                if (engine === 'indic') {
                    params.set('language', language);
                    params.set('decode_mode', decodeMode);
                }

                url += '?' + params.toString();

                console.log("FINAL STT URL:", url);

                const form = new FormData();

                form.append(
                    'audio_file',
                    blob,
                    fileName
                );

                const res = await fetch(url, {
                    method: 'POST',
                    body: form
                });

                result = await res.json();

                if (!res.ok) {
                    throw new Error(
                        result.detail || 'STT failed'
                    );
                }
            }

            console.log("STT RESPONSE:", result);

            if (textInput) {
                textInput.value = result.text || '';
            }

            if (recordStatus) {
                recordStatus.textContent =
                    `✓ ${engine}`;
            }

        } catch (err) {

            console.error("STT ERROR:", err);

            alert(`STT Error: ${err.message}`);

            if (recordStatus) {
                recordStatus.textContent =
                    'STT failed!';
            }
        }
    }

    // ================================================================
    // RECORD BUTTON
    // ================================================================
    if (recordButton) {

        recordButton.addEventListener(
            'click',
            () => {

                if (isRecording) {
                    stopRecording();
                } else {
                    startRecording();
                }
            }
        );
    }

    // ================================================================
    // PUSH TO TALK
    // ================================================================
    document.addEventListener('keydown', async e => {

        if (e.code !== 'Space') return;

        const el = document.activeElement;

        if (
            el &&
            el.tagName.toLowerCase() === 'textarea'
        ) {
            return;
        }

        e.preventDefault();

        if (!isRecording && !spaceBarIsDown) {

            spaceBarIsDown = true;

            isPushToTalkActive = true;

            await startRecording();
        }
    });

    document.addEventListener('keyup', async e => {

        if (e.code !== 'Space') return;

        if (spaceBarIsDown) {

            spaceBarIsDown = false;

            if (
                isRecording &&
                isPushToTalkActive
            ) {

                stopRecording();
            }

            isPushToTalkActive = false;
        }
    });

    // ================================================================
    // TTS STREAMING HANDLER (Web Audio API) - HTTP
    // ================================================================
    async function streamTTSAudio(response) {
        // Get sample rate from response headers
        const headerSampleRate = parseInt(response.headers.get('X-Sample-Rate') || '8000');
        currentAudioSampleRate = headerSampleRate;
        console.log(`TTS Sample Rate from server: ${currentAudioSampleRate}Hz`);
        
        if (!audioContext) {
            audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: currentAudioSampleRate });
        }
        if (audioContext.state === 'suspended') {
            await audioContext.resume();
        }
        
        const reader = response.body.getReader();
        let nextStartTime = audioContext.currentTime + 0.2; // Start with a small buffer
        let leftoverBytes = new Uint8Array(0);

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            // Combine leftover from previous network chunk
            const totalLength = leftoverBytes.length + value.length;
            const combined = new Uint8Array(totalLength);
            combined.set(leftoverBytes, 0);
            combined.set(value, leftoverBytes.length);
            
            // Number of complete floats (4 bytes each)
            const numFloats = Math.floor(combined.length / 4);
            const completeBytesLength = numFloats * 4;
            
            // Save the remainder
            leftoverBytes = combined.slice(completeBytesLength);
            
            if (numFloats > 0) {
                // Ensure the buffer is copied out so we can construct a valid Float32Array
                const slicedBuffer = combined.buffer.slice(combined.byteOffset, combined.byteOffset + completeBytesLength);
                const float32Data = new Float32Array(slicedBuffer);
                
                const audioBuffer = audioContext.createBuffer(1, numFloats, currentAudioSampleRate);
                audioBuffer.copyToChannel(float32Data, 0);
                
                const source = audioContext.createBufferSource();
                source.buffer = audioBuffer;
                source.connect(audioContext.destination);
                
                // If we fell behind, catch up to current time
                if (nextStartTime < audioContext.currentTime) {
                    nextStartTime = audioContext.currentTime + 0.05;
                }
                
                source.start(nextStartTime);
                nextStartTime += audioBuffer.duration;
            }
        }
    }

    // ================================================================
    // BASE64 UTILITIES
    // ================================================================
    function base64ToArrayBuffer(base64) {
        const binaryString = atob(base64);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
            bytes[i] = binaryString.charCodeAt(i);
        }
        return bytes.buffer;
    }

    function arrayBufferToBase64(buffer) {
        const bytes = new Uint8Array(buffer);
        let binary = '';
        for (let i = 0; i < bytes.byteLength; i++) {
            binary += String.fromCharCode(bytes[i]);
        }
        return btoa(binary);
    }

    // ================================================================
    // TTS WEBSOCKET HANDLER - Base64 Audio Streaming
    // ================================================================
    function streamTTSAudioWebSocket(wsUrl, ttsRequest = null) {
        return new Promise((resolve, reject) => {
            const ws = new WebSocket(wsUrl);
            ws.onopen = () => {
                console.log("TTS WebSocket connected");
                if (ttsRequest) {
                    ws.send(JSON.stringify(ttsRequest));
                }
            };
            ws.onmessage = async (event) => {
                try {
                    const message = JSON.parse(event.data);
                    console.log("[TTS WS]", message.type, message);

                    if (message.type === "tts_started") {
                        // Get sample rate from server message or use default
                        currentAudioSampleRate = message.sample_rate || 8000;
                        console.log(`TTS started with sample rate: ${currentAudioSampleRate}Hz`);
                        
                        if (!audioContext) {
                            audioContext = new (window.AudioContext || window.webkitAudioContext)({ 
                                sampleRate: currentAudioSampleRate
                            });
                        }
                        if (audioContext.state === 'suspended') {
                            await audioContext.resume();
                        }
                        isPlayingAudio = true;
                        nextAudioStartTime = audioContext.currentTime + 0.2;
                    } 
                    else if (message.type === "tts_audio_chunk") {
                        // Decode base64 audio
                        const audioBase64 = message.audio;
                        const arrayBuffer = base64ToArrayBuffer(audioBase64);
                        const float32Data = new Float32Array(arrayBuffer);

                        if (float32Data.length > 0) {
                            const audioBuffer = audioContext.createBuffer(1, float32Data.length, currentAudioSampleRate);
                            audioBuffer.copyToChannel(float32Data, 0);

                            const source = audioContext.createBufferSource();
                            source.buffer = audioBuffer;
                            source.connect(audioContext.destination);

                            // If we fell behind, catch up to current time
                            if (nextAudioStartTime < audioContext.currentTime) {
                                nextAudioStartTime = audioContext.currentTime + 0.05;
                            }

                            source.start(nextAudioStartTime);
                            nextAudioStartTime += audioBuffer.duration;
                        }
                    } 
                    else if (message.type === "tts_done") {
                        console.log("TTS streaming completed");
                        isPlayingAudio = false;
                        ws.close();
                        resolve();
                    } 
                    else if (message.type === "error") {
                        console.error("TTS WebSocket error:", message.message);
                        ws.close();
                        reject(new Error(message.message));
                    }
                } catch (err) {
                    console.error("Error processing TTS message:", err);
                    reject(err);
                }
            };
            ws.onerror = (error) => {
                console.error("TTS WebSocket error:", error);
                reject(error);
            };
            ws.onclose = () => {
                console.log("TTS WebSocket closed");
                isPlayingAudio = false;
            };
        });
    }

    // ================================================================
    // STT WEBSOCKET HANDLER - Base64 Audio Upload
    // ================================================================
    async function sendAudioViaWebSocket(blob, wsUrl) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                const arrayBuffer = e.target.result;
                const uint8Array = new Uint8Array(arrayBuffer);
                const base64Audio = arrayBufferToBase64(uint8Array);

                const ws = new WebSocket(wsUrl);
                ws.onopen = () => {
                    console.log("STT WebSocket connected, sending audio");
                    
                    const {
                        engine,
                        language,
                        decodeMode
                    } = getSttParams();

                    const payload = {
                        engine: engine,
                        language: language,
                        decode_mode: decodeMode,
                        audio: base64Audio
                    };

                    ws.send(JSON.stringify(payload));
                };
                ws.onmessage = (event) => {
                    try {
                        const message = JSON.parse(event.data);
                        console.log("[STT WS]", message.type, message);

                        if (message.type === "stt_result") {
                            resolve({
                                text: message.text,
                                engine: message.engine,
                                language: message.language
                            });
                            ws.close();
                        } 
                        else if (message.type === "error") {
                            console.error("STT error:", message.message);
                            reject(new Error(message.message));
                            ws.close();
                        }
                    } catch (err) {
                        console.error("Error processing STT message:", err);
                        reject(err);
                    }
                };
                ws.onerror = (error) => {
                    console.error("STT WebSocket error:", error);
                    reject(error);
                };
                ws.onclose = () => {
                    console.log("STT WebSocket closed");
                };
            };
            reader.onerror = () => {
                reject(new Error("Failed to read blob as ArrayBuffer"));
            };
            reader.readAsArrayBuffer(blob);
        });
    }

    // ================================================================
    // FORM SUBMIT
    // ================================================================
    if (appForm) {
        appForm.addEventListener(
            'submit',
            async e => {
                e.preventDefault();

                const userText = textInput.value.trim();

                if (!userText) {
                    alert("Enter some text.");
                    return;
                }

                // 1. Add User Message
                addUserMessage(userText);
                textInput.value = '';

                // Read mode
                const mode = modeSelect ? modeSelect.value : 'llm_only';

                // Setup UI for generation
                if (generateButton) {
                    generateButton.disabled = true;
                    generateButton.textContent = "Generating...";
                }

                // ================================================================
                // TTS ONLY MODE
                // ================================================================
                if (mode === 'tts_only') {
                    try {
                        const voice = ttsVoiceSelect ? ttsVoiceSelect.value : "tara";
                        const ttsTemp = parseFloat(document.getElementById('tts_temp_slider')?.value || "2.0");
                        const ttsTopP = parseFloat(document.getElementById('tts_top_p_slider')?.value || "0.9");
                        const bufferGroups = parseInt(document.getElementById('tts_buffer_groups_slider')?.value || "5");
                        const paddingMs = parseInt(document.getElementById('tts_padding_ms_slider')?.value || "0");
                        const batchGroups = parseInt(document.getElementById('tts_batch_groups_slider')?.value || "7");

                        const ttsReq = {
                            text: userText,
                            voice: voice,
                            tts_temperature: ttsTemp,
                            tts_top_p: ttsTopP,
                            buffer_groups: bufferGroups,
                            padding_ms: paddingMs,
                            min_decode_batch_groups: batchGroups
                        };

                        if (useWebSocketForAudio) {
                            // Use WebSocket streaming
                            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                            const wsUrl = `${protocol}//${window.location.host}/ws/tts`;
                            
                            const ws = new WebSocket(wsUrl);
                            ws.onopen = () => {
                                ws.send(JSON.stringify(ttsReq));
                            };
                            ws.onmessage = async (event) => {
                                const message = JSON.parse(event.data);
                                if (message.type === "tts_done") {
                                    ws.close();
                                }
                            };
                            ws.onerror = (error) => {
                                throw new Error("WebSocket connection failed");
                            };

                            await streamTTSAudioWebSocket(wsUrl);
                        } else {
                            // Use HTTP streaming
                            const response = await fetch('/api/tts/stream', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify(ttsReq)
                            });

                            if (!response.ok) throw new Error("TTS Request failed");

                            // Play audio dynamically as chunks arrive
                            await streamTTSAudio(response);
                        }

                    } catch (err) {
                        console.error("TTS Error:", err);
                        alert("TTS Generation failed.");
                    } finally {
                        if (generateButton) {
                            generateButton.disabled = false;
                            generateButton.textContent = "Generate";
                        }
                    }
                    return;
                }

                // ================================================================
                // LLM (+ Optional TTS) MODE
                // ================================================================
                
                const assistantContentElement = addAssistantMessage("", true);
                currentAssistantMessageContentElement = assistantContentElement;
                accumulatedLLMTextForDisplay = "";

                try {
                    // Prepare LLM parameters
                    const temp = parseFloat(document.getElementById('llm_temp_slider')?.value || "0.7");
                    const topP = parseFloat(document.getElementById('llm_top_p_slider')?.value || "0.9");
                    const repPenalty = parseFloat(document.getElementById('llm_rep_penalty_slider')?.value || "1.1");
                    const topK = parseInt(document.getElementById('llm_top_k_slider')?.value || "45");
                    const maxTokens = parseInt(document.getElementById('llm_max_tokens_input')?.value || "-1");
                    const model = llmModelSelect ? llmModelSelect.value : null;

                    // History for API (Exclude the user prompt we just added to the UI array)
                    // We must map it to only include 'role' and 'content' because the backend Pydantic
                    // model (Dict[str, str]) will reject booleans like 'isStreaming: false'.
                    const historyForAPI = chatHistory
                        .filter(msg => !msg.isStreaming && msg.role !== 'system')
                        .slice(0, -1)
                        .map(msg => ({ role: msg.role, content: msg.content }));

                    const requestBody = {
                        prompt: userText,
                        history: historyForAPI,
                        model: model,
                        temperature: temp,
                        top_p: topP,
                        max_tokens: maxTokens,
                        repetition_penalty: repPenalty,
                        top_k: topK
                    };

                    const response = await fetch('/api/llm/chat/stream', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(requestBody)
                    });

                    if (!response.ok) {
                        throw new Error(`Server returned ${response.status}: ${response.statusText}`);
                    }

                    const reader = response.body.getReader();
                    const decoder = new TextDecoder('utf-8');
                    let done = false;

                    // Stream LLM chunks to the UI
                    while (!done) {
                        const { value, done: readerDone } = await reader.read();
                        done = readerDone;
                        
                        if (value) {
                            const chunk = decoder.decode(value, { stream: true });
                            accumulatedLLMTextForDisplay += chunk;
                            
                            if (currentAssistantMessageContentElement) {
                                currentAssistantMessageContentElement.textContent = accumulatedLLMTextForDisplay;
                                if (chatHistoryDisplay) {
                                    chatHistoryDisplay.scrollTop = chatHistoryDisplay.scrollHeight;
                                }
                            }
                        }
                    }

                    // Update final chat history state
                    const lastMsg = chatHistory[chatHistory.length - 1];
                    if (lastMsg && lastMsg.role === 'assistant') {
                        lastMsg.content = accumulatedLLMTextForDisplay;
                        lastMsg.isStreaming = false;
                    }

                    // If LLM+TTS mode, send the completed text to the TTS endpoint
                    if (mode === 'llm_tts' && accumulatedLLMTextForDisplay.trim() !== '') {
                        if (generateButton) generateButton.textContent = "Generating TTS...";
                        
                        const voice = ttsVoiceSelect ? ttsVoiceSelect.value : "tara";
                        const ttsTemp = parseFloat(document.getElementById('tts_temp_slider')?.value || "2.0");
                        const ttsTopP = parseFloat(document.getElementById('tts_top_p_slider')?.value || "0.9");
                        const bufferGroups = parseInt(document.getElementById('tts_buffer_groups_slider')?.value || "5");
                        const paddingMs = parseInt(document.getElementById('tts_padding_ms_slider')?.value || "0");
                        const batchGroups = parseInt(document.getElementById('tts_batch_groups_slider')?.value || "7");

                        const ttsReq = {
                            text: accumulatedLLMTextForDisplay,
                            voice: voice,
                            tts_temperature: ttsTemp,
                            tts_top_p: ttsTopP,
                            buffer_groups: bufferGroups,
                            padding_ms: paddingMs,
                            min_decode_batch_groups: batchGroups
                        };

                        try {
                            if (useWebSocketForAudio) {
                                // Use WebSocket streaming
                                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                                const wsUrl = `${protocol}//${window.location.host}/ws/tts`;
                                
                                await streamTTSAudioWebSocket(wsUrl, ttsReq);
                            } else {
                                // Use HTTP streaming
                                const ttsResponse = await fetch('/api/tts/stream', {
                                    method: 'POST',
                                    headers: { 'Content-Type': 'application/json' },
                                    body: JSON.stringify(ttsReq)
                                });

                                if (ttsResponse.ok) {
                                    // Play audio dynamically as chunks arrive
                                    await streamTTSAudio(ttsResponse);
                                } else {
                                    console.error("TTS generation failed after LLM completion.");
                                }
                            }
                        } catch (err) {
                            console.error("TTS generation error:", err);
                        }
                    }

                } catch (err) {
                    console.error("LLM Stream Error:", err);
                    accumulatedLLMTextForDisplay += `\n[Error: ${err.message}]`;
                    if (currentAssistantMessageContentElement) {
                        currentAssistantMessageContentElement.textContent = accumulatedLLMTextForDisplay;
                    }
                } finally {
                    if (generateButton) {
                        generateButton.disabled = false;
                        generateButton.textContent = "Generate";
                    }
                    if (currentAssistantMessageContentElement) {
                        currentAssistantMessageContentElement.classList.remove('streaming-llm-content');
                    }
                }
            }
        );
    }

    // ================================================================
    // BOOT
    // ================================================================
    initTTSVoices();

    renderChatHistory();

    if (recordButton) {
        recordButton.innerHTML = SVG_MIC;
    }

});