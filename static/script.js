// DOM Elements
const chatMessages = document.getElementById('chatMessages');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const loadingModal = document.getElementById('loadingModal');
const statusIndicator = document.getElementById('statusIndicator');
const statusText = document.getElementById('statusText');
const statusDot = statusIndicator.querySelector('.status-dot');

// Chat History Elements
const chatHistorySidebar = document.getElementById('chatHistorySidebar');
const chatHistoryList = document.getElementById('chatHistoryList');
const historyToggleBtn = document.getElementById('historyToggleBtn');
const closeSidebarBtn = document.getElementById('closeSidebarBtn');
const newChatActionBtn = document.getElementById('newChatActionBtn');
let hasMedicalHistory = false;

function updateTimelineVisibility() {
    const timelineBtn = document.getElementById('timelineBtn');
    if (!timelineBtn) return;
    if (currentUser && hasMedicalHistory) {
        timelineBtn.style.display = 'inline-block';
    } else {
        timelineBtn.style.display = 'none';
    }
}
async function loadTimeline() {
    const modal = document.getElementById('timelineModal');
    const container = document.getElementById('timelineContent');
    modal.classList.remove('hidden');
    container.innerHTML = "<p>Loading timeline...</p>";
    await new Promise(resolve => setTimeout(resolve, 0));
    try {
        const res = await fetch('/api/get_history_timeline');
        const data = await res.json();
        if (data.error) {
            container.innerHTML = `<p>${data.error}</p>`;
            return;
        }
        renderTimeline(data.history);
    } catch (err) {
        console.error(err);
        container.innerHTML = "<p>Error loading timeline</p>";
    }
}
function renderTimeline(history) {
    const container = document.getElementById('timelineContent');
    container.innerHTML = '';
    if (!history || history.length === 0) {
        container.innerHTML = '<p>No medical history for this chat.</p>';
        return;
    }
    history.forEach(visit => {
        const block = document.createElement('div');
        block.style.borderLeft = '3px solid #3498db';
        block.style.margin = '10px 0';
        block.style.paddingLeft = '10px';
        let html = `<strong>${new Date(visit.timestamp).toLocaleString()}</strong><br>`;
        visit.entries.forEach(e => {
            html += `• ${e.symptom} (Severity: ${e.severity}, Status: ${e.status})<br>`;
        });
        block.innerHTML = html;
        container.appendChild(block);
    });
}
async function downloadPDF() {
    const btn = document.getElementById('downloadPdfBtn');
    btn.disabled = true;
    btn.textContent = "Preparing PDF...";
    try {
        const response = await fetch('/api/export_history_pdf', { method: 'POST' });
        if (!response.ok) { alert("Failed to download PDF"); return; }
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'medical_history.pdf';
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);
    } catch (err) {
        console.error(err);
    } finally {
        btn.disabled = false;
        btn.textContent = "Download Medical History";
    }
}
function closeTimeline() {
    document.getElementById('timelineModal').classList.add('hidden');
}

let currentConversationId = null;
let authMode = "login";
let currentUser = null;

userInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 120) + 'px';
});

userInput.addEventListener('keydown', function(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
});

sendBtn.addEventListener('click', sendMessage);
historyToggleBtn.addEventListener('click', () => { chatHistorySidebar.classList.toggle('hidden'); });
closeSidebarBtn.addEventListener('click', () => { chatHistorySidebar.classList.add('hidden'); });
newChatActionBtn.addEventListener('click', startNewChat);

document.addEventListener('DOMContentLoaded', async () => {
    await startNewChat();
    await loadChatHistory();
    checkHealth();
    document.getElementById('loginBtn').addEventListener('click', () => openAuthModal("login"));
    document.getElementById('signupBtn').addEventListener('click', () => openAuthModal("signup"));
    document.getElementById('logoutBtn').addEventListener('click', logout);
    document.getElementById('authSubmitBtn').addEventListener('click', submitAuth);
    const timelineBtn = document.getElementById('timelineBtn');
    if (timelineBtn) timelineBtn.addEventListener('click', loadTimeline);
    const downloadBtn = document.getElementById('downloadPdfBtn');
    if (downloadBtn) downloadBtn.addEventListener('click', downloadPDF);
});

setInterval(checkHealth, 30000);

function openAuthModal(mode) {
    authMode = mode;
    document.getElementById('authTitle').textContent = mode === "login" ? "Login" : "Sign Up";
    document.getElementById('authModal').classList.remove('hidden');
}
function closeAuthModal() {
    document.getElementById('authModal').classList.add('hidden');
}
async function submitAuth() {
    const username = document.getElementById('authUsername').value;
    const password = document.getElementById('authPassword').value;
    const endpoint = authMode === "login" ? "/api/login" : "/api/signup";
    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password })
        });
        const data = await response.json();
        if (response.ok) {
            currentUser = username;
            updateAuthUI();
            closeAuthModal();
            resetChatUI();
            chatHistoryList.innerHTML = '<p style="padding:16px;color:#999;text-align:center;">Loading chats...</p>';
            await startNewChat();
            await loadChatHistory();
        } else {
            alert(data.error || "Auth failed");
        }
    } catch (err) {
        console.error(err);
        alert("Error connecting to server");
    }
}
async function logout() {
    try {
        await fetch('/api/logout', { method: 'POST' });
        currentUser = null;
        currentConversationId = null;
        resetChatUI();
        chatHistoryList.innerHTML = `<p style="padding: 16px; color: #999; text-align: center;">No chats (logged out)</p>`;
        chatHistorySidebar.classList.add('hidden');
        updateAuthUI();
    } catch (err) {
        console.error('Logout failed:', err);
    }
}
async function handleAuthSuccess() {
    resetChatUI();
    await startNewChat();
    await loadChatHistory();
}
async function handleLogout() {
    await fetch('/api/logout', { method: 'POST' });
    resetChatUI();
    chatHistoryList.innerHTML = `<p style="padding: 16px; color: #999; text-align: center;">No chats (logged out)</p>`;
    currentConversationId = null;
}
function resetChatUI() {
    chatMessages.innerHTML = `
        <div class="message bot-message welcome-message">
            <div class="message-avatar">🏥</div>
            <div class="message-content">
                <p><strong>Welcome to MedAI Assistant!</strong></p>
                <p>I'm an AI-powered medical assistant trained to help identify symptoms and provide relevant medical information.</p>
                <p class="info-text">📝 Type your symptoms or medical concerns below.</p>
            </div>
        </div>
    `;
}
function updateAuthUI() {
    const loginBtn = document.getElementById('loginBtn');
    const signupBtn = document.getElementById('signupBtn');
    const logoutBtn = document.getElementById('logoutBtn');
    const userDisplay = document.getElementById('userDisplay');
    const timelineBtn = document.getElementById('timelineBtn');
    updateTimelineVisibility();
    if (currentUser) {
        loginBtn.style.display = 'none';
        signupBtn.style.display = 'none';
        logoutBtn.style.display = 'inline-block';
        userDisplay.style.display = 'inline-block';
        userDisplay.textContent = `👤 ${currentUser}`;
        if (timelineBtn) timelineBtn.style.display = 'inline-block';
    } else {
        loginBtn.style.display = 'inline-block';
        signupBtn.style.display = 'inline-block';
        logoutBtn.style.display = 'none';
        userDisplay.style.display = 'none';
        if (timelineBtn) timelineBtn.style.display = 'none';
    }
}

async function startNewChat() {
    try {
        hasMedicalHistory = false;
        updateTimelineVisibility();
        const response = await fetch('/api/new_chat', { method: 'POST' });
        const data = await response.json();
        currentConversationId = data.conversation_id;
        chatMessages.innerHTML = '';
        chatMessages.innerHTML = `
            <div class="message bot-message welcome-message">
                <div class="message-avatar">🏥</div>
                <div class="message-content">
                    <p><strong>Welcome to MedAI Assistant!</strong></p>
                    <p>I'm an AI-powered medical assistant trained to help identify symptoms and provide relevant medical information.</p>
                    <p class="info-text">📝 Type your symptoms or medical concerns below, and I'll analyze them to provide relevant information.</p>
                </div>
            </div>
        `;
        await loadChatHistory();
        closeSidebarBtn.click();
    } catch (error) {
        console.error('Failed to start new chat:', error);
    }
}

async function loadChatHistory() {
    try {
        const response = await fetch('/api/chat_history');
        const data = await response.json();
        if (data.error) {
            chatHistoryList.innerHTML = '<p style="padding: 16px; color: #999;">Error loading chat history</p>';
            return;
        }
        const chats = data.chats || [];
        if (chats.length === 0) {
            chatHistoryList.innerHTML = '<p style="padding: 16px; color: #999; text-align: center;">No previous chats</p>';
            return;
        }
        chatHistoryList.innerHTML = '';
        chats.forEach(chat => {
            const chatItem = document.createElement('div');
            chatItem.className = 'chat-item';
            const date = new Date(chat.timestamp * 1000);
            const timeStr = date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            chatItem.innerHTML = `
                <div class="chat-item-time">${timeStr}</div>
                <div class="chat-item-preview">${escapeHtml(chat.preview || 'No messages')}</div>
            `;
            if (chat.id === currentConversationId) chatItem.classList.add('active');
            chatItem.addEventListener('click', () => loadExistingChat(chat.id));
            chatHistoryList.appendChild(chatItem);
        });
    } catch (error) {
        console.error('Error loading chat history:', error);
        chatHistoryList.innerHTML = '<p style="padding: 16px; color: #999;">Error loading chats</p>';
    }
}

async function loadExistingChat(conversationId) {
    try {
        const response = await fetch('/api/load_chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ conversation_id: conversationId })
        });
        if (response.ok) {
            currentConversationId = conversationId;
            const messagesResponse = await fetch(`/api/get_chat/${conversationId}`);
            const messagesData = await messagesResponse.json();
            if (messagesResponse.ok && messagesData.messages) {
                chatMessages.innerHTML = '';
                messagesData.messages.forEach(msg => {
                    const role = msg.role || 'bot';
                    const content = msg.message || msg.content || '';
                    let displayText = content;
                    if (role === 'assistant' && typeof content === 'string') {
                        try {
                            const parsed = JSON.parse(content);
                            displayAIResponse(parsed);
                            return;
                        } catch (e) {}
                    }
                    addMessage(displayText, role);
                });
                await loadChatHistory();
                closeSidebarBtn.click();
            } else {
                alert('Failed to load chat messages');
            }
        } else {
            alert('Failed to load chat');
        }
    } catch (error) {
        console.error('Error loading chat:', error);
        alert('Error loading chat: ' + error.message);
    }
}

async function checkHealth() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        if (data.ready) {
            statusDot.classList.remove('error');
            statusText.textContent = 'Connected';
            sendBtn.disabled = false;
        } else {
            statusDot.classList.add('error');
            statusText.textContent = 'Models loading...';
            sendBtn.disabled = true;
        }
    } catch (error) {
        statusDot.classList.add('error');
        statusText.textContent = 'Offline';
        sendBtn.disabled = true;
    }
}

// ─── SEND MESSAGE ────────────────────────────────────────────────────────────
async function sendMessage() {
    const message = userInput.value.trim();
    if (!message) return;

    addMessage(message, 'user');
    userInput.value = '';
    userInput.style.height = 'auto';
    showLoading(true);

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: message })
        });
        const data = await response.json();

        if (response.ok) {
            // Afișează rezultatele inițiale
            displayAIResponse(data);

            // ── FOLLOW-UP INTEGRATION ──────────────────────────────────────
            // Dacă backend-ul zice că rezultatele sunt ambigue, afișăm întrebări
            if (typeof FollowUp !== 'undefined') {
                FollowUp.handle(
                    data,
                    data.extracted_symptoms || [],
                    (updatedData) => {
                        // Callback: utilizatorul a răspuns → afișăm rezultatele actualizate
                        displayAIResponse(updatedData);
                        scrollToBottom();
                    }
                );
            }
            // ──────────────────────────────────────────────────────────────

        } else {
            addMessage(`Error: ${data.error || 'Failed to process message'}`, 'bot', true);
        }
    } catch (error) {
        console.error('Error:', error);
        addMessage('Sorry, I encountered an error processing your message. Please try again.', 'bot', true);
    } finally {
        showLoading(false);
    }
}

// ─── ADD MESSAGE ─────────────────────────────────────────────────────────────
function addMessage(text, sender = 'user', isError = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    if (isError) messageDiv.classList.add('error-message');

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = sender === 'user' ? '👤' : '🏥';

    const content = document.createElement('div');
    content.className = 'message-content';
    content.innerHTML = `<p>${escapeHtml(text)}</p>`;

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(content);
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

// ─── DISPLAY AI RESPONSE ─────────────────────────────────────────────────────
function displayAIResponse(data) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message bot-message';

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = '🏥';

    const content = document.createElement('div');
    content.className = 'message-content';

    let html = '<p><strong>Analysis Results:</strong></p>';

    // Symptoms
    if (data.extracted_symptoms && data.extracted_symptoms.length > 0) {
        html += '<p><strong>Identified Symptoms:</strong></p>';
        data.extracted_symptoms.forEach(symptom => {
            html += `<span class="symptoms-badge">${escapeHtml(symptom)}</span>`;
        });
        html += '<br>';
        hasMedicalHistory = true;
        updateTimelineVisibility();
    } else {
        html += '<p><em>No specific symptoms detected.</em></p>';
    }

    // Diseases + Chart
    if (data.possible_diseases && data.possible_diseases.length > 0) {
        data.possible_diseases.sort((a, b) => b.percentage - a.percentage);
        const colors = getColorPalette(data.possible_diseases.length);

        html += `<div class="retrieved-docs">`;
        html += `
        <div class="results-container">
            <div class="disease-list">
                <div class="retrieved-docs-title">🔍 Possible Conditions:</div>
        `;

        data.possible_diseases.forEach((disease, index) => {
            const color = colors[index];
            html += `
            <div class="doc-item" style="border-left-color:${color}">
                🏥 <strong>#${disease.rank} ${escapeHtml(disease.disease)}</strong>
                <span style="float:right">${(disease.percentage || 0).toFixed(1)}%</span>
                <br>
                <small>${disease.overlap_symptoms?.map(s => escapeHtml(s)).join(', ') || ''}</small>
            </div>
            `;
        });

        html += `
            </div>
            <div class="chart-container">
                <canvas class="diseaseChart"></canvas>
            </div>
        </div>
        </div>
        `;
    } else {
        html += '<p><em>No matching medical conditions found.</em></p>';
    }

    // ── LOW CONFIDENCE BADGE ─────────────────────────────────────────────────
    // Arată un badge vizual dacă rezultatele sunt ambigue (follow-up în curs)
    if (data.needs_followup) {
        html += `
        <div style="
            display:inline-flex; align-items:center; gap:6px;
            background:#fff8e1; border:1px solid #ffe082;
            border-radius:8px; padding:6px 12px; margin:8px 0;
            font-size:13px; color:#f57f17;
        ">
            ⚠️ <strong>Low confidence</strong> — answering the questions below will improve accuracy.
        </div>`;
    }
    // ─────────────────────────────────────────────────────────────────────────

    // Warnings
    if (data.warnings) {
        html += `
        <div class="retrieved-docs" style="border-left:4px solid #e74c3c;">
            <div class="retrieved-docs-title">Warnings:</div>
            <div class="doc-item">${escapeHtml(data.warnings)}</div>
        </div>`;
    }

    // Top disease
    if (data.top_disease) {
        const td = data.top_disease;
        html += `<div class="retrieved-docs">
        <div class="retrieved-docs-title">🧠 Top Condition Details:</div>
        <div class="doc-item"><strong>${escapeHtml(td.name)}</strong></div>`;

        if (td.summary) {
            const sources = td.summary.sources || [];
            let sourcesHtml = "N/A";
            if (Array.isArray(sources) && sources.length > 0) {
                sourcesHtml = sources.map(src => {
                    if (typeof src === "string") return escapeHtml(src);
                    return `
                        <div style="margin-bottom:8px;">
                            • <strong>${escapeHtml(src.title || "Unknown source")}</strong>
                            ${src.journal ? ` | ${escapeHtml(src.journal)}` : ""}
                            ${src.year ? ` (${escapeHtml(src.year)})` : ""}
                        </div>
                    `;
                }).join("");
            }
            html += `
            <div class="doc-item"><strong>Overview:</strong> ${escapeHtml(formatField(td.summary.overview) || 'N/A')}</div>
            <div class="doc-item"><strong>Causes:</strong> ${escapeHtml(formatField(td.summary.causes) || 'N/A')}</div>
            <div class="doc-item"><strong>Treatment:</strong> ${escapeHtml(formatField(td.summary.treatment) || 'N/A')}</div>
            <div class="doc-item"><strong>Prevention:</strong> ${escapeHtml(formatField(td.summary.prevention) || 'N/A')}</div>
            <div class="doc-item"><strong>When to see a doctor:</strong> ${escapeHtml(formatField(td.summary.when_to_see_a_doctor) || 'N/A')}</div>
            <div class="doc-item"><strong>Sources:</strong><br>${sourcesHtml}</div>
            `;
        }
        html += '</div>';
    }

    content.innerHTML = html;

    // Chart
    if (data.possible_diseases && data.possible_diseases.length > 0) {
        const canvas = content.querySelector('.diseaseChart');
        if (canvas) {
            const labels = data.possible_diseases.map(d => d.disease);
            const values = data.possible_diseases.map(d => d.percentage || 0);
            const colors = getColorPalette(labels.length);
            new Chart(canvas, {
                type: 'pie',
                data: { labels, datasets: [{ data: values, backgroundColor: colors }] },
                options: { responsive: true, plugins: { legend: { display: false } } }
            });
        }
    }

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(content);
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

// ─── HELPERS ──────────────────────────────────────────────────────────────────
function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}
function showLoading(show) {
    if (show) {
        loadingModal.classList.remove('hidden');
    } else {
        loadingModal.classList.add('hidden');
    }
}
function escapeHtml(text) {
    if (text === null || text === undefined) return '';
    if (typeof text === 'object') text = JSON.stringify(text);
    text = String(text);
    return text.replace(/[&<>"']/g, m => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;'
    }[m]));
}
function formatField(value) {
    if (Array.isArray(value)) return value.join(" ");
    return value;
}
function getColorPalette(n) {
    const colors = [
        '#3498db', '#e74c3c', '#2ecc71', '#f1c40f',
        '#9b59b6', '#1abc9c', '#e67e22', '#34495e'
    ];
    return Array.from({ length: n }, (_, i) => colors[i % colors.length]);
}
