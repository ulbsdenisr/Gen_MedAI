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
        const response = await fetch('/api/export_history_pdf', {
            method: 'POST'
        });

        if (!response.ok) {
            alert("Failed to download PDF");
            return;
        }

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

// Current conversation ID
let currentConversationId = null;
let authMode = "login"; // or "signup"
let currentUser = null;

// Auto-resize textarea
userInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 120) + 'px';
});

// Send message on Enter (Shift+Enter for newline)
userInput.addEventListener('keydown', function(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
});

// Send button click
sendBtn.addEventListener('click', sendMessage);

// Chat history events
historyToggleBtn.addEventListener('click', () => {
    chatHistorySidebar.classList.toggle('hidden');
});

closeSidebarBtn.addEventListener('click', () => {
    chatHistorySidebar.classList.add('hidden');
});

newChatActionBtn.addEventListener('click', startNewChat);

// Check connection status on load
document.addEventListener('DOMContentLoaded', async () => {
    // Sidebar is now visible by default
    // if (window.innerWidth <= 768) {
    //     chatHistorySidebar.classList.add('hidden');
    // }
    
    await startNewChat();
    await loadChatHistory();
    checkHealth();
    document.getElementById('loginBtn').addEventListener('click', () => openAuthModal("login"));
    document.getElementById('signupBtn').addEventListener('click', () => openAuthModal("signup"));
    document.getElementById('logoutBtn').addEventListener('click', logout);
    document.getElementById('authSubmitBtn').addEventListener('click', submitAuth);
     // ✅ SAFE timeline button
    const timelineBtn = document.getElementById('timelineBtn');
    if (timelineBtn) {
        timelineBtn.addEventListener('click', loadTimeline);
    }

    // ✅ SAFE download button
    const downloadBtn = document.getElementById('downloadPdfBtn');
    if (downloadBtn) {
        downloadBtn.addEventListener('click', downloadPDF);
    }
});

// Periodically check health
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
            // Clear guest chat & sidebar
            resetChatUI();
            chatHistoryList.innerHTML = '<p style="padding:16px;color:#999;text-align:center;">Loading chats...</p>';
            // Start a new conversation for this user
            await startNewChat();
            await loadChatHistory(); // reload chats for user
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

        // Reset chat UI to welcome message
        resetChatUI();

        // Clear sidebar completely
        chatHistoryList.innerHTML = `
            <p style="padding: 16px; color: #999; text-align: center;">
                No chats (logged out)
            </p>
        `;

        // Hide sidebar completely if you want
        chatHistorySidebar.classList.add('hidden');

        // Update auth buttons
        updateAuthUI();
    } catch (err) {
        console.error('Logout failed:', err);
    }
}
async function handleAuthSuccess() {
    // 1. Clear chat UI
    resetChatUI();

    // 2. Start fresh chat
    await startNewChat();

    // 3. Reload sidebar history
    await loadChatHistory();
}
async function handleLogout() {
    await fetch('/api/logout', { method: 'POST' });

    // Reset chat UI
    resetChatUI();

    // Clear sidebar
    chatHistoryList.innerHTML = `
        <p style="padding: 16px; color: #999; text-align: center;">
            No chats (logged out)
        </p>
    `;

    // Reset current conversation
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
        // 🔥 SHOW timeline button
        if (timelineBtn) timelineBtn.style.display = 'inline-block';
    } else {
        loginBtn.style.display = 'inline-block';
        signupBtn.style.display = 'inline-block';
        logoutBtn.style.display = 'none';

        userDisplay.style.display = 'none';
         // 🔥 HIDE timeline button
        if (timelineBtn) timelineBtn.style.display = 'none';
    }
}

/**
 * Start a new chat
 */
async function startNewChat() {
    try {
        hasMedicalHistory = false;
        updateTimelineVisibility();
        const response = await fetch('/api/new_chat', {
            method: 'POST'
        });
        const data = await response.json();
        currentConversationId = data.conversation_id;
        
        // Clear chat messages
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
        
        // Reload chat history
        await loadChatHistory();
        closeSidebarBtn.click();
    } catch (error) {
        console.error('Failed to start new chat:', error);
    }
}

/**
 * Load and display chat history
 */
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
            
            // Format timestamp
            const date = new Date(chat.timestamp * 1000);
            const timeStr = date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], {
                hour: '2-digit',
                minute: '2-digit'
            });
            
            chatItem.innerHTML = `
                <div class="chat-item-time">${timeStr}</div>
                <div class="chat-item-preview">${escapeHtml(chat.preview || 'No messages')}</div>
            `;
            
            // Mark current chat as active
            if (chat.id === currentConversationId) {
                chatItem.classList.add('active');
            }
            
            chatItem.addEventListener('click', () => loadExistingChat(chat.id));
            chatHistoryList.appendChild(chatItem);
        });
    } catch (error) {
        console.error('Error loading chat history:', error);
        chatHistoryList.innerHTML = '<p style="padding: 16px; color: #999;">Error loading chats</p>';
    }
}

/**
 * Load an existing chat
 */
async function loadExistingChat(conversationId) {
    try {
        // First, set the current conversation on the backend
        const response = await fetch('/api/load_chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ conversation_id: conversationId })
        });
        
        if (response.ok) {
            currentConversationId = conversationId;
            
            // Now fetch and display the messages
            const messagesResponse = await fetch(`/api/get_chat/${conversationId}`);
            const messagesData = await messagesResponse.json();
            
            if (messagesResponse.ok && messagesData.messages) {
                // Clear and display messages
                chatMessages.innerHTML = '';
                
                messagesData.messages.forEach(msg => {
                    // msg can be format: {timestamp, role, message} or other formats
                    const role = msg.role || 'bot';
                    const content = msg.message || msg.content || '';
                    
                    // Parse if it's a JSON string response from AI
                    let displayText = content;
                    if (role === 'assistant' && typeof content === 'string') {
                        try {
                            const parsed = JSON.parse(content);
                            displayAIResponse(parsed);
                            return; // Skip the standard addMessage
                        } catch (e) {
                            // Not JSON, display as is
                        }
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

/**
 * Check if backend is ready
 */
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

/**
 * Send user message and get AI response
 */
async function sendMessage() {
    const message = userInput.value.trim();
    
    if (!message) return;
    
    // Add user message to chat
    addMessage(message, 'user');
    
    // Clear input
    userInput.value = '';
    userInput.style.height = 'auto';
    
    // Show loading
    showLoading(true);
    
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: message })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            displayAIResponse(data);
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

/**
 * Add a message to the chat
 */
function addMessage(text, sender = 'user', isError = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    
    if (isError) {
        messageDiv.classList.add('error-message');
    }
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = sender === 'user' ? '👤' : '🏥';
    
    const content = document.createElement('div');
    content.className = 'message-content';
    content.innerHTML = `<p>${escapeHtml(text)}</p>`;
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(content);
    chatMessages.appendChild(messageDiv);
    
    // Scroll to bottom
    scrollToBottom();
}

/**
 * Display AI response with extracted symptoms and retrieved documents
 */
function displayAIResponse(data) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message bot-message';
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = '🏥';
    
    const content = document.createElement('div');
    content.className = 'message-content';
    
    let html = '<p><strong>Analysis Results:</strong></p>';
    
    // Display extracted symptoms
    if (data.extracted_symptoms && data.extracted_symptoms.length > 0) {
        html += '<p><strong>Identified Symptoms:</strong></p>';
        data.extracted_symptoms.forEach(symptom => {
            html += `<span class="symptoms-badge">${escapeHtml(symptom)}</span>`;
        });
        html += '<br>';
        hasMedicalHistory = true;
        updateTimelineVisibility();
    } else {
        html += '<p><em>No specific symptoms detected. Please provide more details about your symptoms.</em></p>';
    }
    
    // Display possible diseases
    if (data.possible_diseases && data.possible_diseases.length > 0) {
        html += '<div class="retrieved-docs">';
        html += '<div class="retrieved-docs-title">🔍 Possible Conditions:</div>';
        data.possible_diseases.forEach(disease => {
    html += `<div class="doc-item">`;
    html += `🏥 <strong>#${disease.rank} ${escapeHtml(disease.disease)}</strong><br>`;

    if (disease.overlap_symptoms && disease.overlap_symptoms.length > 0) {
        html += `<small>Matched symptoms: ${disease.overlap_symptoms.map(s => escapeHtml(s)).join(', ')}</small>`;
    }

    html += `</div>`;
    });
        html += '</div>';
    } else {
        html += '<p><em>No matching medical conditions found for these symptoms.</em></p>';
    }
    // Display medical warnings
    if (data.warnings) {
        html += '<div class="retrieved-docs" style="border-left:4px solid #e74c3c;">';
        html += '<div class="retrieved-docs-title">Warnings: </div>';
        html += `<div class="doc-item">${escapeHtml(data.warnings)}</div>`;
        html += '</div>';
    }
if (data.top_disease) {
    const td = data.top_disease;

    html += '<div class="retrieved-docs">';
    html += '<div class="retrieved-docs-title">🧠 Top Condition Details:</div>';

    html += `<div class="doc-item"><strong>${escapeHtml(td.name)}</strong></div>`;

    // Summary
    if (td.summary) {
        html += `<div class="doc-item"><strong>Overview:</strong> ${escapeHtml(td.summary.overview || 'N/A')}</div>`;
        html += `<div class="doc-item"><strong>Causes:</strong> ${escapeHtml(td.summary.causes || 'N/A')}</div>`;
        html += `<div class="doc-item"><strong>Treatment:</strong> ${escapeHtml(td.summary.treatment || 'N/A')}</div>`;
        html += `<div class="doc-item"><strong>Prevention:</strong> ${escapeHtml(td.summary.prevention || 'N/A')}</div>`;
        html += `<div class="doc-item"><strong>When to see a doctor:</strong> ${escapeHtml(td.summary.when_to_see_doctor || 'N/A')}</div>`;
    }

    // Articles
    if (td.articles && td.articles.length > 0) {
        html += `<div class="doc-item"><strong>Articles:</strong></div>`;
        td.articles.forEach(article => {
            html += `<div class="doc-item">📄 ${escapeHtml(article.title || 'Untitled')}</div>`;
        });
    } else {
        html += `<div class="doc-item"><em>No articles found</em></div>`;
    }

    html += '</div>';
}
    
    content.innerHTML = html;
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(content);
    chatMessages.appendChild(messageDiv);
    
    scrollToBottom();
}

/**
 * Scroll chat to bottom
 */
function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

/**
 * Show/hide loading indicator
 */
function showLoading(show) {
    if (show) {
        loadingModal.classList.remove('hidden');
    } else {
        loadingModal.classList.add('hidden');
    }
}

/**
 * Escape HTML special characters
 */
 /**
function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, m => map[m]);
}
*/
function escapeHtml(text) {
    if (text === null || text === undefined) return '';

    if (typeof text === 'object') {
        text = JSON.stringify(text);
    }

    text = String(text);

    return text.replace(/[&<>"']/g, m => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    }[m]));
}