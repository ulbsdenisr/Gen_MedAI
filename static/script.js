// DOM Elements
const chatMessages = document.getElementById('chatMessages');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const loadingModal = document.getElementById('loadingModal');
const statusIndicator = document.getElementById('statusIndicator');
const statusText = document.getElementById('statusText');
const statusDot = statusIndicator.querySelector('.status-dot');

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

// Check connection status on load
document.addEventListener('DOMContentLoaded', async () => {
    await startNewChat();
    checkHealth();
});

// Periodically check health
setInterval(checkHealth, 30000);

/**
 * Check if backend is ready
 */
async function startNewChat() {
    try {
        await fetch('/api/new_chat', {
            method: 'POST'
        });
    } catch (error) {
        console.error('Failed to start new chat:', error);
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
    } else {
        html += '<p><em>No specific symptoms detected. Please provide more details about your symptoms.</em></p>';
    }
    
    // Display possible diseases
    if (data.possible_diseases && data.possible_diseases.length > 0) {
        html += '<div class="retrieved-docs">';
        html += '<div class="retrieved-docs-title">🔍 Possible Conditions:</div>';
        data.possible_diseases.forEach((disease, index) => {
            html += `<div class="doc-item">🏥 ${escapeHtml(disease)}</div>`;
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
