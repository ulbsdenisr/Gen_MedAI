/**
 * PATCH: loadExistingChat
 * Adauga aceasta functie in script.js (sau inlocuieste varianta existenta)
 * Foloseste campul "parsed" din raspunsul /api/get_chat pentru a redari
 * mesajele assistant cu displayAIResponse().
 */

async function loadExistingChat(conversationId) {
    try {
        // Seteaza conversatia curenta pe server
        await fetch('/api/load_chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ conversation_id: conversationId })
        });

        // Obtine mesajele
        const response = await fetch(`/api/get_chat/${conversationId}`);
        const data = await response.json();

        if (data.error) {
            console.error('Error loading chat:', data.error);
            return;
        }

        // Goleste zona de mesaje
        const chatMessages = document.getElementById('chatMessages');
        chatMessages.innerHTML = '';

        const messages = data.messages || [];

        for (const msg of messages) {
            if (msg.role === 'user') {
                // Afiseaza mesaj user
                addUserMessage(msg.message || msg.content || '');

            } else if (msg.role === 'assistant') {
                // Daca avem datele parsate, le redaram cu displayAIResponse
                if (msg.parsed && typeof displayAIResponse === 'function') {
                    displayAIResponse(msg.parsed);
                } else if (msg.message) {
                    // Fallback: incearca sa parseze string-ul JSON
                    try {
                        const parsed = JSON.parse(msg.message);
                        if (typeof displayAIResponse === 'function') {
                            displayAIResponse(parsed);
                        }
                    } catch (e) {
                        // Afiseaza ca text simplu daca nu e JSON valid
                        addBotMessage(msg.message);
                    }
                }
            }
        }

        // Scroll la final
        chatMessages.scrollTop = chatMessages.scrollHeight;

        // Inchide sidebar-ul dupa incarcare (pe mobile)
        const sidebar = document.getElementById('chatHistorySidebar');
        if (sidebar && window.innerWidth < 768) {
            sidebar.classList.remove('open');
        }

    } catch (error) {
        console.error('Error loading chat:', error);
    }
}


/**
 * PATCH: loadChatHistory
 * Incarca lista de conversatii in sidebar.
 * Inlocuieste sau adauga in script.js.
 */
async function loadChatHistory() {
    try {
        const response = await fetch('/api/chat_history');
        const data = await response.json();

        const historyList = document.getElementById('chatHistoryList');
        if (!historyList) return;

        historyList.innerHTML = '';

        const chats = data.chats || [];

        if (chats.length === 0) {
            historyList.innerHTML = '<div class="no-history">No conversations yet</div>';
            return;
        }

        chats.forEach(chat => {
            const item = document.createElement('div');
            item.className = 'chat-history-item';
            item.dataset.id = chat.id;

            // Formateaza timestamp
            const date = new Date(chat.timestamp * 1000);
            const dateStr = date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'});

            item.innerHTML = `
                <div class="chat-history-preview">${escapeHtml(chat.preview || 'Chat ' + chat.id)}</div>
                <div class="chat-history-date">${dateStr}</div>
            `;

            item.addEventListener('click', () => {
                // Evidentiaza elementul selectat
                document.querySelectorAll('.chat-history-item').forEach(el => el.classList.remove('active'));
                item.classList.add('active');
                loadExistingChat(chat.id);
            });

            historyList.appendChild(item);
        });

    } catch (error) {
        console.error('Error loading chat history:', error);
    }
}


/**
 * Helper: adauga mesaj user in chat (daca nu exista deja in script.js)
 */
function addUserMessage(text) {
    const chatMessages = document.getElementById('chatMessages');
    if (!chatMessages) return;

    const messageDiv = document.createElement('div');
    messageDiv.className = 'message user-message';
    messageDiv.innerHTML = `
        <div class="message-content">
            <p>${escapeHtml(text)}</p>
        </div>
        <div class="message-avatar">
            <i class="fas fa-user"></i>
        </div>
    `;
    chatMessages.appendChild(messageDiv);
}


/**
 * Helper: adauga mesaj simplu bot (fallback)
 */
function addBotMessage(text) {
    const chatMessages = document.getElementById('chatMessages');
    if (!chatMessages) return;

    const messageDiv = document.createElement('div');
    messageDiv.className = 'message bot-message';
    messageDiv.innerHTML = `
        <div class="message-avatar">
            <i class="fas fa-hospital"></i>
        </div>
        <div class="message-content">
            <p>${escapeHtml(text)}</p>
        </div>
    `;
    chatMessages.appendChild(messageDiv);
}
