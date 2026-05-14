/**
 * followup.js
 * -----------
 * Gestionează afișarea și colectarea răspunsurilor la întrebările de follow-up.
 *
 * Cum se integrează:
 *   1. Adaugă <script src="/static/followup.js"></script> în index.html
 *   2. După ce primești răspunsul de la /api/chat, apelează:
 *        FollowUp.handle(response, symptomsArray)
 *   3. Dacă needs_followup === true, modulul afișează butoanele automat în chat.
 */

const FollowUp = (() => {

    // --- State ---
    let _currentSymptoms = [];
    let _pendingQuestions = [];
    let _collectedAnswers = {};
    let _onResultCallback = null;   // apelat când avem răspunsul final

    // --- Config ---
    const FOLLOWUP_CONTAINER_ID = 'followup-container';
    const CHAT_MESSAGES_ID = 'chatMessages';

    // ---------------------------------------------------------------
    // Public API
    // ---------------------------------------------------------------

    /**
     * Punctul de intrare principal.
     * Apelează după fiecare răspuns de la /api/chat sau /api/followup.
     *
     * @param {Object} apiResponse   - răspunsul JSON de la server
     * @param {string[]} symptoms    - simptomele extrase până acum
     * @param {Function} onResult    - callback(response) apelat după re-scorare
     */
    function handle(apiResponse, symptoms, onResult) {
        _currentSymptoms = symptoms || [];
        _onResultCallback = onResult || null;
        _collectedAnswers = {};

        if (apiResponse.needs_followup && apiResponse.followup_questions?.length > 0) {
            _pendingQuestions = apiResponse.followup_questions;
            _renderQuestions(_pendingQuestions);
        }
    }

    // ---------------------------------------------------------------
    // Rendering
    // ---------------------------------------------------------------

    function _renderQuestions(questions) {
        _removeExistingContainer();

        const container = document.createElement('div');
        container.id = FOLLOWUP_CONTAINER_ID;
        container.className = 'followup-container';

        // Header
        const header = document.createElement('div');
        header.className = 'followup-header';
        header.innerHTML = `
            <span class="followup-icon">🔍</span>
            <span>To narrow down the results, please answer a few quick questions:</span>
        `;
        container.appendChild(header);

        // Întrebări
        const questionsWrap = document.createElement('div');
        questionsWrap.className = 'followup-questions';

        questions.forEach((q, idx) => {
            const row = _buildQuestionRow(q, idx);
            questionsWrap.appendChild(row);
        });

        container.appendChild(questionsWrap);

        // Buton Submit
        const submitBtn = document.createElement('button');
        submitBtn.className = 'followup-submit-btn';
        submitBtn.textContent = 'Submit answers';
        submitBtn.disabled = true;  // activat după ce toate întrebările sunt răspunse
        submitBtn.addEventListener('click', _onSubmit);
        container.appendChild(submitBtn);

        // Adaugă în chat
        const chatMessages = document.getElementById(CHAT_MESSAGES_ID);
        if (chatMessages) {
            chatMessages.appendChild(container);
            container.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        } else {
            // Fallback: adaugă în body dacă nu găsim containerul
            document.body.appendChild(container);
        }
    }

    function _buildQuestionRow(question, idx) {
        const row = document.createElement('div');
        row.className = 'followup-question-row';
        row.dataset.symptom = question.symptom;
        row.dataset.answered = 'false';

        const label = document.createElement('span');
        label.className = 'followup-question-text';
        label.textContent = question.question;

        const btnGroup = document.createElement('div');
        btnGroup.className = 'followup-btn-group';

        const yesBtn = _buildAnswerBtn('Yes', true, row);
        const noBtn = _buildAnswerBtn('No', false, row);

        btnGroup.appendChild(yesBtn);
        btnGroup.appendChild(noBtn);

        row.appendChild(label);
        row.appendChild(btnGroup);

        return row;
    }

    function _buildAnswerBtn(label, value, row) {
        const btn = document.createElement('button');
        btn.className = `followup-answer-btn followup-answer-${label.toLowerCase()}`;
        btn.textContent = label;

        btn.addEventListener('click', () => {
            // Marchează selecția vizual
            const siblings = row.querySelectorAll('.followup-answer-btn');
            siblings.forEach(b => b.classList.remove('selected'));
            btn.classList.add('selected');

            // Salvează răspunsul
            const symptom = row.dataset.symptom;
            _collectedAnswers[symptom] = value;
            row.dataset.answered = 'true';

            // Activează submit dacă toate au răspuns
            _checkAllAnswered();
        });

        return btn;
    }

    function _checkAllAnswered() {
        const container = document.getElementById(FOLLOWUP_CONTAINER_ID);
        if (!container) return;

        const rows = container.querySelectorAll('.followup-question-row');
        const allAnswered = Array.from(rows).every(r => r.dataset.answered === 'true');

        const submitBtn = container.querySelector('.followup-submit-btn');
        if (submitBtn) {
            submitBtn.disabled = !allAnswered;
        }
    }

    // ---------------------------------------------------------------
    // Submit
    // ---------------------------------------------------------------

    async function _onSubmit() {
        const container = document.getElementById(FOLLOWUP_CONTAINER_ID);
        if (container) {
            // Dezactivăm butoanele în timp ce așteptăm
            container.querySelectorAll('button').forEach(b => b.disabled = true);
            const submitBtn = container.querySelector('.followup-submit-btn');
            if (submitBtn) submitBtn.textContent = 'Analyzing...';
        }

        try {
            const resp = await fetch('/api/followup', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    symptoms: _currentSymptoms,
                    answers: _collectedAnswers,
                }),
            });

            if (!resp.ok) {
                throw new Error(`Server error: ${resp.status}`);
            }

            const data = await resp.json();

            // Îndepărtăm containerul curent
            _removeExistingContainer();

            // Dacă mai sunt întrebări (încă ambiguu), le afișăm
            if (data.needs_followup && data.followup_questions?.length > 0) {
                _pendingQuestions = data.followup_questions;
                // Actualizăm simptomele cunoscute cu ce a confirmat utilizatorul
                const confirmed = Object.entries(_collectedAnswers)
                    .filter(([_, v]) => v === true)
                    .map(([k]) => k);
                _currentSymptoms = [...new Set([..._currentSymptoms, ...confirmed])];
                _collectedAnswers = {};
                _renderQuestions(_pendingQuestions);
            }

            // Apelăm callback-ul cu noul răspuns
            if (_onResultCallback) {
                _onResultCallback(data);
            }

        } catch (err) {
            console.error('Follow-up request failed:', err);
            _removeExistingContainer();
            _showError('Could not process your answers. Please try again.');
        }
    }

    // ---------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------

    function _removeExistingContainer() {
        const existing = document.getElementById(FOLLOWUP_CONTAINER_ID);
        if (existing) existing.remove();
    }

    function _showError(msg) {
        const chatMessages = document.getElementById(CHAT_MESSAGES_ID);
        if (!chatMessages) return;
        const err = document.createElement('div');
        err.className = 'followup-error';
        err.textContent = msg;
        chatMessages.appendChild(err);
    }

    // ---------------------------------------------------------------
    // Public
    // ---------------------------------------------------------------
    return { handle };

})();
