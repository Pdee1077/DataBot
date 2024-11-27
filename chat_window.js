// Update the datetime dynamically
function updateDateTime() {
    const now = new Date();
    document.getElementById('datetime').textContent = now.toLocaleString();
}
setInterval(updateDateTime, 1000);

// Send the user's message to the server
function sendMessage() {
    const userInput = document.getElementById('user_input').value;
    if (!userInput.trim()) return false;

    const csrfToken = document.getElementById('csrf_token').value;

fetch('', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken': csrfToken
    },
    body: JSON.stringify({ message: userInput })
})
.then(response => {
    if (!response.ok) {
        throw new Error('Network response was not ok');
    }
    return response.json();
})
.then(data => {
    console.log("Server Response: ", data);
    const chatMessages = document.getElementById('chat-messages');
    const timestamp = new Date().toLocaleTimeString();

    const userMessage = `
        <div class="user-message">
            <strong>You:</strong> ${userInput}
            <span class="timestamp">${timestamp}</span>
        </div>`;
    const botMessage = `
        <div class="bot-message">
            <strong>Bot (${data.api_used || 'unknown'}):</strong> ${data.reply || 'No response received.'}
            <span class="timestamp">${timestamp}</span>
        </div>`;
    chatMessages.innerHTML += userMessage + botMessage;
    chatMessages.scrollTop = chatMessages.scrollHeight;
})
.catch(error => {
    console.error('Error:', error);
    document.getElementById('error-message').textContent = "An error occurred while sending the message.";
});

function sendMessage() {
    const userInput = document.getElementById('user_input').value;
    if (!userInput.trim()) return false;

    const csrfToken = document.getElementById('csrf_token').value;

fetch('', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken': csrfToken
    },
    body: JSON.stringify({ message: userInput })
})
.then(response => {
    if (!response.ok) {
        throw new Error('Network response was not ok');
    }
    return response.json();
})
.then(data => {
    console.log("Server Response: ", data);
    const chatMessages = document.getElementById('chat-messages');
    const timestamp = new Date().toLocaleTimeString();

    const userMessage = `
        <div class="user-message">
            <strong>You:</strong> ${userInput}
            <span class="timestamp">${timestamp}</span>
        </div>`;
    const botMessage = `
        <div class="bot-message">
            <strong>Bot (${data.api_used || 'unknown'}):</strong> ${data.reply || 'No response received.'}
            <span class="timestamp">${timestamp}</span>
        </div>`;
    chatMessages.innerHTML += userMessage + botMessage;
    chatMessages.scrollTop = chatMessages.scrollHeight;
})
.catch(error => {
    console.error('Error:', error);
    document.getElementById('error-message').textContent = "An error occurred while sending the message.";
});

function sendMessage() {
    const userInput = document.getElementById('user_input').value;
    if (!userInput.trim()) return false;

    const csrfToken = document.getElementById('csrf_token').value;

    fetch('', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': csrfToken
        },
        body: JSON.stringify({ message: userInput })
    })
    .then(response => response.json())
    .then(data => {
        const chatMessages = document.getElementById('chat-messages');
        const timestamp = new Date().toLocaleTimeString();

        const userMessage = `
            <div class="user-message">
                <strong>You:</strong> ${userInput}
                <span class="timestamp">${timestamp}</span>
            </div>`;
        const botMessage = `
            <div class="bot-message">
                <strong>Bot:</strong> ${data.reply || 'No response received.'}
                <span class="timestamp">${timestamp}</span>
            </div>`;
        chatMessages.innerHTML += userMessage + botMessage;
        chatMessages.scrollTop = chatMessages.scrollHeight;
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while sending the message.');
    });

    document.getElementById('user_input').value = '';
    return false;
}


