const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const clearBtn = document.getElementById('clearBtn');
const predictBtn = document.getElementById('predictBtn');
const resultDisplay = document.getElementById('resultDisplay');
const confidenceValue = document.getElementById('confidenceValue');
const confidenceBar = document.getElementById('confidenceBar');
const topList = document.getElementById('topList');
const overlay = document.getElementById('canvasOverlay');

// Global drawing state
let isDrawing = false;
let hasDrawn = false;

// Initialize Canvas
ctx.lineWidth = 15;
ctx.lineCap = 'round';
ctx.strokeStyle = 'white';
ctx.fillStyle = 'black';
ctx.fillRect(0, 0, canvas.width, canvas.height); // Start with black background

// Events
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

// Touch support
canvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    const touch = e.touches[0];
    startDrawing(touch);
}, { passive: false });

canvas.addEventListener('touchmove', (e) => {
    e.preventDefault();
    const touch = e.touches[0];
    draw(touch);
}, { passive: false });

function startDrawing(e) {
    isDrawing = true;
    overlay.style.opacity = '0';
    hasDrawn = true;
    
    // Get correct coordinates
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX || e.pageX) - rect.left;
    const y = (e.clientY || e.pageY) - rect.top;
    
    ctx.beginPath();
    ctx.moveTo(x, y);
}

function draw(e) {
    if (!isDrawing) return;
    
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX || e.pageX) - rect.left;
    const y = (e.clientY || e.pageY) - rect.top;
    
    ctx.lineTo(x, y);
    ctx.stroke();
}

function stopDrawing() {
    isDrawing = false;
    ctx.closePath();
}

clearBtn.addEventListener('click', () => {
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    overlay.style.opacity = '1';
    hasDrawn = false;
    resetResults();
});

function resetResults() {
    resultDisplay.innerHTML = '<span class="placeholder">?</span>';
    confidenceValue.textContent = '0%';
    confidenceBar.style.width = '0%';
    topList.innerHTML = '<li><span class="empty">Draw something first</span></li>';
}

predictBtn.addEventListener('click', async () => {
    if (!hasDrawn) {
        alert("Please draw a character first!");
        return;
    }
    
    predictBtn.innerText = 'Analyzing...';
    predictBtn.disabled = true;

    try {
        const imageData = canvas.toDataURL('image/png');
        
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ image: imageData })
        });
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        updateUI(data);
    } catch (err) {
        console.error(err);
        resultDisplay.innerHTML = '<span class="placeholder">ERR</span>';
    } finally {
        predictBtn.innerText = 'Predict Now';
        predictBtn.disabled = false;
    }
});

function updateUI(data) {
    // Main character
    resultDisplay.textContent = data.character;
    
    // Confidence
    const conf = Math.round(data.confidence * 100);
    confidenceValue.textContent = `${conf}%`;
    confidenceBar.style.width = `${conf}%`;
    
    // Top suggestions
    const sorted = Object.entries(data.all_predictions)
        .sort(([,a], [,b]) => b - a)
        .slice(0, 5);
        
    topList.innerHTML = sorted.map(([char, prob]) => `
        <li>
            <span>${char}</span>
            <span>${Math.round(prob * 100)}%</span>
        </li>
    `).join('');
}
