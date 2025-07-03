// Global variables
let availableDates = [];
let currentDate = null;

// Initialize the application
document.addEventListener('DOMContentLoaded', async function() {
    // Load the manifest to get available dates
    await loadManifest();
    
    // Set up the date picker
    setupDatePicker();
    
    // Load today's papers by default
    loadToday();
});

// Load the manifest file to get available dates
async function loadManifest() {
    try {
        const response = await fetch('manifest.json');
        if (response.ok) {
            const data = await response.json();
            availableDates = data.dates || [];
        } else {
            // If manifest doesn't exist, try to scan for files
            console.warn('Manifest not found, will rely on direct file access');
        }
    } catch (error) {
        console.error('Error loading manifest:', error);
    }
}

// Set up the date picker constraints
function setupDatePicker() {
    const datePicker = document.getElementById('datePicker');
    const today = new Date();
    
    // Set max date to today
    datePicker.max = formatDateForInput(today);
    
    // Set min date to a reasonable past date (e.g., 1 year ago)
    const minDate = new Date(today);
    minDate.setFullYear(today.getFullYear() - 1);
    datePicker.min = formatDateForInput(minDate);
    
    // Set default value to today
    datePicker.value = formatDateForInput(today);
}

// Format date for input field (YYYY-MM-DD)
function formatDateForInput(date) {
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    return `${year}-${month}-${day}`;
}

// Format date for filename (YYYY_MMDD)
function formatDateForFilename(date) {
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    return `${year}_${month}${day}`;
}

// Load papers for the selected date
async function loadSelectedDate() {
    const datePicker = document.getElementById('datePicker');
    const selectedDate = new Date(datePicker.value);
    loadPapersForDate(selectedDate);
}

// Load today's papers
function loadToday() {
    const today = new Date();
    document.getElementById('datePicker').value = formatDateForInput(today);
    loadPapersForDate(today);
}

// Load papers for a specific date
async function loadPapersForDate(date) {
    currentDate = date;
    const container = document.getElementById('contentContainer');
    const navigation = document.getElementById('navigation');
    
    // Show loading state
    container.innerHTML = '<div class="loading">Loading papers...</div>';
    
    // Format the date for the filename
    const dateStr = formatDateForFilename(date);
    const filename = `out/output_${dateStr}.md`;
    
    try {
        // Try to fetch the markdown file
        const response = await fetch(filename);
        
        if (response.ok) {
            const markdown = await response.text();
            
            // Convert markdown to HTML
            const html = await convertMarkdownToHtml(markdown);
            
            // Display the content
            container.innerHTML = `<div class="markdown-content">${html}</div>`;
            
            // Show navigation
            navigation.style.display = 'flex';
            updateNavigationButtons();
        } else if (response.status === 404) {
            // No data for this date
            container.innerHTML = `
                <div class="no-data">
                    <h2>No papers available for ${date.toLocaleDateString()}</h2>
                    <p>Papers may not have been processed for this date, or it might be a weekend/holiday.</p>
                </div>
            `;
            navigation.style.display = 'none';
        } else {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
    } catch (error) {
        console.error('Error loading papers:', error);
        container.innerHTML = `
            <div class="error">
                <h2>Error loading papers</h2>
                <p>${error.message}</p>
            </div>
        `;
        navigation.style.display = 'none';
    }
}

// Convert markdown to HTML (basic implementation)
async function convertMarkdownToHtml(markdown) {
    // For GitHub Pages, we might have access to a markdown parser
    // For now, we'll use a basic implementation or load a library
    
    // Check if we have marked.js available
    if (typeof marked !== 'undefined') {
        return marked.parse(markdown);
    }
    
    // Try to load marked.js dynamically
    try {
        if (!window.markedLoaded) {
            await loadScript('https://cdn.jsdelivr.net/npm/marked/marked.min.js');
            window.markedLoaded = true;
        }
        return marked.parse(markdown);
    } catch (error) {
        // Fallback to basic markdown conversion
        return basicMarkdownToHtml(markdown);
    }
}

// Basic markdown to HTML converter (fallback)
function basicMarkdownToHtml(markdown) {
    let html = markdown;
    
    // Convert headers
    html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
    html = html.replace(/^## (.*$)/gim, '<h2>$1</h2>');
    html = html.replace(/^# (.*$)/gim, '<h1>$1</h1>');
    
    // Convert bold
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    
    // Convert italic
    html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
    
    // Convert links
    html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>');
    
    // Convert line breaks
    html = html.replace(/\n\n/g, '</p><p>');
    html = '<p>' + html + '</p>';
    
    // Convert lists
    html = html.replace(/^\* (.+)$/gim, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');
    
    // Convert code blocks
    html = html.replace(/```([^`]+)```/g, '<pre><code>$1</code></pre>');
    
    // Convert inline code
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    
    return html;
}

// Load external script
function loadScript(src) {
    return new Promise((resolve, reject) => {
        const script = document.createElement('script');
        script.src = src;
        script.onload = resolve;
        script.onerror = reject;
        document.head.appendChild(script);
    });
}

// Load previous day's papers
function loadPreviousDay() {
    if (!currentDate) return;
    
    const prevDate = new Date(currentDate);
    prevDate.setDate(prevDate.getDate() - 1);
    
    document.getElementById('datePicker').value = formatDateForInput(prevDate);
    loadPapersForDate(prevDate);
}

// Load next day's papers
function loadNextDay() {
    if (!currentDate) return;
    
    const nextDate = new Date(currentDate);
    nextDate.setDate(nextDate.getDate() + 1);
    
    // Don't go beyond today
    const today = new Date();
    if (nextDate > today) {
        nextDate.setTime(today.getTime());
    }
    
    document.getElementById('datePicker').value = formatDateForInput(nextDate);
    loadPapersForDate(nextDate);
}

// Update navigation button states
function updateNavigationButtons() {
    const prevButton = document.getElementById('prevButton');
    const nextButton = document.getElementById('nextButton');
    const today = new Date();
    
    // Disable next button if we're at today
    if (currentDate && formatDateForInput(currentDate) === formatDateForInput(today)) {
        nextButton.disabled = true;
    } else {
        nextButton.disabled = false;
    }
    
    // Could also check against manifest for previous button
    prevButton.disabled = false;
}