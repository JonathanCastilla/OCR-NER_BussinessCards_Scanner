// Document Scanning With JavaScrip
// Author : Srikanth
// https://www.udemy.com/user/freeai-space/

// ============================================================================
// Global State Variables
// ============================================================================
var circles = [];                 // Array to store the 4 draggable corner objects
var canvas;                       // Reference to the HTML5 Canvas DOM element
var context;                      // The 2D rendering context of the canvas
var previousSelectedCircle;       // Tracks the currently selected/dragged circle
var isDragging = false;           // Boolean flag indicating if a drag event is active

// ============================================================================
// Initialization & Event Binding
// ============================================================================
window.onload = function() {
    canvas = document.getElementById("canvas");
    context = canvas.getContext("2d");
    
    // Bind mouse interaction events to the canvas
    canvas.onmousedown = canvasClick;
    canvas.onmouseup = stopDragging;
    canvas.onmouseout = stopDragging;  // Prevents the circle from sticking if the mouse leaves the canvas
    canvas.onmousemove = dragCircle;
};

// ============================================================================
// Data Structures
// ============================================================================
class Circle {
    /**
     * Represents a draggable geometric node (corner point) on the canvas.
     * @param {number} x - The X spatial coordinate.
     * @param {number} y - The Y spatial coordinate.
     * @param {number} radius - The visual radius of the node.
     * @param {string} color - The hexadecimal color code for rendering.
     */
    constructor(x, y, radius, color) {
        this.x = x;
        this.y = y;
        this.radius = radius;
        this.color = color;
        this.isSelected = false; // Tracks active interaction state
    }
}

// ============================================================================
// Coordinate Mapping Utility (THE RESPONSIVENESS FIX)
// ============================================================================
/**
 * Maps the physical screen mouse coordinates to the internal resolution of the canvas.
 * This ensures precise interactions even if the canvas is scaled via CSS.
 */
function getMousePos(canvas, evt) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;   // Relationship bitmap vs. element for X
    const scaleY = canvas.height / rect.height; // Relationship bitmap vs. element for Y

    return {
        x: (evt.clientX - rect.left) * scaleX,
        y: (evt.clientY - rect.top) * scaleY
    };
}

// ============================================================================
// Image Loading & Processing
// ============================================================================
function processFunction(file) {
    process(file);
}

function loadImage(file) {
    return new Promise((resolve, reject) => {
        const url = file;
        let img = new Image();
        img.onload = () => {
            resolve(img);
        };
        img.src = url;
    });
}

/**
 * Parses the initial coordinate payload provided by the Python backend (OpenCV)
 * and initializes the interactive Circle objects.
 */
function loadPoints(points) {
    for (var i = 0; i < points.length; i++) {
        var radius = 6;
        var x = points[i].x;
        var y = points[i].y;
        
        var color = "#FFFF00"; // Yellow nodes
        var circle = new Circle(x, y, radius, color);
        circles.push(circle);
    }
    // Initiate the image rendering pipeline
    processFunction('/static/media/resize_image.jpg');
}

// ============================================================================
// Rendering Loop
// ============================================================================
function drawCircles() {
    // Clear the previous frame to prevent rendering artifacts
    context.clearRect(0, 0, canvas.width, canvas.height);
    
    // Render the base document image
    context.drawImage(window.img, 0, 0, window.img.width, window.img.height);

    for (var i = 0; i < circles.length; i++) {
        var circle = circles[i];
        context.globalAlpha = 0.85;
        
        // Render the geometric node (circle)
        context.beginPath();
        context.arc(circle.x, circle.y, circle.radius, 0, Math.PI * 2);
        context.fillStyle = circle.color;
        context.strokeStyle = "#76FF03"; // Green outline

        // Increase visual weight if the node is currently being dragged
        if (circle.isSelected) {
            context.lineWidth = 4;
        } else {
            context.lineWidth = 2;
        }
        context.fill();
        context.stroke();
        
        // Render the topological boundary (lines connecting the 4 points)
        context.beginPath();
        context.moveTo(circle.x, circle.y);
        // Connect current node to the previous node (looping back from 0 to 3)
        context.lineTo(circles[i - 1 >= 0 ? i - 1 : 3].x, circles[i - 1 >= 0 ? i - 1 : 3].y);
        context.stroke();
    }
}

// ============================================================================
// Interaction Handlers (Mouse Events)
// ============================================================================
function canvasClick(e) {
    // Calculate the precise internal coordinates of the mouse click
    const pos = getMousePos(canvas, e);
    var clickX = pos.x;
    var clickY = pos.y;

    // Iterate backwards through the array to detect collisions (top-most elements first)
    for (var i = circles.length - 1; i >= 0; i--) {
        var circle = circles[i];
        
        // Execute Euclidean distance calculation to detect if the click falls within the node's radius
        var distanceFromCenter = Math.sqrt(Math.pow(circle.x - clickX, 2) + Math.pow(circle.y - clickY, 2));
        
        if (distanceFromCenter <= circle.radius + 5) { // Added a 5px padding for easier clicking
            if (previousSelectedCircle != null) previousSelectedCircle.isSelected = false;
            previousSelectedCircle = circle;
            
            circle.isSelected = true;
            isDragging = true;
            
            drawCircles();
            return;
        }
    }
}

function stopDragging() {
    isDragging = false;
}

function dragCircle(e) {
    // Update the spatial coordinates of the selected node in real-time
    if (isDragging == true) {
        if (previousSelectedCircle != null) {
            const pos = getMousePos(canvas, e);

            previousSelectedCircle.x = pos.x;
            previousSelectedCircle.y = pos.y;

            // Re-render the frame with the updated node position
            drawCircles();
        }
    }
}

// ============================================================================
// Asynchronous Server Communication (AJAX)
// ============================================================================
$(document).ready(function() {
    $("#sendData").click(function() {
        // Trigger visual feedback (loading state)
        $("#loader").html('<div class="spinner-border text-primary" role="status"><span class="visually-hidden">Loading...</span></div>');
        
        // Construct the structured JSON payload containing the finalized spatial coordinates
        $.ajax({
            type: 'POST',
            url: "/transform",
            contentType: 'application/json;charset=UTF-8',
            data: JSON.stringify({
                "data": [
                    [circles[0].x, circles[0].y],
                    [circles[1].x, circles[1].y],
                    [circles[2].x, circles[2].y],
                    [circles[3].x, circles[3].y]
                ]
            }),
            success: function() {
                // Redirect the client to the prediction visualization route upon successful transformation
                window.location.href = 'prediction';
            }
        });
    });
});

// ============================================================================
// Base Pipeline Execution
// ============================================================================
// ============================================================================
// Base Pipeline Execution
// ============================================================================
window.process = async (file) => { 
    // FIX: Buscar el canvas directamente en el DOM para evitar que esté "undefined"
    let currentCanvas = document.getElementById("canvas");
    let ctx = currentCanvas.getContext('2d');
    
    const img = await loadImage(file);
    
    // Sincronizar el canvas con la resolución real de la imagen
    currentCanvas.width = img.width;
    currentCanvas.height = img.height;
    
    // Guardar la imagen en caché global
    window.img = img;
    
    // Asegurarse de que nuestras variables globales estén inicializadas
    canvas = currentCanvas;
    context = ctx;
    
    // Dibujar el primer frame
    ctx.drawImage(img, 0, 0, img.width, img.height);
    drawCircles();
};