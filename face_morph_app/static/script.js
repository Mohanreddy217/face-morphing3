const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const result = document.getElementById('result');

// Start the webcam
navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => {
    video.srcObject = stream;
  })
  .catch(err => {
    console.error('Webcam error:', err);
    result.textContent = "Error: Unable to access the webcam.";
  });

function captureImage() {
    const context = canvas.getContext('2d');

    // Set canvas size to match video stream
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw current frame
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(blob => {
        const formData = new FormData();
        formData.append('image', blob, 'webcam.png');

        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(res => {
            if (!res.ok) {
                return res.text().then(text => {
                    throw new Error(`Server error: ${res.status} - ${text}`);
                });
            }

            const contentType = res.headers.get("content-type");
            if (!contentType || !contentType.includes("application/json")) {
                throw new Error("Invalid JSON response from server");
            }

            return res.json();
        })
        .then(data => {
            if (data.result) {
                result.textContent = "Result: " + data.result;
            } else if (data.error) {
                result.textContent = "Error: " + data.error;
            } else {
                result.textContent = "Unexpected response from server.";
            }
        })
        .catch(err => {
            console.error('Prediction error:', err);
            result.textContent = "Error: " + err.message;
        });
    }, 'image/jpeg');
}
