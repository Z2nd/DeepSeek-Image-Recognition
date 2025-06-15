document.addEventListener('DOMContentLoaded', () => {
    const imageUpload = document.getElementById('imageUpload');
    const processBtn = document.getElementById('processBtn');
    const spinner = document.getElementById('spinner');
    const resultsDiv = document.getElementById('results');
    const errorDiv = document.getElementById('error');
    const annotatedImage = document.getElementById('annotatedImage');
    const descriptionText = document.getElementById('descriptionText');

    processBtn.addEventListener('click', async () => {
        const file = imageUpload.files[0];
        if (!file) {
            alert('Please select an image file first.');
            return;
        }

        // Reset UI
        resultsDiv.style.display = 'none';
        errorDiv.style.display = 'none';
        spinner.style.display = 'block';

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/process_image', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Server error');
            }

            const data = await response.json();

            // Update UI with results
            annotatedImage.src = data.annotated_image;
            descriptionText.textContent = data.description;
            resultsDiv.style.display = 'flex';

        } catch (err) {
            errorDiv.textContent = `An error occurred: ${err.message}`;
            errorDiv.style.display = 'block';
        } finally {
            spinner.style.display = 'none';
        }
    });
});