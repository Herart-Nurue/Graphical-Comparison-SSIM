from flask import Flask, request, render_template_string, send_from_directory, jsonify
import os
import time
from skimage.metrics import structural_similarity as ssim
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import base64
import cv2
import numpy as np
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Capture Canvas</title>
</head>
<body>
    <h1>Capture Canvas</h1>
    <form action="/capture" method="post">
        <input type="text" name="url" placeholder="Enter URL" required>
        <button type="submit">Capture</button>
    </form>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML)

@app.route('/capture', methods=['POST'])
def capture():
    url = request.form['url']
    expected_output = 'expected_output.png'
    actual_output = 'actual_output.png'
    result, message = capture_and_compare(url, 'canvas', actual_output, expected_output)
    if result:
        images_html = f"<img src='/images/{actual_output}' alt='Actual Output' style='width:300px;height:auto;'>"
        images_html += f"<img src='/images/{expected_output}' alt='Expected Output' style='width:300px;height:auto;'>"
        response = f"<h1>Comparison Result: {message}</h1>" + images_html
    else:
        response = f"<h1>Error: {message}</h1>"
    return response

@app.route('/images/<filename>')
def images(filename):
    return send_from_directory(os.getcwd(), filename)

def setup_driver():
    service = Service(ChromeDriverManager().install())
    options = webdriver.ChromeOptions()
    options.headless = True
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    driver = webdriver.Chrome(service=service, options=options)
    return driver

def capture_and_compare(url, canvas_css_selector, output_filename, expected_img):
    driver = setup_driver()
    try:
        driver.get(url)
        WebDriverWait(driver, 35).until(EC.presence_of_element_located((By.TAG_NAME, "canvas")))
        time.sleep(15) 
        canvas = WebDriverWait(driver, 30).until(EC.visibility_of_element_located((By.CSS_SELECTOR, canvas_css_selector)))
        canvas_base64 = driver.execute_script("return arguments[0].toDataURL('image/png').substring(22);", canvas)
        if not canvas_base64:
            raise ValueError("Failed to capture canvas data.")
        canvas_png = base64.b64decode(canvas_base64)
        with open(output_filename, 'wb') as file:
            file.write(canvas_png)
        logging.info(f"Canvas data captured and written to {output_filename}.")
        score = compare_images(expected_img, output_filename)
        return True, f"{score:.2f}"
    except Exception as e:
        logging.error("Failed to capture or compare images.", exc_info=True)
        return False, str(e)
    finally:
        driver.quit()

def compare_images(img1_path, img2_path):
    image1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    if image1 is None or image2 is None:
        logging.warning("One or both images failed to load, returning 0 similarity score.")
        return 0
    if image1.shape != image2.shape:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    image1 = cv2.GaussianBlur(image1, (5, 5), 0)
    image2 = cv2.GaussianBlur(image2, (5, 5), 0)
    ssim_index, _ = ssim(image1, image2, full=True)
    return ssim_index

if __name__ == "__main__":
    app.run(debug=True)