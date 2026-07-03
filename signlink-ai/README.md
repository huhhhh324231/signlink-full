# SignLink Letters Web Demo

This folder contains the polished presentation UI for your sign-language alphabet project.

The current web app is designed for submission/demo use:
- modern responsive interface for desktop and phone
- live browser camera preview
- mock A-Z detection cards for presentation flow
- wording matched to your real product: collect data, train model, run realtime recognition

## What it is

This is the **web presentation layer** for the project.

Your real recognition pipeline still lives in the Python app inside:
- [gesture_recognition.py](/C:/Users/Admin/Desktop/projectai/gesture_recognition.py)
- [train.py](/C:/Users/Admin/Desktop/projectai/train.py)
- [collect_data.py](/C:/Users/Admin/Desktop/projectai/collect_data.py)

## What it is not yet

This UI is **not connected directly** to the Python model yet.
For submission/demo, it works well as a polished interface mockup with live camera access in the browser.

## Run locally

Prerequisite: Node.js installed on the machine.

1. Open terminal in:
   `C:\Users\Admin\Desktop\projectai\signlink-ai`
2. Install dependencies:
   `npm install`
3. Start dev server:
   `npm run dev`
4. Open:
   [http://localhost:3000](http://localhost:3000)

## Demo on phone

1. Connect phone and laptop to the same Wi-Fi.
2. Start the app with:
   `npm run dev`
3. Find your laptop IPv4 address with:
   `ipconfig`
4. Open on phone:
   `http://YOUR-IP:3000`

## Submission suggestion

For the strongest demo:
- use the Python app for real recognition
- use this web UI for polished visual presentation
- explain that the interface can later be connected to the trained backend model
