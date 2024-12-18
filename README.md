
# Indian Sign Language Translator Tool

This repository contains a tool for translating Indian Sign Language (ISL) hand signs to text and vice versa using various libraries such as MediaPipe, OpenCV, and tkinter. The tool leverages machine learning to recognize hand signs and convert them into corresponding text.

## Project Structure

- `model.p`: Contains the trained model for hand sign recognition.
- `gui_twoway.py`: The main script for running the application.
- Other scripts and resources for handling hand sign to text conversion, speech recognition, and GUI elements.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/ThiccBoiPala/Indian-SIgn-Language-Translator-Tool.git
    cd Indian-SIgn-Language-Translator-Tool
    ```

2. Place the trained model file `model.p` in the root directory.

## Usage

Run the gui_twoway.py script to start the application:
```sh
python gui_twoway.py
```

### Hand Sign to Text Conversion

- The application will open the webcam and start recognizing hand signs.
- The recognized text will be displayed on the screen.

### Speech to Hand Sign

- The application can also convert spoken words into corresponding hand signs displayed as images.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

