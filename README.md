# tCam-mini-with-Tensorflow-Lite
Incorporate machine learning capability into tCam-mini

***Special thank to Dan Julio for clarification on all of my inquiries regarding the project and his hardware***

This project is based on the tCam thermal imaging sensor made by Dan Julio.
Here is the original repository.
[Here](https://github.com/danjulio/tCam)

Specifically here is where the firmware is adapted from.
[Here](https://github.com/danjulio/tCam/tree/main/tCam-Mini/firmware)

Please check out Dan Julio [website](http://www.danjuliodesigns.com/products/tcam_mini.html) for more details 

The aim of this project is to use tCam-mini to capture and detect the current frame, and determine 
the classification of the frame. This prototype is meant to be used in a static environment where the camera
itself does not move.

| Classification | 
| -------------- | 
|   Wet floor    | 
|   Dry floor    | 
| Human in frame |
|  Mopped floor  |

Lepton 3.5 thermal imaging sensor delievers an image in **120 x 160 (height x width)** resolution. A pixel data is of type **unsigned-integer-16bit(uint16)**,
each pixel data ranges from 0 to 65535 which represents temperature. 

### Prediction Model

Tensorflow image recognition model consist of 6 layers

|        | Conv2D | MaxPool2D | Conv2D | MaxPool2D | Flatten | Dense |
|--------|--------|-----------|--------|-----------|---------|-------|
| Filter |    6   |     -     |   3    |     -     |    -    |   -   |
| Kernel |  3 x 3 |     -     | 3 x 3  |     -     |    -    |   -   |
|PoolSize|    -   |   2 x 2   |    -   |   2 x 2   |    -    |   -   |
| Node   |    -   |     -     |    -   |     -     |    -    |   4   |

NOTE: The Python file used will be uploaded later on, I'll need to tidy things up.

The model input is **30 x 40 pixels** in resolution, this is the most optimal value memory wise as well as accuracy wise.

Original Tensorflow library uses too much memory and is computationally too complex to use in an embedded system.
Tensorflow Lite compresses the orignal model down to make operations in embedded systems possible.
List of the supported boards:

+Arduino Nano 33 BLE Sense
+SparkFun Edge
+STM32F746 Discovery kit
+Adafruit EdgeBadge
+Adafruit TensorFlow Lite for Microcontrollers Kit
+Adafruit Circuit Playground Bluefruit
+Espressif ESP32-DevKitC
+Espressif ESP-EYE
+Wio Terminal: ATSAMD51
+Himax WE-I Plus EVB Endpoint AI Development Board
+Synopsys DesignWare ARC EM Software Development Platform
+Sony Spresense



