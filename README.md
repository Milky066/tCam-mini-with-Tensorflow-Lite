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

## Prediction Model

The Tensorflow image recognition model that we are using(version 13) consist of 6 layers

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

+ Arduino Nano 33 BLE Sense
+ SparkFun Edge
+ STM32F746 Discovery kit
+ Adafruit EdgeBadge
+ Adafruit TensorFlow Lite for Microcontrollers Kit
+ Adafruit Circuit Playground Bluefruit
+ Espressif ESP32-DevKitC
+ Espressif ESP-EYE
+ Wio Terminal: ATSAMD51
+ Himax WE-I Plus EVB Endpoint AI Development Board
+ Synopsys DesignWare ARC EM Software Development Platform
+ Sony Spresense


**Steps for prediction**

1. First, we have to initialise the interpreter, this interpreter object has to last for the entire program lifetime if we are to call Invoke() to make prediction later on.

Declare all the variables
```
const tflite::Model *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input = nullptr;
const int TENSOR_AREANA_SIZE_BYTE = 33 * 1024;
static uint8_t *tensor_arena;
TfLiteTensor *output = nullptr;

```
Get the model and make the interpreter
```
model = tflite::GetModel(floor_wetness_v13_tflite);

static tflite::MicroMutableOpResolver<6> micro_op_resolver;
micro_op_resolver.AddConv2D();
micro_op_resolver.AddMaxPool2D();
micro_op_resolver.AddReshape();
micro_op_resolver.AddSoftmax();
micro_op_resolver.AddFullyConnected();
micro_op_resolver.AddLogistic();
    
static tflite::MicroInterpreter static_interpreter(model, micro_op_resolver, tensor_arena, TENSOR_AREANA_SIZE_BYTE);
interpreter = &static_interpreter;
interpreter->AllocateTensors();

input = interpreter->input(0);

```

NOTE: For TENSOR_AREANA_SIZE_BYTE, it depends on the model used, once we have allocated the tensors we should  
call **"interpreter->arena_used_bytes()"** to read the    actual size.

The reason why "input = interpreter->input(0)" is due to the model using just one tensor per prediction, if the model was to use more than one, we should assign more 
to input(1), input(2) and so on.


2. The model was trained with **normalised float32** representation of the same data, before we feed an image into the model we have to nomalise the pixel data. Here is the method used for normalising later on.
```
static void _normalise_image_uint16(float *dest_img, uint16_t *src_img, int src_width, int src_height, int src_channel)
{
    const int PIXEL_COUNT = src_width * src_height * src_channel;
    float pixel_max = 0, pixel_min = 65535, pixel_delta = 0;
    for (int pixel = 0; pixel < PIXEL_COUNT; pixel++)
    {
        pixel_max = (src_img[pixel] > pixel_max) ? src_img[pixel] : pixel_max;
        pixel_min = (src_img[pixel] < pixel_min) ? src_img[pixel] : pixel_min;
    }
    pixel_delta = pixel_max - pixel_min;

    for (int pixel = 0; pixel < PIXEL_COUNT; pixel++)
    {
        dest_img[pixel] = ((float)src_img[pixel] - pixel_min) / pixel_delta; // Change this to lookup table later on
    }
}
```
3. Now it's time to feed the data into the model. If you're familiar with Tensorflow or PyTorch, when we actually feed in a **Tensor** or an **Array** but, for microcontroller, the model input is converted to FlatBuffer format which is just a long one dimensional array of pixel data in our case, it has no sense of **shape** so we have to be careful when feeding the image.

Example:  
+ 4 x 4 pixels grascale image 2D Tensor has the shape of (4, 4, 1) the last being its channel, the values would look like this  
[  
[0.2342] [0.43534] [0.14123] [0.64443]  
[0.5234] [0.64532] [0.32113] [0.68947]  
[0.3242] [0.54545] [0.99504] [0.36684]  
[0.8628] [0.84782] [0.74638] [0.94773]  
]  

+ For FlatBuffer, it is instead 16 in size 1D array(or Tensor)  
[0.2342] [0.43534] [0.14123] [0.64443] ...... [0.84782] [0.74638] [0.94773]  

Allocate some memory for storing image data.  
```
const int PIXEL_COUNT = 120 * 160;
uint16_t *image_buffer = (uint16_t *)heap_caps_malloc(PIXEL_COUNT * sizeof(uint16_t), MALLOC_CAP_SPIRAM);
uint16_t *resized_imgage = (uint16_t *)heap_caps_malloc(RESIZED_IMAGE_SIZE_BYTE, MALLOC_CAP_SPIRAM);
uint16_t *rsp_buffer_pointer = ((lep_buffer_t *)&rsp_lep_buffer[image_number])->lep_bufferP;
```
Get the pixel data from Lepton buffer
```
xSemaphoreTake(rsp_lep_buffer[image_number].lep_mutex, portMAX_DELAY); 
    for (int pixel = 0; pixel < PIXEL_COUNT; pixel++)
    {
        image_buffer[pixel] = rsp_buffer_pointer[pixel];
    }
xSemaphoreGive(rsp_lep_buffer[image_number].lep_mutex);
```
If you're using **embedded image**, replace "." file extension with "_"  
For example, "image1.jpg" to "_binary_image1_jpg_start"
```
extern const char image_start[] asm("_binary_{YOUR IMAGE NAME}_start");
extern const char image_end[] asm("_binary_{YOUR IMAGE NAME}_end");
```
Now we have to resize from 120 x 160 to 30 x 40  
I am using **stbir library** to resize the image.  
All the thanks to **Jorge L Rodriguez**
[Here](https://github.com/nothings/stb/blob/master/stb_image_resize.h) you can find the files

Resizing and Normalising the images
```
stbir_resize_uint16_generic(img_buffer, 160, 120, 0, resized_img, IMAGE_WIDTH, IMAGE_HEIGHT, 0, IMAGE_CHANNEL, -1, 0,
                                STBIR_EDGE_CLAMP, STBIR_FILTER_DEFAULT, STBIR_COLORSPACE_LINEAR, NULL);
_normalise_image_uint16(input->data.f, resized_image, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL);
```

4. Feeding in the pixel data, this is **inside the normalising method**
```
const int RESIZED_PIXEL_COUNT = 30 * 40;
for(int i = 0; i < RESIZED_PIXEL_COUNT; i++){
    input->data.f[i] = resized_image[i]
}
```
5. Invoke the model, do the prediction.
```
interpreter->Invoke();
```
6. Get the output
```
TfLiteTensor *output = interpreter->output(0);
uint8_t max_probability_index = 0;
for (int i = 0; i < LABEL_COUNT; i++)
   {
    if (output->data.f[i] > output->data.f[max_probability_index])
       {
           max_probability_index = i;
        }
   }
printf("Label predicted: %d with probability of %.2f", max_probability_index, output->data.f[max_probability_index]);
```
7. Free the memory allocated for images
```
free(resized_image);
free(image_buffer);
```

In progress...





