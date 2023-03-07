#define STB_IMAGE_RESIZE_IMPLEMENTATION

#include "floor_wetness_v13/floor_wetness_v13.h"
#include "floor_wetness_v13/floor_wetness_v13_settings.h"

#include "tflite_task.h"
#include <math.h>
#include "sys_utilities.h"
#include "json_utilities.h"
#include "vospi.h"
#include "mbedtls/base64.h"
#include "stb_image_resize.h"
#include <stdint.h>
#include <iostream>
#include "nvs_flash.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"

#include "esp_http_client.h"
#include "esp_log.h"
#include "esp_tls.h"
#include "esp_crt_bundle.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// #include "esp_firebase/app.h"
// #include "esp_firebase/rtdb.h"

#define DRY_STATE 0
#define WET_STATE 1
#define HUMAN_STATE 2
#define WIPED_STATE 3

#define FIREBASE_API_KEY "AIzaSyDx6mkDELTtqeRfubqzoTYSl40sRMnHqco"
#define FIREBASE_DATABASE_URL "https://tcamimagedatabase-default-rtdb.asia-southeast1.firebasedatabase.app/"
#define FIREBASE_USER_EMAIL "test_esp32@gmail.com"
#define FIREBASE_USER_PASSWORD "1234"

#define DL_IMAGE_MIN(A, B) ((A) < (B) ? (A) : (B))
#define DL_IMAGE_MAX(A, B) ((A) < (B) ? (B) : (A))

#define NULL_PREDICTION_STATE 99

// #define DEBUG_PRINT_BASE64
// #define DEBUG_PRINT_HEAP

bool is_init = false;

const char *TAG = "tflite_task";
const int LEPTON_IMAGE_SIZE_BYTE = 120 * 160 * 2;
const int RESIZED_IMAGE_SIZE_BYTE = 30 * 40 * 2;
const int PREDICT_INTERVAL = 60; // Time waiting til the next prediction in second
const uint8_t STATE_WINDOW_SIZE = 10;

const tflite::Model *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input = nullptr;
const int TENSOR_AREANA_SIZE_BYTE = 33 * 1024;
static uint8_t *tensor_arena;
// TfLiteTensor *output = nullptr;
uint8_t state_index = 0;
uint8_t state_buffer[STATE_WINDOW_SIZE];
TfLiteStatus prediction_status;
int32_t element_count = 0;

uint16_t *resized_img;
uint16_t *img_buffer;
unsigned char *base64_image = nullptr;
size_t base64_obj_len;

extern const char google_sheet_cert_pem_start[] asm("_binary_google_sheet_cert_pem_start");
extern const char google_sheet_cert_pem_end[] asm("_binary_google_sheet_cert_pem_end");

static esp_err_t _http_event_handler(esp_http_client_event_t *evt);
static uint8_t _get_final_result_probability_index(uint8_t *state_buffer, uint8_t state_buffer_size);
static void _normalise_image_uint16(float *dest_img, uint16_t *src_img, int src_width, int src_height, int src_channel);
static void _monitor_heap();

bool tflite_init()
{
    /*
    Initialisation steps:
    1. Get the model, an really long array of hexadecimal.
    2. Allocate the tensor_arena(an area in memory where all tensorflow operations will be executed)
    3. Create an interpreter
    */
    model = tflite::GetModel(floor_wetness_v13_tflite); // Here we retrive the model array in a separate file

    resized_img = (uint16_t *)heap_caps_malloc(RESIZED_IMAGE_SIZE_BYTE, MALLOC_CAP_SPIRAM);
    img_buffer = (uint16_t *)heap_caps_malloc(LEPTON_IMAGE_SIZE_BYTE, MALLOC_CAP_SPIRAM);

    if (model->version() != TFLITE_SCHEMA_VERSION)
    {
        ESP_LOGE(TAG, "Model provided is of version %d but TFLite used is of version %d", model->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }

    tensor_arena = (uint8_t *)heap_caps_malloc(TENSOR_AREANA_SIZE_BYTE, MALLOC_CAP_SPIRAM); // heap_caps_malloc(size, MALLOC_CAP_8BIT) is the same as malloc(size * 1)

    if (tensor_arena == NULL)
    {
        ESP_LOGE(TAG, "Couldn't allocate memory for Tensorflow Lite, required %d bytes", TENSOR_AREANA_SIZE_BYTE);
        ESP_LOGE(TAG, "Available: %d bytes", heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM));
        return false;
    }

    /*
    MicroMutableOpResolver resolves or does calculations from each layer used in the model.
    For example, if the model has just 2 Dense() layers in Python we just have to add AddFullyConnected()
    since it just has to resolve only Dense() type layers.
    */
    static tflite::MicroMutableOpResolver<6> micro_op_resolver;
    micro_op_resolver.AddConv2D();
    micro_op_resolver.AddMaxPool2D();
    micro_op_resolver.AddReshape();
    micro_op_resolver.AddSoftmax();
    micro_op_resolver.AddFullyConnected();
    micro_op_resolver.AddLogistic();

    // Makes an interpreter for the "model" with required "resolvers" that points to the start of tensor arena with the size
    static tflite::MicroInterpreter static_interpreter(model, micro_op_resolver, tensor_arena, TENSOR_AREANA_SIZE_BYTE); // Constructor object creation

    // This is for ease of use
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        ESP_LOGE(TAG, "Tensor allocation failed");
        return false;
    }
    input = interpreter->input(0);
    ESP_LOGI(TAG, "Floor model v.%d", MODEL_VERSION);
    ESP_LOGI(TAG, "Tensorflow Lite initiation successful");
    ESP_LOGI(TAG, "Memory available: %d bytes", heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM));
    ESP_LOGI(TAG, "Tensor size: %d bytes", interpreter->arena_used_bytes());
    ESP_LOGI(TAG, "Tensor dimension: %d", interpreter->input(0)->dims->size);

    ESP_LOGI(TAG, "Initialising state_buffer");
    for (int i = 0; i < (STATE_WINDOW_SIZE); i++)
    {
        state_buffer[i] = NULL_PREDICTION_STATE;
    }
    is_init = true;
    return true;
}

void tflite_task()
{

    if (is_init)
    {
        ESP_LOGI(TAG, "TFLite task initialised");
        ESP_LOGI(TAG, "Starting prediction cycles with interval of %d s", PREDICT_INTERVAL);
        while (true)
        {
            // TODO: Move Prediction function here
            // xTaskNotifyWait(1,1,0x10,portMAX_DELAY);
            ESP_LOGI(TAG, "Predicting\n");
            predict_image_from_buffer(0);
        }
    }
    else
    {
        ESP_LOGE(TAG, "Abort TFLite task");
        vTaskDelete(NULL);
    }
}
/*
UPDATE 3/3/2023: predict_image_from_buffer(int image_number) is now statically allocated.
                 It is memory safe. Memory is no longer reallocated in every execution,
                 instead they have their own global buffer.
                 See buffers above.
*/
int predict_image_from_buffer(int image_number)
{
    ESP_LOGI(TAG, "Start predict image %d", image_number);

    uint16_t *rsp_buffer_pointer = ((lep_buffer_t *)&rsp_lep_buffer[image_number])->lep_bufferP;
    const int PIXEL_COUNT = 160 * 120;

    // Atomically get the pixel data of the selected image in the buffer.
    xSemaphoreTake(rsp_lep_buffer[image_number].lep_mutex, portMAX_DELAY); // Get the pixel data from buffer, pasuing any write operation to prevent data corruption

    for (int pixel = 0; pixel < PIXEL_COUNT; pixel++)
    {
        img_buffer[pixel] = rsp_buffer_pointer[pixel];
    }

    xSemaphoreGive(rsp_lep_buffer[image_number].lep_mutex); // Release semaphore back to the OS.
    stbir_resize_uint16_generic(img_buffer, 160, 120, 0, resized_img, IMAGE_WIDTH, IMAGE_HEIGHT, 0, IMAGE_CHANNEL, -1, 0,
                                STBIR_EDGE_CLAMP, STBIR_FILTER_DEFAULT, STBIR_COLORSPACE_LINEAR, NULL);

    _normalise_image_uint16(input->data.f, resized_img, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL);

    if (interpreter->Invoke() != kTfLiteOk)
    {
        ESP_LOGE(TAG, "Cannot make a prediction, Invoke fails");
    }
    else
    {
#ifdef DEBUG_PRINT_HEAP
        _monitor_heap();
#endif
        TfLiteTensor *output = interpreter->output(0);
        uint8_t max_probability_index = 0;
        for (int i = 0; i < LABEL_COUNT; i++)
        {
            // ESP_LOGI(TAG, "%d : %f", i, output->data.f[i]);
            if (output->data.f[i] > output->data.f[max_probability_index])
            {
                max_probability_index = i;
            }
        }

        // Get JSON image base64 encoded string, this is sent to TCP socket.
        mbedtls_base64_encode(base64_image, 0, &base64_obj_len, (uint8_t *)resized_img, IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(uint16_t));
        base64_image = (unsigned char *)heap_caps_malloc(base64_obj_len, MALLOC_CAP_SPIRAM);
        mbedtls_base64_encode(base64_image, base64_obj_len, &base64_obj_len, (uint8_t *)resized_img, IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(uint16_t));

#ifdef DEBUG_PRINT_BASE64
        ESP_LOGI(TAG, "Base64 uint16\n%s\n", base64_image);
#endif

        if (state_index < (STATE_WINDOW_SIZE - 1))
        {
            state_buffer[state_index] = (uint8_t)max_probability_index;
            sys_image_rsp_buffer.length = json_get_prediction_file_string(sys_image_rsp_buffer.bufferP + 1,
                                                                          &rsp_lep_buffer[image_number], output->data.f,
                                                                          (uint8_t *)resized_img, state_buffer, state_index);
            ESP_LOGI(TAG, "Filling state [%d]", state_index);
            ESP_LOGI(TAG, "_______._______._______._______");
            ESP_LOGI(TAG, "%7s|%7s|%7s|%7s", "dry", "wet", "human", "wiped");
            ESP_LOGI(TAG, "%.05f|%.05f|%.05f|%.05f", output->data.f[0], output->data.f[1], output->data.f[2], output->data.f[3]);
            ESP_LOGI(TAG, "_______|_______|_______|_______");
            ESP_LOGI(TAG, "Result = [%s]", floor_label_list[max_probability_index]);
            state_index++;
        }
        else
        {
            state_buffer[state_index] = (uint8_t)max_probability_index;
            sys_image_rsp_buffer.length = json_get_prediction_file_string(sys_image_rsp_buffer.bufferP + 1,
                                                                          &rsp_lep_buffer[image_number], output->data.f,
                                                                          (uint8_t *)resized_img, state_buffer, state_index);
            ESP_LOGI(TAG, "Final Result = --------<<<<%s>>>>--------", floor_label_list[_get_final_result_probability_index(state_buffer, STATE_WINDOW_SIZE)]);
            state_index = 0;
        }
        free(base64_image);
    }

    // free(img_buffer);
    // free(resized_img);
    ESP_LOGI(TAG, "Exiting prediction function");
    return sys_image_rsp_buffer.length;
}

int gather_images_to_cloud(int image_number)
{
    ESP_LOGI(TAG, "Sending image to Firebase\n");
    // Steps
    // 1. Check the last img on firebase and get its index
    // 2. Encond image buffer into Base64
    // 3. Make a JSON object
    // 4. Upload to the cloud
    // 5. Free memory
    const int PIXEL_COUNT = 30 * 40;
    int pixel_max = 0, pixel_min = 65535, pixel_delta = 0;
    uint16_t *rsp_buffer_pointer = ((lep_buffer_t *)&rsp_lep_buffer[image_number])->lep_bufferP;

    xSemaphoreTake(rsp_lep_buffer[image_number].lep_mutex, portMAX_DELAY); // Get the pixel data from buffer

    for (int pixel = 0; pixel < PIXEL_COUNT; pixel++)
    {
        img_buffer[pixel] = rsp_buffer_pointer[pixel];
    }

    xSemaphoreGive(rsp_lep_buffer[image_number].lep_mutex); // Release semaphore back to the OS.
    ESP_LOGI(TAG, "Max: %d Min: %d Delta: %d\n", pixel_max, pixel_min, pixel_delta);
    stbir_resize_uint16_generic(img_buffer, 160, 120, 0, resized_img, 40, 30, 0, 1, -1, 0,
                                STBIR_EDGE_CLAMP, STBIR_FILTER_DEFAULT, STBIR_COLORSPACE_LINEAR, NULL);
    unsigned char *base64_image = nullptr;
    size_t base64_obj_len;
    mbedtls_base64_encode(base64_image, 0, &base64_obj_len, (uint8_t *)resized_img, PIXEL_COUNT * sizeof(uint16_t));
    // ESP_LOGI(TAG,"%d\n", base64_obj_len);
    base64_image = (unsigned char *)heap_caps_malloc(base64_obj_len, MALLOC_CAP_8BIT);
    mbedtls_base64_encode(base64_image, base64_obj_len, &base64_obj_len, (uint8_t *)resized_img, PIXEL_COUNT * sizeof(uint16_t));

    std::string base64_image_string;
    base64_image_string.assign(resized_img, resized_img + base64_obj_len);

    return 0;
}

int gather_images_to_sheet(int image_number, int image_start, int image_stop)
{

    const int url_string_offset = 152;
    const int image_chunk_size = 51200 / 8;
    const int http_client_buffer_size = image_chunk_size + url_string_offset + 1;
    const int PIXEL_COUNT = 120 * 160;
    unsigned char *base64_image = nullptr;
    size_t base64_obj_len;
    int base64_image_index = 0;

    char *google_sheet_url = (char *)heap_caps_calloc(1, http_client_buffer_size, MALLOC_CAP_SPIRAM);
    uint16_t *image_buffer = (uint16_t *)heap_caps_malloc(PIXEL_COUNT * sizeof(uint16_t), MALLOC_CAP_SPIRAM);
    uint16_t *rsp_buffer_pointer = ((lep_buffer_t *)&rsp_lep_buffer[image_number])->lep_bufferP;

    ESP_LOGI(TAG, "Gathering images");

    xSemaphoreTake(rsp_lep_buffer[image_number].lep_mutex, portMAX_DELAY); // Get the pixel data from buffer
    for (int pixel = 0; pixel < PIXEL_COUNT; pixel++)
    {
        image_buffer[pixel] = rsp_buffer_pointer[pixel];
    }
    xSemaphoreGive(rsp_lep_buffer[image_number].lep_mutex); // Release semaphore back to the OS

    mbedtls_base64_encode(base64_image, 0, &base64_obj_len, (uint8_t *)image_buffer, PIXEL_COUNT * sizeof(uint16_t));
    base64_image = (unsigned char *)heap_caps_malloc(base64_obj_len, MALLOC_CAP_SPIRAM);
    mbedtls_base64_encode(base64_image, base64_obj_len, &base64_obj_len, (uint8_t *)image_buffer, PIXEL_COUNT * sizeof(uint16_t));

    sprintf(google_sheet_url,
            "https://script.google.com/macros/s/AKfycbw7p9wUdnDZS7dG61w8I58_gbVVSErB_IeU9Kwd2wkfBXr7yVti4tf2vLET-1jDnJEm0g/exec?image_name=image_%05d&base64_string=",
            image_start);

#ifdef DEBUG_PRINT_HEAP
    _monitor_heap();
#endif
    esp_http_client_config_t client_config = {
        .url = google_sheet_url,
        .cert_pem = google_sheet_cert_pem_start,
        .method = HTTP_METHOD_GET,
        .event_handler = _http_event_handler,
        .buffer_size_tx = (1024 * 7),
    };

    esp_http_client_handle_t client = esp_http_client_init(&client_config);

    for (int i = 0; i < 8; i++)
    {
        for (int j = 0; j < image_chunk_size; j++)
        {
            google_sheet_url[j + url_string_offset] = base64_image[base64_image_index];
            base64_image_index++;
        }
        ESP_LOGI(TAG, "Sending slice: %d/8", i + 1);
        google_sheet_url[image_chunk_size + url_string_offset] = '\0';
        esp_http_client_set_url(client, google_sheet_url);
        esp_http_client_perform(client);
    }
    ESP_LOGI(TAG, "All slices sent");

    esp_http_client_cleanup(client);
    free(base64_image);
    free(image_buffer);
    free(google_sheet_url);

    return 0;
}

// TO DO: Create wrapper function, refactoring

// static int _send_image_slices(esp_http_client_handle_t client, uint16_t image_chunk_size, uint8_t url_string_offset, )

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

static void _monitor_heap()
{
    ESP_LOGI(TAG, "Int Heap free: %d / Min: %d - SPIRAM free: %d / Min %d",
             heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
             heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL),
             heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
             heap_caps_get_minimum_free_size(MALLOC_CAP_SPIRAM));
}

static uint8_t _get_final_result_probability_index(uint8_t *state_buffer, uint8_t state_window_size)
{
    uint8_t label_state_count[] = {0, 0, 0, 0};
    uint8_t max_probability_index = 0;
    for (int i = 0; i < state_window_size; i++)
    {
        switch (state_buffer[i])
        {
        case DRY_STATE:
            label_state_count[DRY_STATE]++;
            break;
        case WET_STATE:
            label_state_count[WET_STATE]++;
            break;
        case HUMAN_STATE:
            label_state_count[HUMAN_STATE]++;
            break;
        case WIPED_STATE:
            label_state_count[WIPED_STATE]++;
            break;
        default:
            ESP_LOGW(TAG, "Invalid state detected");
            break;
        }
    }
    for (int i = 0; i < LABEL_COUNT; i++)
    {
        if (label_state_count[max_probability_index] <= label_state_count[i])
        {
            max_probability_index = i;
        }
    }
    ESP_LOGI(TAG, "[%d][%d][%d][%d]", label_state_count[DRY_STATE], label_state_count[WET_STATE], label_state_count[HUMAN_STATE], label_state_count[WIPED_STATE]);
    return max_probability_index;
}

static esp_err_t _http_event_handler(esp_http_client_event_t *evt)
{
    static char *output_buffer; // Buffer to store response of http request from event handler
    static int output_len;      // Stores number of bytes read
    switch (evt->event_id)
    {
    case HTTP_EVENT_ERROR:
        ESP_LOGD(TAG, "HTTP_EVENT_ERROR");
        break;
    case HTTP_EVENT_ON_CONNECTED:
        ESP_LOGD(TAG, "HTTP_EVENT_ON_CONNECTED");
        break;
    case HTTP_EVENT_HEADER_SENT:
        ESP_LOGD(TAG, "HTTP_EVENT_HEADER_SENT");
        break;
    case HTTP_EVENT_ON_HEADER:
        ESP_LOGD(TAG, "HTTP_EVENT_ON_HEADER, key=%s, value=%s", evt->header_key, evt->header_value);
        break;
    case HTTP_EVENT_ON_DATA:
        ESP_LOGD(TAG, "HTTP_EVENT_ON_DATA, len=%d", evt->data_len);
        /*
         *  Check for chunked encoding is added as the URL for chunked encoding used in this example returns binary data.
         *  However, event handler can also be used in case chunked encoding is used.
         */
        if (!esp_http_client_is_chunked_response(evt->client))
        {
            // If user_data buffer is configured, copy the response into the buffer
            if (evt->user_data)
            {
                memcpy(evt->user_data + output_len, evt->data, evt->data_len);
            }
            else
            {
                if (output_buffer == NULL)
                {
                    output_buffer = (char *)malloc(esp_http_client_get_content_length(evt->client));
                    output_len = 0;
                    if (output_buffer == NULL)
                    {
                        ESP_LOGE(TAG, "Failed to allocate memory for output buffer");
                        return ESP_FAIL;
                    }
                }
                memcpy(output_buffer + output_len, evt->data, evt->data_len);
            }
            output_len += evt->data_len;
        }

        break;
    case HTTP_EVENT_ON_FINISH:
        ESP_LOGD(TAG, "HTTP_EVENT_ON_FINISH");
        if (output_buffer != NULL)
        {
            // Response is accumulated in output_buffer. Uncomment the below line to print the accumulated response
            // ESP_LOG_BUFFER_HEX(TAG, output_buffer, output_len);
            free(output_buffer);
            output_buffer = NULL;
        }
        output_len = 0;
        break;
    case HTTP_EVENT_DISCONNECTED:
        ESP_LOGI(TAG, "HTTP_EVENT_DISCONNECTED");
        int mbedtls_err = 0;
        esp_err_t err = esp_tls_get_and_clear_last_error((esp_tls_error_handle_t)evt->data, &mbedtls_err, NULL);
        if (err != 0)
        {
            ESP_LOGI(TAG, "Last esp error code: 0x%x", err);
            ESP_LOGI(TAG, "Last mbedtls failure: 0x%x", mbedtls_err);
        }
        if (output_buffer != NULL)
        {
            free(output_buffer);
            output_buffer = NULL;
        }
        output_len = 0;
        break;
    }
    return ESP_OK;
}
