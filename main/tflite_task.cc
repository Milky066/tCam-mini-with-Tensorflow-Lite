#include "flower_model_settings.h"
#include "flower_model_data.h"
#include "tflite_task.h"
#include <math.h>
#include "sys_utilities.h"
#include "json_utilities.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"

#include "esp_log.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#define CUSTOM_FLOWER_MODEL 0
#define CUSTOM_FACE_MODEL 1
#define CUSTOM_FLOOR_WETNESS_MODEL 2

#define DL_IMAGE_MIN(A, B) ((A) < (B) ? (A) : (B))
#define DL_IMAGE_MAX(A, B) ((A) < (B) ? (B) : (A))

const char *TAG = "tflite_task";
const uint16_t leptonImageSizeByte = 120 * 160 * 2;

const tflite::Model *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input = nullptr;
const int tensorArenaSize = 21 * 1024;
static uint8_t *tensor_arena;

void image_zoom_in_twice(uint8_t *dimage, int dw, int dh, int dc, uint8_t *simage, int sw, int sc);
void image_resize_linear(uint8_t *dst_image, uint8_t *src_image, int dst_w, int dst_h, int dst_c, int src_w, int src_h);
void normalise_image_buffer_uint8(float *dest_image_buffer, uint8_t *imageBuffer, int size);
void normalise_image_buffer_uint16(float *dest_image_buffer, uint8_t *imageBuffer, int size);

bool tflite_init()
{
    model = tflite::GetModel(flower_model_tflite);

    if (model->version() != TFLITE_SCHEMA_VERSION)
    {
        ESP_LOGE(TAG, "Model provided is of version %d but TFLite used is of version %d", model->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }

    tensor_arena = (uint8_t *)heap_caps_malloc(tensorArenaSize, MALLOC_CAP_8BIT); /*  MALLOC_CAP_INTERNAL with OR can be experitented with */

    if (tensor_arena == NULL)
    {
        ESP_LOGE(TAG, "Couldn't allocate memory for Tensorflow Lite, required %d bytes", tensorArenaSize);
        ESP_LOGE(TAG, "Available: %d bytes", heap_caps_get_largest_free_block(MALLOC_CAP_8BIT));
        return false;
    }

    static tflite::MicroMutableOpResolver<5> micro_op_resolver;
    micro_op_resolver.AddConv2D();
    micro_op_resolver.AddMaxPool2D();
    micro_op_resolver.AddReshape();
    micro_op_resolver.AddSoftmax();
    micro_op_resolver.AddFullyConnected();

    static tflite::MicroInterpreter static_interpreter(model, micro_op_resolver, tensor_arena, tensorArenaSize);

    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        ESP_LOGE(TAG, "Tensor allocation failed");
        return false;
    }
    input = interpreter->input(0);
    ESP_LOGI(TAG, "Tensorflow Lite initiation successful");
    ESP_LOGI(TAG, "Memory available: %d bytes", heap_caps_get_largest_free_block(MALLOC_CAP_8BIT));

    return true;
}

void tflite_task()
{

    if (tflite_init())
    {
        while (1)
        {
            vTaskDelay(pdMS_TO_TICKS(3000));

            tflite_predict();
            vTaskDelay(pdMS_TO_TICKS(3000));
            ESP_LOGI(TAG, "Deleteing the test task");
            vTaskDelete(NULL);
        }
    }
    else
    {
        ESP_LOGE(TAG, "Abort TFLite task");
        vTaskDelete(NULL);
    }
}

void tflite_predict()
{
    if (interpreter != nullptr)
    {
        extern const unsigned char imageStart[] asm("_binary_daisy_01s_rgb_start");
        extern const unsigned char imageEnd[] asm("_binary_daisy_01s_rgb_end");

        uint8_t *temp_buffer = (uint8_t *)malloc(imageSizeInByte);
        uint8_t *picture = (uint8_t *)imageStart;

        image_resize_linear(temp_buffer, picture, 32, 32, 3, 32, 32);

        normalise_image_buffer_uint8((interpreter->input(0))->data.f, temp_buffer, imageSizeInByte);

        interpreter->Invoke();

        TfLiteTensor *output = interpreter->output(0);

        free(temp_buffer);
        uint8_t max_probability_index = 0;
        printf("Data out %f\n", output->data.f[0]);
        for (int i = 0; i < labelCount; i++)
        {
            printf("%d : %f\n", i, output->data.f[i]);
            if (output->data.f[i] > output->data.f[max_probability_index])
            {
                max_probability_index = i;
            }
        }
        ESP_LOGE("Prediction", "Result: %s", labels[max_probability_index]);
    }
    else
    {
        ESP_LOGE(TAG, "Attemping to predict from uninitialised interpreter. "
                      "Run tflite_init() first or tflite_init() had failed earlier. Make sure the lifetime of tflite_init() equals to the program's");
    }
}

void predict_image_from_buffer(int imageNumber)
{

    xSemaphoreTake(rsp_lep_buffer[imageNumber].lep_mutex, portMAX_DELAY); // Get the pixel data from buffer
    sys_image_rsp_buffer.length = json_get_image_file_string(sys_image_rsp_buffer.bufferP + 1, &rsp_lep_buffer[imageNumber]);
    //  The operation above loads picture from lepton buffer into system image buffer
    // printf("Length: %d \n", sys_image_rsp_buffer.length);
    xSemaphoreGive(rsp_lep_buffer[imageNumber].lep_mutex);
    // uint8_t *temp_buffer = (uint8_t *)malloc(imageSizeInByte);

    normalise_image_buffer_uint16((interpreter->input(0)->data.f), (uint8_t *)(sys_image_rsp_buffer.bufferP), imageSizeInByte);
    interpreter->Invoke();
    TfLiteTensor *output = interpreter->output(0);

    // free(temp_buffer);
    uint8_t max_probability_index = 0;
    printf("Data out %f\n", output->data.f[0]);
    for (int i = 0; i < labelCount; i++)
    {
        printf("%d : %f\n", i, output->data.f[i]);
        if (output->data.f[i] > output->data.f[max_probability_index])
        {
            max_probability_index = i;
        }
    }
    ESP_LOGI(TAG, "sys_image_rsp_buffer pointer %p", sys_image_rsp_buffer.bufferP);
    ESP_LOGI(TAG, "Result: %s", labels[max_probability_index]);
}

void normalise_image_buffer_uint8(float *dest_image_buffer, uint8_t *imageBuffer, int size)
{

    printf("Normalising image buffer\n");
    for (int i = 0; i < size; i++)
    {
        // dest_image_buffer[i] = imageBuffer->data[i]/255.0f;
        // dest_image_buffer[i] = get_normalised_value(imageBuffer[i]);
        dest_image_buffer[i] = (imageBuffer[i] / 255.0f); // Change to look up table
    }
}

void image_zoom_in_twice(uint8_t *dimage,
                         int dw,
                         int dh,
                         int dc,
                         uint8_t *simage,
                         int sw,
                         int sc)
{
    for (int dyi = 0; dyi < dh; dyi++)
    {
        int _di = dyi * dw;

        int _si0 = dyi * 2 * sw;
        int _si1 = _si0 + sw;

        for (int dxi = 0; dxi < dw; dxi++)
        {
            int di = (_di + dxi) * dc;
            int si0 = (_si0 + dxi * 2) * sc;
            int si1 = (_si1 + dxi * 2) * sc;

            if (1 == dc)
            {
                dimage[di] = (uint8_t)((simage[si0] + simage[si0 + 1] + simage[si1] + simage[si1 + 1]) >> 2);
            }
            else if (3 == dc)
            {
                dimage[di] = (uint8_t)((simage[si0] + simage[si0 + 3] + simage[si1] + simage[si1 + 3]) >> 2);
                dimage[di + 1] = (uint8_t)((simage[si0 + 1] + simage[si0 + 4] + simage[si1 + 1] + simage[si1 + 4]) >> 2);
                dimage[di + 2] = (uint8_t)((simage[si0 + 2] + simage[si0 + 5] + simage[si1 + 2] + simage[si1 + 5]) >> 2);
            }
            else
            {
                for (int dci = 0; dci < dc; dci++)
                {
                    dimage[di + dci] = (uint8_t)((simage[si0 + dci] + simage[si0 + 3 + dci] + simage[si1 + dci] + simage[si1 + 3 + dci] + 2) >> 2);
                }
            }
        }
    }
    return;
}

void normalise_image_buffer_uint16(float *dest_image_buffer, uint8_t *imageBuffer, int size)
{
    uint16_t temp_pixel_value = 0;
    for (int i = 0; i < size; i++)
    {
        // dest_image_buffer[i] = get_normalised_value(imageBuffer[i]);
        // 9/2/2023: Change to lookup table later on
        // Convert 2 uint8 values inside the image buffer into 1 uint16 value by shifting the 2nd element by 8 bit
        // image buffer uses little endian
        temp_pixel_value = imageBuffer[i] + (imageBuffer[i + 1] << 8);
        dest_image_buffer[i] = (temp_pixel_value / 65535.0f); // Change to look up table
    }
}

void image_resize_linear(uint8_t *dst_image, uint8_t *src_image, int dst_w, int dst_h, int dst_c, int src_w, int src_h)
{
    float scale_x = (float)src_w / dst_w;
    float scale_y = (float)src_h / dst_h;

    int dst_stride = dst_c * dst_w;
    int src_stride = dst_c * src_w;

    if (fabs(scale_x - 2) <= 1e-6 && fabs(scale_y - 2) <= 1e-6)
    {
        image_zoom_in_twice(
            dst_image,
            dst_w,
            dst_h,
            dst_c,
            src_image,
            src_w,
            dst_c);
    }
    else
    {
        for (int y = 0; y < dst_h; y++)
        {
            float fy[2];
            fy[0] = (float)((y + 0.5) * scale_y - 0.5); // y
            int src_y = (int)fy[0];                     // y1
            fy[0] -= src_y;                             // y - y1
            fy[1] = 1 - fy[0];                          // y2 - y
            src_y = DL_IMAGE_MAX(0, src_y);
            src_y = DL_IMAGE_MIN(src_y, src_h - 2);

            for (int x = 0; x < dst_w; x++)
            {
                float fx[2];
                fx[0] = (float)((x + 0.5) * scale_x - 0.5); // x
                int src_x = (int)fx[0];                     // x1
                fx[0] -= src_x;                             // x - x1
                if (src_x < 0)
                {
                    fx[0] = 0;
                    src_x = 0;
                }
                if (src_x > src_w - 2)
                {
                    fx[0] = 0;
                    src_x = src_w - 2;
                }
                fx[1] = 1 - fx[0]; // x2 - x

                for (int c = 0; c < dst_c; c++)
                {
                    dst_image[y * dst_stride + x * dst_c + c] = round(src_image[src_y * src_stride + src_x * dst_c + c] * fx[1] * fy[1] + src_image[src_y * src_stride + (src_x + 1) * dst_c + c] * fx[0] * fy[1] + src_image[(src_y + 1) * src_stride + src_x * dst_c + c] * fx[1] * fy[0] + src_image[(src_y + 1) * src_stride + (src_x + 1) * dst_c + c] * fx[0] * fy[0]);
                }
            }
        }
    }
}
