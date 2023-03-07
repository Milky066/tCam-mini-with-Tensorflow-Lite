#ifndef _TFLITE_TASK_H
#define _TFLITE_TASK_H

#ifdef __cplusplus

#include <stdint.h>
extern "C"
{
#endif
    bool tflite_init();
    void tflite_task();
    int predict_image_from_buffer(int imageNumber);
    int gather_images_to_cloud(int imageNumber);
    int gather_images_to_sheet(int image_number, int image_start, int image_stop);

#ifdef __cplusplus
}
#endif

#endif